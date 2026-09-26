//! c4label: exact scores for every legal move of many Connect Four positions.
//!
//! Input, one position per line (stdin):
//!   * a move string of 1-indexed columns, e.g. `4453`, or `-` for the empty board;
//!   * `b:` + 42 board characters, top row first, left to right: `x` = the player to move,
//!     `o` = the opponent, `.` = empty (the `Position::from_board_string` convention).
//! Output, one line per input line, same order (stdout), tab separated:
//!   input  ply  s1,s2,...,s7  micros
//! where `sK` is the exact score of playing column K for the player to move (`connect-four-ai`
//! convention: positive = win, larger = sooner; 0 = draw; negative = loss, more negative = sooner),
//! `x` for a full column. A position that is already won, full, or unparsable yields
//! `input  -1  won|full|error:<why>  0` so the caller can never silently misalign rows.
//!
//! `--threads N` sets the worker count (default: all cores). `--serve` answers line by line with a
//! flush after each (for interactive players); otherwise lines are processed in parallel chunks.
//! `--no-book` disables the embedded depth-8 opening book (for verifying the book itself).
//! `--value` outputs the position's own score (`solve`, which consults the opening book first)
//! instead of the seven child scores: `input  ply  value  micros`.
//!
//! `--gen GAMES --seed S` plays GAMES games between mixed-strength players and writes every
//! visited decision (>= 2 legal moves) in the output format above, plus a 5th column
//! `game_id:player_a/player_b`. The players pick moves *from the exact scores* (so each visited
//! position is labelled by the same solve that chose the move). Player kinds, drawn per side and
//! per game: `random` (uniform), `safe` (random, but takes an immediate win and blocks an immediate
//! loss when it can), `soft:T` (softmax over exact scores at temperature T), `eps:E` (optimal move,
//! uniformly random with probability E), `perfect` (random among the best-scoring moves).

use connect_four_ai::{Position, Solver};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, BufRead, BufWriter, Write};
use std::time::Instant;

fn parse(line: &str) -> Result<Position, String> {
    let line = line.trim();
    if line == "-" || line.is_empty() {
        return Ok(Position::new());
    }
    if let Some(board) = line.strip_prefix("b:") {
        return Position::from_board_string(board).map_err(|e| format!("{e:?}"));
    }
    Position::from_moves(line).map_err(|e| format!("{e:?}"))
}

fn value(solver: &mut Solver, line: &str) -> String {
    let input = line.trim();
    let position = match parse(input) {
        Ok(p) => p,
        Err(why) => return format!("{input}\t-1\terror:{}\t0", why.replace('\t', " ")),
    };
    if position.is_won_position() || position.get_moves() >= Position::BOARD_SIZE {
        return format!("{input}\t-1\tterminal\t0");
    }
    let started = Instant::now();
    let v = solver.solve(&position);
    format!("{input}\t{}\t{v}\t{}", position.get_moves(), started.elapsed().as_micros())
}

fn label(solver: &mut Solver, line: &str) -> String {
    let input = line.trim();
    let position = match parse(input) {
        Ok(p) => p,
        Err(why) => return format!("{input}\t-1\terror:{}\t0", why.replace('\t', " ")),
    };
    if position.is_won_position() {
        return format!("{input}\t-1\twon\t0");
    }
    if position.get_moves() >= Position::BOARD_SIZE {
        return format!("{input}\t-1\tfull\t0");
    }
    let started = Instant::now();
    let scores = solver.get_all_move_scores(&position);
    let micros = started.elapsed().as_micros();
    let cells: Vec<String> = scores
        .iter()
        .map(|s| s.map(|v| v.to_string()).unwrap_or_else(|| "x".to_string()))
        .collect();
    format!("{input}\t{}\t{}\t{micros}", position.get_moves(), cells.join(","))
}

#[derive(Clone, Copy, Debug)]
enum Kind { Random, Safe, Soft(f64), Eps(f64), Perfect }

fn kind_name(k: Kind) -> String {
    match k {
        Kind::Random => "random".into(),
        Kind::Safe => "safe".into(),
        Kind::Soft(t) => format!("soft:{t}"),
        Kind::Eps(e) => format!("eps:{e}"),
        Kind::Perfect => "perfect".into(),
    }
}

fn draw_kind(rng: &mut StdRng) -> Kind {
    // A deliberately broad mixture: humans in the demo play like `random`/`safe`/high-T players,
    // strong opponents like low-T/eps/perfect. Positions after blunders come from the weak kinds.
    match rng.random_range(0..10) {
        0 => Kind::Random,
        1 | 2 => Kind::Safe,
        3 => Kind::Soft(8.0),
        4 => Kind::Soft(4.0),
        5 => Kind::Soft(2.0),
        6 => Kind::Soft(1.0),
        7 => Kind::Eps(0.2),
        8 => Kind::Eps(0.05),
        _ => Kind::Perfect,
    }
}

fn pick(kind: Kind, scores: &[Option<i8>; 7], position: &Position, rng: &mut StdRng) -> usize {
    let legal: Vec<usize> = (0..7).filter(|&c| scores[c].is_some()).collect();
    let best = legal.iter().map(|&c| scores[c].unwrap()).max().unwrap();
    let optimal: Vec<usize> = legal.iter().copied().filter(|&c| scores[c] == Some(best)).collect();
    let uniform = |set: &Vec<usize>, rng: &mut StdRng| set[rng.random_range(0..set.len())];
    match kind {
        Kind::Random => uniform(&legal, rng),
        Kind::Safe => {
            let wins: Vec<usize> = legal.iter().copied().filter(|&c| position.is_winning_move(c)).collect();
            if !wins.is_empty() { return uniform(&wins, rng); }
            // columns that do not hand the opponent an immediate win (1-ply look-ahead only)
            let safe: Vec<usize> = legal.iter().copied().filter(|&c| {
                let mut child = *position;
                child.play(c);
                !child.can_win_next()
            }).collect();
            if safe.is_empty() { uniform(&legal, rng) } else { uniform(&safe, rng) }
        }
        Kind::Soft(t) => {
            let weights: Vec<f64> = legal.iter().map(|&c| ((scores[c].unwrap() - best) as f64 / t).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut x = rng.random::<f64>() * total;
            for (i, w) in weights.iter().enumerate() {
                x -= w;
                if x <= 0.0 { return legal[i]; }
            }
            *legal.last().unwrap()
        }
        Kind::Eps(e) => if rng.random::<f64>() < e { uniform(&legal, rng) } else { uniform(&optimal, rng) },
        Kind::Perfect => uniform(&optimal, rng),
    }
}

thread_local! {
    // Early positions repeat across generated games; their (exact) child scores are cached per
    // thread, keyed by the *unmirrored* bitboards so a mirrored position never returns mirrored
    // scores. Only positions with <= CACHE_PLY stones are cached, which bounds the memory.
    static CACHE: RefCell<HashMap<(u64, u64), [Option<i8>; 7]>> = RefCell::new(HashMap::new());
}
const CACHE_PLY: usize = 16;

fn cached_scores(solver: &mut Solver, position: &Position) -> [Option<i8>; 7] {
    if position.get_moves() > CACHE_PLY {
        return solver.get_all_move_scores(position);
    }
    let key = (position.position, position.mask);
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&key).copied()) {
        return hit;
    }
    let scores = solver.get_all_move_scores(position);
    CACHE.with(|c| c.borrow_mut().insert(key, scores));
    scores
}

fn play_game(solver: &mut Solver, game: u64, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ game);
    let kinds = [draw_kind(&mut rng), draw_kind(&mut rng)];
    let tag = format!("g{game}:{}/{}", kind_name(kinds[0]), kind_name(kinds[1]));
    let mut position = Position::new();
    let mut moves = String::new();
    let mut rows = Vec::new();
    loop {
        if position.get_moves() >= Position::BOARD_SIZE { break; }
        let started = Instant::now();
        let scores = cached_scores(solver, &position);
        let micros = started.elapsed().as_micros();
        let legal = scores.iter().filter(|s| s.is_some()).count();
        if legal >= 2 {
            let cells: Vec<String> = scores.iter().map(|s| s.map(|v| v.to_string()).unwrap_or_else(|| "x".into())).collect();
            let key = if moves.is_empty() { "-".to_string() } else { moves.clone() };
            rows.push(format!("{key}\t{}\t{}\t{micros}\t{tag}", position.get_moves(), cells.join(",")));
        }
        let column = pick(kinds[position.get_moves() % 2], &scores, &position, &mut rng);
        if position.is_winning_move(column) { break; }
        position.play(column);
        moves.push(char::from(b'1' + column as u8));
    }
    rows
}

fn make_solver(book: bool) -> Solver {
    if book { Solver::new() } else { Solver::empty() }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let serve = args.iter().any(|a| a == "--serve");
    let value_mode = args.iter().any(|a| a == "--value");
    let book = !args.iter().any(|a| a == "--no-book");
    if let Some(i) = args.iter().position(|a| a == "--threads") {
        let n: usize = args.get(i + 1).and_then(|v| v.parse().ok()).expect("--threads N");
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().unwrap();
    }
    if let Some(i) = args.iter().position(|a| a == "--gen") {
        let games: u64 = args.get(i + 1).and_then(|v| v.parse().ok()).expect("--gen GAMES");
        let seed: u64 = args.iter().position(|a| a == "--seed")
            .and_then(|j| args.get(j + 1)).and_then(|v| v.parse().ok()).expect("--seed S");
        let mut out = BufWriter::new(io::stdout().lock());
        const BLOCK: u64 = 256;
        let mut start = 0;
        while start < games {
            let end = (start + BLOCK * 64).min(games);
            let blocks: Vec<Vec<String>> = (start..end)
                .into_par_iter()
                .map_init(|| make_solver(book), |solver, g| play_game(solver, g, seed))
                .collect();
            for rows in blocks { for r in rows { writeln!(out, "{r}").unwrap(); } }
            start = end;
        }
        return;
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    if serve {
        let mut solver = make_solver(book);
        let mut out = stdout.lock();
        for line in stdin.lock().lines() {
            let line = line.expect("stdin");
            writeln!(out, "{}", label(&mut solver, &line)).unwrap();
            out.flush().unwrap();
        }
        return;
    }
    let mut out = BufWriter::new(stdout.lock());
    let mut lines = stdin.lock().lines();
    const CHUNK: usize = 1 << 14;
    loop {
        let chunk: Vec<String> = lines.by_ref().take(CHUNK).map(|l| l.expect("stdin")).collect();
        if chunk.is_empty() {
            break;
        }
        let results: Vec<String> = chunk
            .par_iter()
            .map_init(|| make_solver(book), |solver, line| if value_mode { value(solver, line) } else { label(solver, line) })
            .collect();
        for r in results {
            writeln!(out, "{r}").unwrap();
        }
    }
}
