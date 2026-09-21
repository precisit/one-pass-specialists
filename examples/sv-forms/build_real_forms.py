"""Build a real-form evaluation set from two of our own shipped Swedish forms.

The forms are Kanslist's and Pratsam's lead forms: their field labels, roles and option lists are
copied from the shipped components (`site/src/components/LeadForm.tsx` in each repository), not
invented. The *values* are synthetic, and the entity-to-field mapping is an explicit table written
by hand in this file, which is the honest description of what this set is: a test of the real
vocabulary and structure, not a study of real submissions.

Each row is one decision in the same shape the model was trained on:

    context  = "UPPGIFT ...\\nFORM <real form title>\\nELEMENT <role> \\"<real label>\\" value=\\"\\""
    options  = ["fyll <label>: <value>", ...] + ["kryssa", "klicka", "hoppa över"]
    label    = index of the correct option
    meta     = {"form": ..., "field": ..., "action": ..., "note": "why this is the gold decision"}

Run:  python build_real_forms.py --checkpoint runs/sv-tinyx-10000/model
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onepass._vendor import ensure_vendor  # noqa: E402
from onepass.synth import TASK_LINE, Element, Entity, OPTION_ACTIONS, render_context  # noqa: E402

ensure_vendor()

# --------------------------------------------------------------------------------------------
# The two forms, transcribed from the shipped components.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """One element of a real form, as the browser renders it."""

    key: str
    role: str
    label: str
    kind: str          # "text" | "select" | "checkbox" | "radio" | "submit" | "honeypot"
    entity: str | None = None   # entity label that fills it, when kind is text/select
    fact: str | None = None     # scenario fact that decides a checkbox, when kind is checkbox
    required: bool = False
    note: str = ""


@dataclass(frozen=True)
class Form:
    key: str
    title: str
    fields: tuple[Field, ...]
    facts: tuple[str, ...] = field(default_factory=tuple)


PRATSAM = Form(
    key="pratsam",
    title="Pratsam – intresseanmälan",
    fields=(
        Field("email", "Edit", "E-postadress", "text", entity="E-post", required=True,
              note="required; the document's e-mail fills it"),
        Field("consent", "CheckBox", "Ja, ni får spara min adress och skicka e-post om Pratsam. (krävs)",
              "checkbox", fact="samtycke", required=True, note="required; check when the person consents"),
        Field("signingUpFor", "Radio", "Jag anmäler intresse för…", "radio", fact="intresse_rad",
              note="optional radio group; select when the document states who it concerns"),
        Field("submit", "Button", "Anmäl intresse", "submit",
              note="click only when the required fields are satisfied in this scenario"),
        Field("website", "Edit", "Lämna detta fält tomt", "honeypot",
              note="honeypot: a real visitor leaves it empty, so the correct decision is always skip"),
    ),
    facts=("samtycke", "intresse_rad"),
)

KANSLIST = Form(
    key="kanslist",
    title="Kanslist – intresseanmälan",
    fields=(
        Field("email", "Edit", "E-post", "text", entity="E-post", required=True,
              note="required; the document's e-mail fills it"),
        Field("consent", "CheckBox", "Ja, ni får spara mina uppgifter så att ni kan kontakta mig. (krävs)",
              "checkbox", fact="samtycke", required=True, note="required; check when the board consents"),
        Field("role", "ComboBox", "Roll i föreningen", "select", entity="Roll",
              note="optional select; the document's role is one of the form's own options"),
        Field("units", "ComboBox", "Antal lägenheter", "select", entity="Antal lägenheter",
              note="optional select; value taken from the form's own option list"),
        Field("propertyManager", "ComboBox", "Förvaltare", "select", entity="Förvaltare",
              note="optional select"),
        Field("emailProvider", "ComboBox", "E-postleverantör", "select", entity="E-postleverantör",
              note="optional select"),
        Field("documents", "ComboBox", "Dokumenthantering", "select", entity="Dokumenthantering",
              note="optional select"),
        Field("chat", "ComboBox", "Chattkanal", "select", entity="Chattkanal",
              note="optional select"),
        Field("comment", "Edit", "Något ni vill berätta?", "text", entity="Meddelande",
              note="optional free text; filled only when the document carries a message"),
        Field("wantsNewsletter", "CheckBox", "Skicka nyhetsbrev när det händer något", "checkbox",
              fact="nyhetsbrev", note="optional; check when the document asks for it"),
        Field("wantsContact", "CheckBox", "Vi vill vara med och testa", "checkbox", fact="testa",
              note="optional; check when the document offers to test"),
        Field("submit", "Button", "Skicka", "submit",
              note="click only when the required fields are satisfied in this scenario"),
        Field("website", "Edit", "Lämna detta fält tomt", "honeypot",
              note="honeypot: always skip"),
    ),
    facts=("samtycke", "nyhetsbrev", "testa"),
)

# --------------------------------------------------------------------------------------------
# Synthetic entity pool. Values are invented; the labels mirror what the forms actually ask for.
# --------------------------------------------------------------------------------------------

ENTITY_POOL: dict[str, tuple[str, ...]] = {
    "E-post": ("anna.lindqvist@exempel.invalid", "brf.björken@exempel.invalid", "kansli@exempel.invalid"),
    "Förnamn": ("Anna", "Björn", "Karin"),
    "Efternamn": ("Lindqvist", "Bergström", "Nyström"),
    "Telefon": ("070-341 22 87", "073-918 44 10"),
    "Personnummer": ("740721-3466", "880314-1201"),
    "Organisationsnummer": ("769600-3311", "556042-7106"),
    "Föreningens namn": ("BRF Björken", "BRF Eken 12", "BRF Norrgården"),
    "Adress": ("Storgatan 14", "Björkvägen 3B"),
    "Postnummer": ("753 20", "113 45"),
    "Ort": ("Uppsala", "Stockholm"),
    "Roll": ("Ordförande", "Ledamot", "Sekreterare", "Ekonomiansvarig", "Boende", "Annat"),
    "Antal lägenheter": ("Färre än 10", "10–25", "25–75", "Fler än 75", "Vet ej"),
    "Förvaltare": ("SBC", "Nabo", "ERF", "Annan", "Ingen – vi sköter själva"),
    "E-postleverantör": ("one.com", "Gmail eller Google Workspace", "Microsoft 365", "Annan"),
    "Dokumenthantering": ("Google Drive", "Microsoft", "Annat", "Vet ej"),
    "Chattkanal": ("WhatsApp", "Signal", "Telegram", "Slack", "Discord", "Ingen", "Annat"),
    "Meddelande": ("Vi vill höra mer om digitala protokoll.", "Vi har frågor om årsredovisningen."),
    "Bankgiro": ("741-4686", "523-1180"),
    "Fastighetsbeteckning": ("Björken 12:3", "Eken 4:11"),
}

FACT_TRUE_VALUES = {"samtycke": "ja", "nyhetsbrev": "ja", "testa": "ja", "intresse_rad": "mig själv"}


def scenario(rng: random.Random, form: Form, index: int) -> dict:
    """One synthetic document: a subset of entities plus the facts a checkbox depends on."""
    entities: dict[str, str] = {}
    for label, values in ENTITY_POOL.items():
        # a document rarely carries everything, which is the interesting case for a form
        if rng.random() < 0.55:
            entities[label] = rng.choice(values)
    for fact in form.facts:
        entities[fact] = FACT_TRUE_VALUES[fact] if rng.random() < 0.6 else "nej"
    return {"index": index, "entities": entities}


def rows_for(form: Form, scenarios: list[dict]) -> list[dict]:
    """Turn scenarios into decision rows, with the gold decision written by the table above."""
    rows: list[dict] = []
    for scenario in scenarios:
        entities = scenario["entities"]
        options = [Entity(label, value).option() for label, value in entities.items()] + list(OPTION_ACTIONS)
        required_ok = all(
            (f.entity is not None and f.entity in entities) or (f.fact is not None and entities.get(f.fact) == "ja")
            for f in form.fields
            if f.required
        )
        for field in form.fields:
            if field.kind in ("text", "select"):
                if field.entity in entities:
                    action, target, note = "fill", field.entity, f"{field.note}; the document carries {field.entity}"
                else:
                    action, target, note = "skip", None, f"{field.note}; the document carries no {field.entity}"
            elif field.kind in ("checkbox", "radio"):
                wants = entities.get(field.fact) == "ja"
                action, target = ("check", None) if wants else ("skip", None)
                note = f"{field.note}; fact {field.fact}={entities.get(field.fact)}"
            elif field.kind == "submit":
                action, target = ("click", None) if required_ok else ("skip", None)
                note = f"{field.note}; required fields satisfied={required_ok}"
            else:  # honeypot
                action, target, note = "skip", None, field.note
            label = options.index(Entity(target, entities[target]).option()) if action == "fill" else len(entities) + OPTION_ACTIONS.index(
                {"check": "kryssa", "click": "klicka", "skip": "hoppa över"}[action]
            )
            rows.append(
                {
                    "context": render_context(form.title, Element(field.role, field.label)),
                    "options": options,
                    "label": label,
                    "meta": {
                        "form": form.key,
                        "field": field.key,
                        "action": action,
                        "required": field.required,
                        "note": note,
                        "scenario": scenario["index"],
                    },
                }
            )
    return rows


def build(scenarios_per_form: int = 15, seed: int = 2026) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    built = {}
    for form in (KANSLIST, PRATSAM):
        scenarios = [scenario(rng, form, index) for index in range(scenarios_per_form)]
        built[form.key] = rows_for(form, scenarios)
    return built


def score(checkpoint: Path, rows: list[dict]) -> dict:
    """The same metrics the harness reports, on the real-form rows."""
    import torch

    from onepass._vendor import ensure_vendor as _ensure  # noqa: PLC0415

    _ensure()
    from cua_s1.model import ChoiceExample, load_checkpoint  # noqa: PLC0415

    model, collator, _ = load_checkpoint(checkpoint, "cpu")
    model.eval()
    examples = [ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in rows]
    with torch.no_grad():
        predicted = model(collator(examples)).argmax(-1).tolist()
    correct = sum(1 for row, guess in zip(rows, predicted) if guess == row["label"])
    per_action: dict[str, list[int]] = {}
    per_field: dict[str, list[int]] = {}
    silent = 0
    for row, guess in zip(rows, predicted):
        action = row["meta"]["action"]
        hit = int(guess == row["label"])
        bucket = per_action.setdefault(action, [0, 0])
        bucket[0] += hit
        bucket[1] += 1
        field_bucket = per_field.setdefault(row["meta"]["field"], [0, 0])
        field_bucket[0] += hit
        field_bucket[1] += 1
        if action == "fill" and row["options"][guess] == "hoppa över":
            silent += 1
    fills = per_action.get("fill", [0, 1])[1]
    return {
        "examples": len(rows),
        "top1": correct / len(rows),
        "per_action": {k: round(v[0] / v[1], 3) for k, v in sorted(per_action.items())},
        "per_field": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_field.items())},
        "silent_skips": silent,
        "silent_skip_rate_of_fills": silent / fills if fills else None,
    }


def shuffled_control(checkpoint: Path, rows: list[dict], seed: int = 7) -> dict:
    """Rotate contexts between rows: a model that reads the label should fall apart."""
    rng = random.Random(seed)
    contexts = [row["context"] for row in rows]
    rng.shuffle(contexts)
    rotated = [dict(row, context=context) for row, context in zip(rows, contexts)]
    return score(checkpoint, rotated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/sv-tinyx-10000/model"))
    parser.add_argument("--out", type=Path, default=Path("RESULTS-real-forms.json"))
    parser.add_argument("--scenarios", type=int, default=15)
    args = parser.parse_args()

    built = build(args.scenarios)
    report: dict[str, object] = {}
    for form, rows in built.items():
        path = Path(f"real-forms-{form}.jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        report[form] = score(args.checkpoint, rows)
        report[f"{form}__shuffled_context"] = shuffled_control(args.checkpoint, rows)
        print(f"{form}: {len(rows)} decisions -> {json.dumps(report[form], ensure_ascii=False)}", flush=True)
        print(f"{form} shuffled control -> top1={report[f'{form}__shuffled_context']['top1']:.3f}", flush=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
