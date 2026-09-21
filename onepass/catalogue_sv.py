"""Swedish concept catalogue for a CUA-S1-style form specialist.

Shape mirrors `cua_s1.concepts` (MIT, trycua/cua, libs/cua-s1) but the labels,
document synonyms, value formats and form titles are Swedish. Values are
synthetic: `.invalid` domains, fiction names, and personnummer-shaped values
generated locally (no real individual's data, no real personnummer).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Callable, Sequence

PersonData = dict[str, str]
ValueGenerator = Callable[[random.Random, PersonData], str]

FIRST_NAMES = (
    "Anna", "Erik", "Maria", "Lars", "Karin", "Johan", "Sara", "Anders", "Emma", "Nils",
    "Eva", "Per", "Linnéa", "Gustav", "Astrid", "Oskar", "Ingrid", "Henrik", "Elsa", "Björn",
    "Malin", "Fredrik", "Ylva", "Torbjörn", "Saga", "Elias", "Alva", "Hjalmar", "Maja", "Viktor",
    "Sofia", "Mikael", "Kristina", "Daniel", "Helena", "Stefan", "Åsa", "Magnus", "Ulrika", "Jonas",
)
LAST_NAMES = (
    "Andersson", "Johansson", "Karlsson", "Nilsson", "Eriksson", "Larsson", "Olsson", "Persson",
    "Svensson", "Gustafsson", "Pettersson", "Jonsson", "Jansson", "Hansson", "Bengtsson",
    "Lindberg", "Lindqvist", "Bergström", "Sandberg", "Holm", "Wallin", "Ekström", "Norén",
    "Åberg", "Sundqvist", "Falk", "Dahlberg", "Månsson", "Rydberg", "Sjögren",
)
STREETS = (
    "Storgatan", "Kungsgatan", "Drottninggatan", "Vasagatan", "Järnvägsgatan", "Skolgatan",
    "Kyrkogatan", "Trädgårdsgatan", "Industrivägen", "Björkvägen", "Ekvägen", "Solvägen",
    "Norra Esplanaden", "Södra Förstadsgatan", "Ringvägen", "Hantverkargatan",
)
CITIES = (
    ("Uppsala", "753 20"), ("Stockholm", "111 22"), ("Göteborg", "411 03"), ("Malmö", "211 34"),
    ("Linköping", "582 23"), ("Västerås", "722 12"), ("Örebro", "702 10"), ("Umeå", "903 26"),
    ("Lund", "222 21"), ("Jönköping", "553 20"), ("Visby", "621 56"), ("Kiruna", "981 31"),
)
COMPANIES = (
    "Nordljus Teknik AB", "Testverket AB", "Exempelgruppen AB", "Norrsken Logistik AB",
    "Provhagen Industri AB", "Exempeldata System AB", "Testfabriken AB", "Nordljus Vård AB",
    "Exempelmontage AB", "Provstaden Bygg AB", "Testkonsult i Uppsala AB", "Exempelkemi AB",
)
INSURERS = (
    "Exempelförsäkring AB", "Provtrygghet Försäkring", "Testbolaget Sakförsäkring",
    "Nordljus Försäkring", "Exempel Liv & Sak", "Provskydd AB",
)
UNIVERSITIES = (
    "Uppsala universitet", "Lunds universitet", "Chalmers tekniska högskola",
    "Kungliga Tekniska högskolan", "Umeå universitet", "Linköpings universitet",
    "Stockholms universitet",
)
JOB_TITLES = (
    "Systemutvecklare", "Sjuksköterska", "Projektledare", "Ekonomichef", "Lagerarbetare",
    "Grundskollärare", "Civilingenjör", "Redovisningsekonom", "Produktdesigner", "Undersköterska",
    "Maskinoperatör", "Verksamhetsutvecklare",
)
CAR_MAKES = (
    ("Exempelbilar AB", "Nord"), ("Testmotors", "Prov"), ("Exempelfordon", "Test"),
    ("Nordljus Bil AB", "Ljus"),
)
PROGRAMS = (
    "Ekonomprogrammet", "Sjuksköterskeprogrammet", "Civilingenjörsprogrammet i teknikfysik",
    "Systemvetenskapliga programmet", "Förskollärarprogrammet", "Maskinteknik, högskoleingenjör",
)
COURSES = (
    "Programmeringsteknik I", "Anatomi och fysiologi", "Statistik för ekonomer",
    "Maskinelement", "Organisation och ledarskap", "Databasteknik",
)
COUNTIES = ("Uppsala län", "Stockholms län", "Västra Götalands län", "Skåne län", "Västerbottens län")
DEPARTMENTS = ("Ekonomi", "Produktion", "Kundservice", "IT och digitalisering", "HR", "Logistik")
HEALTH_CENTERS = (
    "Exempelvårdcentralen", "Provhälsan Vårdcentral", "Testvården Hus 3",
    "Nordljus Vårdcentral", "Exempel Närvård",
)
DOCTORS = (
    "Dr. Lindqvist", "Leg. läkare Bergström", "Dr. Sjögren", "Leg. läkare Norén", "Dr. Månsson",
)
ALLERGIES = ("Penicillin", "Ingen känd", "Nötter", "Pollen", "Laktos", "Ägg", "Jordnötter")
BLOOD_TYPES = ("A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-")
BANKS = ("Exempelbanken", "Provsparbanken", "Testkredit AB", "Nordljus Bank")
MUNICIPALITIES = ("Uppsala kommun", "Stockholms stad", "Göteborgs stad", "Malmö stad", "Örebro kommun")


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _luhn_check(number: str) -> str:
    total = 0
    for position, digit in enumerate(reversed(number)):
        value = int(digit)
        if position % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - total % 10) % 10)


def gen_personnummer(rng: random.Random) -> str:
    """Synthetic personnummer-shaped value (correct Luhn, fictional person).

    Shape is the Swedish one: ``YYMMDD-XXXX`` (10 digits), where the last of the four is the
    Luhn check digit over the first nine. Values are locally shaped but belong to nobody.
    """
    year = rng.randint(1940, 2005)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    body = f"{year % 100:02d}{month:02d}{day:02d}{_digits(rng, 3)}"  # 9 digits: YYMMDD + 3
    return f"{body[:6]}-{body[6:]}{_luhn_check(body)}"


def gen_orgnummer(rng: random.Random) -> str:
    """Synthetic organisationsnummer: ``XXXXXX-XXXX`` (10 digits, Luhn-checked)."""
    body = f"55{_digits(rng, 7)}"  # 9 digits: prefix + 7
    return f"{body[:6]}-{body[6:]}{_luhn_check(body)}"


def gen_phone(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"07{rng.choice('0123')}-{_digits(rng, 3)} {_digits(rng, 2)} {_digits(rng, 2)}"
    return f"0{rng.randint(8, 9)}0-{_digits(rng, 3)} {_digits(rng, 2)} {_digits(rng, 2)}"


def gen_work_phone(rng: random.Random) -> str:
    return f"0{rng.randint(8, 9)}0-{_digits(rng, 3)} {_digits(rng, 2)} {_digits(rng, 2)}"


def gen_date(rng: random.Random, start: int = 2024, end: int = 2027) -> str:
    return f"{rng.randint(start, end)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def gen_money(rng: random.Random) -> str:
    whole = rng.randint(1, 480) * 100 + rng.choice((0, 50, 95, 25))
    space = f"{whole:,}".replace(",", " ")
    return f"{space},{rng.choice(('00', '50', '90'))} kr"


def gen_bankgiro(rng: random.Random) -> str:
    return f"{rng.randint(100, 999)}-{_digits(rng, 4)}"


def gen_plusgiro(rng: random.Random) -> str:
    return f"{_digits(rng, 2)} {_digits(rng, 2)} {_digits(rng, 2)}-{_digits(rng, 1)}"


def gen_iban(rng: random.Random) -> str:
    return f"SE{rng.randint(10, 99)} {_digits(rng, 4)} {_digits(rng, 4)} {_digits(rng, 4)} {_digits(rng, 4)} {_digits(rng, 2)}"


def gen_ocr(rng: random.Random) -> str:
    return _digits(rng, rng.choice((8, 9, 10)))


def gen_vat(rng: random.Random) -> str:
    return f"SE{_digits(rng, 12)}01"


def gen_registration(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNOPRSTUVWXYZ") for _ in range(3))
    return f"{letters} {_digits(rng, 3)}"


def gen_email(rng: random.Random, person: PersonData) -> str:
    local = f"{person['first']}.{person['last']}".lower()
    for source, target in (("å", "a"), ("ä", "a"), ("ö", "o"), ("é", "e")):
        local = local.replace(source, target)
    return f"{local}@exempel.invalid"


def person(rng: random.Random) -> PersonData:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, zip_code = rng.choice(CITIES)
    return {
        "first": first,
        "last": last,
        "full": f"{first} {last}",
        "city": city,
        "zip": zip_code,
        "address": f"{rng.choice(STREETS)} {rng.randint(1, 120)}",
        "employer": rng.choice(COMPANIES),
        "insurer": rng.choice(INSURERS),
        "university": rng.choice(UNIVERSITIES),
        "title": rng.choice(JOB_TITLES),
        "program": rng.choice(PROGRAMS),
        "course": rng.choice(COURSES),
        "county": rng.choice(COUNTIES),
        "department": rng.choice(DEPARTMENTS),
        "center": rng.choice(HEALTH_CENTERS),
        "doctor": rng.choice(DOCTORS),
        "bank": rng.choice(BANKS),
        "municipality": rng.choice(MUNICIPALITIES),
    }


@dataclass(frozen=True)
class Concept:
    key: str
    form_labels: tuple[str, ...]
    doc_labels: tuple[str, ...]
    value: ValueGenerator
    kind: str = "text"  # text | textarea | select
    placeholder: tuple[str, ...] = ()
    group: str = "allmant"  # concepts in the same group are plausible confusers
    options: tuple[str, ...] = ()


def _adapt(value: ValueGenerator) -> ValueGenerator:
    """Accept both ``f(rng)`` and ``f(rng, person)`` value generators."""
    import inspect

    parameters = len(inspect.signature(value).parameters)
    if parameters == 1:
        return lambda rng, _person: value(rng)  # type: ignore[arg-type]
    if parameters == 2:
        return value
    raise TypeError(f"value generator {value!r} must take 1 or 2 parameters")


def C(
    key: str,
    form_labels: Sequence[str],
    doc_labels: Sequence[str],
    value: ValueGenerator,
    *,
    kind: str = "text",
    placeholder: Sequence[str] = (),
    group: str = "allmant",
    options: Sequence[str] = (),
) -> Concept:
    return Concept(key, tuple(form_labels), tuple(doc_labels), _adapt(value), kind, tuple(placeholder), group, tuple(options))


CONCEPTS: list[Concept] = [
    # --- namn -------------------------------------------------------------
    C("fornamn", ("Förnamn", "Tilltalsnamn", "Förnamn enligt folkbokföringen"), ("Förnamn", "Tilltalsnamn"), lambda r, p: p["first"], placeholder=("Förnamn",), group="namn"),
    C("efternamn", ("Efternamn", "Familjenamn", "Släktnamn"), ("Efternamn", "Familjenamn"), lambda r, p: p["last"], placeholder=("Efternamn",), group="namn"),
    C("fullt_namn", ("Namn", "Fullständigt namn", "Namn (för- och efternamn)", "Sökandens namn"), ("Fullständigt namn", "Namn", "Sökande"), lambda r, p: p["full"], placeholder=("För- och efternamn",), group="namn"),
    C("personnummer", ("Personnummer", "Personnummer (ÅÅMMDD-XXXX)", "Pnr"), ("Personnummer", "Pnr"), gen_personnummer, placeholder=("ÅÅMMDD-XXXX",), group="namn"),
    C("anhorig_namn", ("Anhörigs namn", "Närståendes namn", "Kontaktperson"), ("Anhörig", "Närstående", "Kontaktperson"), lambda r, p: f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}", group="namn"),
    # --- kontakt ----------------------------------------------------------
    C("e_post", ("E-post", "E-postadress", "Mejladress", "E-post (jobbet)"), ("E-post", "E-postadress", "Mejl"), gen_email, placeholder=("namn@exempel.se",), group="kontakt"),
    C("telefon", ("Telefon", "Telefonnummer", "Mobil", "Mobilnummer", "Tel"), ("Telefon", "Mobil", "Tel"), gen_phone, placeholder=("070-123 45 67",), group="kontakt"),
    C("arbets_telefon", ("Arbetstelefon", "Telefon arbete", "Jobbtelefon"), ("Arbetstelefon", "Telefon arbete"), gen_work_phone, group="kontakt"),
    C("anhorig_telefon", ("Anhörigs telefon", "Telefon till anhörig", "Nödkontakt telefon", "ICE-telefon"), ("Anhörigs telefon", "Nödkontakt", "ICE"), gen_phone, group="kontakt"),
    # --- adress -----------------------------------------------------------
    C("gatuadress", ("Gatuadress", "Adress", "Utdelningsadress", "Besöksadress"), ("Gatuadress", "Adress", "Utdelningsadress"), lambda r, p: p["address"], placeholder=("Gata och nummer",), group="adress"),
    C("postnummer", ("Postnummer", "Postnr"), ("Postnummer", "Postnr"), lambda r, p: p["zip"], placeholder=("XXX XX",), group="adress"),
    C("ort", ("Ort", "Postort", "Stad"), ("Ort", "Postort", "Stad"), lambda r, p: p["city"], group="adress"),
    C("land", ("Land", "Landskod"), ("Land",), lambda r, p: "Sverige" if r.random() < 0.7 else r.choice(("Norge", "Danmark", "Finland", "Tyskland")), group="adress"),
    C("lan", ("Län", "Region"), ("Län",), lambda r, p: p["county"], kind="select", options=tuple(COUNTIES), group="adress"),
    # --- anställning ------------------------------------------------------
    C("arbetsgivare", ("Arbetsgivare", "Arbetsgivarens namn", "Nuvarande arbetsgivare"), ("Arbetsgivare", "Anställd hos"), lambda r, p: p["employer"], group="anstallning"),
    C("yrke", ("Yrke", "Befattning", "Titel", "Tjänst"), ("Yrke", "Befattning", "Titel"), lambda r, p: p["title"], group="anstallning"),
    C("avdelning", ("Avdelning", "Enhet", "Sektion"), ("Avdelning", "Enhet"), lambda r, p: p["department"], kind="select", options=tuple(DEPARTMENTS), group="anstallning"),
    C("anstallningsnummer", ("Anställningsnummer", "Medarbetar-ID", "Personalnummer"), ("Anställningsnummer", "Personalnummer"), lambda r, p: f"AN-{_digits(r, 6)}", group="anstallning"),
    C("anstallningsdatum", ("Anställningsdatum", "Startdatum", "Tillträdesdatum"), ("Anställningsdatum", "Startdatum"), lambda r, p: gen_date(r, 2024, 2026), group="anstallning"),
    C("slutdatum", ("Slutdatum", "Sista anställningsdag", "Uppsägningstid t.o.m."), ("Slutdatum", "Sista dag"), lambda r, p: gen_date(r, 2026, 2027), group="anstallning"),
    # --- ekonomi ----------------------------------------------------------
    C("bankgiro", ("Bankgiro", "Bankgironummer", "BG"), ("Bankgiro", "BG"), gen_bankgiro, placeholder=("XXX-XXXX",), group="ekonomi"),
    C("plusgiro", ("Plusgiro", "Plusgironummer", "PG"), ("Plusgiro", "PG"), gen_plusgiro, group="ekonomi"),
    C("referens", ("Referens", "Referensnummer", "OCR-nummer", "Betalningsreferens"), ("Referens", "OCR", "Referensnummer"), gen_ocr, group="ekonomi"),
    C("belopp", ("Belopp", "Summa", "Belopp att betala", "Totalt belopp"), ("Belopp", "Summa", "Totalt"), gen_money, placeholder=("0,00 kr",), group="ekonomi"),
    C("fakturadatum", ("Fakturadatum", "Faktureringsdatum", "Utfärdandedatum"), ("Fakturadatum", "Utfärdad"), lambda r, p: gen_date(r, 2026, 2026), group="ekonomi"),
    C("forfallodatum", ("Förfallodatum", "Sista betalningsdag", "Betalas senast"), ("Förfallodatum", "Förfaller"), lambda r, p: gen_date(r, 2026, 2027), group="ekonomi"),
    C("organisationsnummer", ("Organisationsnummer", "Org.nr", "Organisationsnr"), ("Organisationsnummer", "Org.nr"), gen_orgnummer, placeholder=("XXXXXX-XXXX",), group="ekonomi"),
    C("momsregistreringsnummer", ("Momsregistreringsnummer", "VAT-nummer", "Momsnr"), ("Momsregistreringsnummer", "VAT"), gen_vat, group="ekonomi"),
    C("iban", ("IBAN", "Bankkontonummer", "Kontonummer för utbetalning"), ("IBAN", "Konto"), gen_iban, group="ekonomi"),
    C("bank", ("Bank", "Banknamn"), ("Bank",), lambda r, p: p["bank"], group="ekonomi"),
    # --- försäkring -------------------------------------------------------
    C("forsakringsbolag", ("Försäkringsbolag", "Försäkringsgivare", "Bolag"), ("Försäkringsbolag", "Försäkringsgivare"), lambda r, p: p["insurer"], group="forsakring"),
    C("forsakringsnummer", ("Försäkringsnummer", "Försäkringsnr", "Avtalsnummer"), ("Försäkringsnummer", "Avtal"), lambda r, p: f"F-{_digits(r, 4)}-{_digits(r, 5)}", group="forsakring"),
    C("skadenummer", ("Skadenummer", "Ärendenummer skada"), ("Skadenummer", "Skada"), lambda r, p: f"S-{_digits(r, 6)}", group="forsakring"),
    C("skadedatum", ("Skadedatum", "Datum för skadan", "Händelsedatum"), ("Skadedatum", "Händelsedatum"), lambda r, p: gen_date(r, 2026, 2026), group="forsakring"),
    # --- hälsa ------------------------------------------------------------
    C("vardcentral", ("Vårdcentral", "Hälsocentral", "Mottagning"), ("Vårdcentral", "Mottagning"), lambda r, p: p["center"], group="halsa"),
    C("lakare", ("Läkare", "Behandlande läkare", "Ansvarig läkare"), ("Läkare", "Behandlare"), lambda r, p: p["doctor"], group="halsa"),
    C("allergier", ("Allergier", "Kända allergier", "Överkänslighet"), ("Allergier", "Överkänslighet"), lambda r, p: r.choice(ALLERGIES), kind="select", options=ALLERGIES, group="halsa"),
    C("blodgrupp", ("Blodgrupp", "Blodgrupp och Rh"), ("Blodgrupp",), lambda r, p: r.choice(BLOOD_TYPES), kind="select", options=BLOOD_TYPES, group="halsa"),
    # --- utbildning -------------------------------------------------------
    C("universitet", ("Universitet", "Lärosäte", "Skola"), ("Universitet", "Lärosäte"), lambda r, p: p["university"], group="utbildning"),
    C("program", ("Program", "Utbildningsprogram", "Linje"), ("Program", "Utbildning"), lambda r, p: p["program"], group="utbildning"),
    C("kurs", ("Kurs", "Kursnamn", "Kurskod"), ("Kurs", "Kursnamn"), lambda r, p: p["course"], group="utbildning"),
    # --- fordon -----------------------------------------------------------
    C("registreringsnummer", ("Registreringsnummer", "Reg.nr", "Fordonsnummer"), ("Registreringsnummer", "Reg.nr"), gen_registration, placeholder=("ABC 123",), group="fordon"),
    C("fordonsmarke", ("Märke", "Fordonsmärke", "Fabrikat"), ("Fordonsmärke", "Fabrikat"), lambda r, p: f"{r.choice(CAR_MAKES)[0]} {r.choice(CAR_MAKES)[1]}", group="fordon"),
    C("fordonsmodell", ("Modell", "Fordonsmodell"), ("Fordonsmodell", "Modell"), lambda r, p: r.choice(("Prov 200", "Test 3", "Ljus 40", "Nord XL")), group="fordon"),
    C("arshjul", ("Årsmodell", "År"), ("Årsmodell",), lambda r, p: str(r.randint(2012, 2025)), group="fordon"),
    # --- händelse / ärende ------------------------------------------------
    C("handelsedatum", ("Händelsedatum", "Datum för händelsen", "Inträffade"), ("Händelsedatum", "Datum"), lambda r, p: gen_date(r, 2026, 2026), group="handelse"),
    C("handelseort", ("Händelseort", "Plats", "Ort för händelsen"), ("Händelseort", "Plats"), lambda r, p: p["city"], group="handelse"),
    C("arendenummer", ("Ärendenummer", "Diarienummer", "Ärendenr"), ("Ärendenummer", "Dnr"), lambda r, p: f"EX-{r.randint(2024, 2026)}-{_digits(r, 5)}", group="handelse"),
    C("beskrivning", ("Beskrivning av händelsen", "Vad inträffade?", "Händelseförlopp"), ("Beskrivning", "Händelseförlopp"), lambda r, p: r.choice(("Kort sammanfattning av händelseförloppet.", "Kunden rapporterade felet per telefon.", "Skadan upptäcktes vid leverans.", "Ärendet avser en försenad leverans.")), kind="textarea", group="handelse"),
    C("medlemsnummer", ("Medlemsnummer", "Kundnummer"), ("Medlemsnummer", "Kundnummer"), lambda r, p: f"M-{_digits(r, 6)}", group="handelse"),
]

CONCEPT_BY_KEY: dict[str, Concept] = {concept.key: concept for concept in CONCEPTS}

# Concepts that share a group and are easy to confuse; the generator co-locates
# them so the model must read the complete label rather than one token.
HARD_NEGATIVE_PAIRS: tuple[tuple[str, str], ...] = (
    ("e_post", "gatuadress"),
    ("telefon", "anhorig_telefon"),
    ("telefon", "arbets_telefon"),
    ("fullt_namn", "anhorig_namn"),
    ("fornamn", "anhorig_namn"),
    ("fakturadatum", "forfallodatum"),
    ("fakturadatum", "skadedatum"),
    ("handelsedatum", "skadedatum"),
    ("postnummer", "forsakringsnummer"),
    ("ort", "land"),
    ("arbetsgivare", "forsakringsbolag"),
    ("bankgiro", "plusgiro"),
    ("referens", "bankgiro"),
    ("personnummer", "organisationsnummer"),
    ("belopp", "momsregistreringsnummer"),
    ("universitet", "arbetsgivare"),
    ("kurs", "program"),
    ("vardcentral", "ort"),
)

FORM_TITLES: tuple[str, ...] = (
    "Exempelkliniken - Ny patientregistrering",
    "Norrsken Teknik AB - Reseräkning",
    "Testförsäkring - Skadeanmälan",
    "Uppsala kommun - Ansökan om förskoleplats",
    "Exempelbanken - Autogiro och direktbetalning",
    "Testgymnasiet - Ansökan till introduktionsprogrammet",
    "Provhagen AB - Leverantörsregistrering",
    "Exempelvårdcentralen - Hälsodeklaration",
    "Testfastigheter - Hyresansökan",
    "Exempelbiblioteket - Ansökan om lånekort",
    "Testuniversitetet - Anmälan till kurs",
    "Exempeltelekom - Överföring av abonnemang",
    "Testapoteket - Receptöverföring",
    "Exempelgymmet - Medlemsansökan",
    "Testtandvården - Ny patient",
    "Exempel Elnät AB - Nyanslutning",
    "Provskolan - Skolskjutsansökan",
    "Nordljus Logistik AB - Fraktbokning",
)

SUBMIT_LABELS: tuple[str, ...] = (
    "Skicka", "Skicka in", "Skicka formulär", "Spara och skicka", "Registrera", "Ansök", "Skicka ansökan",
)
NON_SUBMIT_BUTTONS: tuple[str, ...] = (
    "Rensa", "Återställ", "Avbryt", "Tillbaka", "Nästa", "Föregående", "Hjälp", "Ladda upp bilaga", "Spara utkast", "Visa sammanfattning",
)
REQUIRED_CHECKBOXES: tuple[str, ...] = (
    "Jag intygar att uppgifterna är korrekta",
    "Jag godkänner villkoren",
    "Jag samtycker till behandling av personuppgifter",
    "Uppgifterna lämnas enligt min kännedom",
    "Jag har tagit del av informationen om behandling av personuppgifter",
)
OPTIONAL_CHECKBOXES: tuple[str, ...] = (
    "Skicka nyhetsbrev",
    "Kontakta mig per e-post",
    "Fakturera per e-post",
    "Jag vill ha information om erbjudanden",
)
OPTIONAL_FIELDS: tuple[str, ...] = (
    "Kommentar", "Övrig information", "Anteckningar", "Ärendebeskrivning", "Meddelande", "Fritext", "Övrigt",
)
DISTRACTOR_ENTITIES: tuple[tuple[str, ValueGenerator], ...] = (
    ("Diarienummer", lambda r: f"DNR-{r.randint(2024, 2026)}-{_digits(r, 4)}"),
    ("Bilagor", lambda r: f"{r.randint(1, 9)} st"),
    ("Avsändare", lambda r: r.choice(COMPANIES)),
    ("Mottagare", lambda r: r.choice(COMPANIES)),
    ("Handläggare", lambda r: f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"),
    ("Beslutsdatum", lambda r: gen_date(r, 2026, 2026)),
    ("Utfärdat av", lambda r: r.choice(MUNICIPALITIES)),
    ("Formulärversion", lambda r: f"v{r.randint(1, 4)}.{r.randint(0, 9)}"),
    ("Sidnummer", lambda r: f"{r.randint(1, 6)} av 6"),
    ("Sekretess", lambda r: r.choice(("Nej", "Ja", "Delvis"))),
)
CHROME_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("Button", "Stäng"),
    ("Button", "Minimera"),
    ("Button", "Maximera"),
    ("Button", "Ladda om"),
    ("Button", "Bakåt"),
    ("Button", "Framåt"),
    ("Button", "Bokmärk denna sida"),
    ("Button", "Ny flik"),
    ("Button", "Inställningar och mer"),
    ("Edit", "Adressfält"),
    ("Edit", "Sök"),
    ("MenuItem", "Arkiv"),
    ("MenuItem", "Visa"),
    ("MenuItem", "Hjälp"),
    ("Pane", "Sidinnehåll"),
    ("ScrollBar", "Vertikal"),
    ("Button", "Sida upp"),
    ("Button", "Sida ner"),
    ("Button", "Reload"),
    ("TitleBar", ""),
)
WINDOW_SUFFIXES: tuple[str, ...] = (" - Google Chrome", " - Microsoft Edge", " - Mozilla Firefox", " - Safari", "")
BLANK_TITLES: tuple[str, ...] = ("", "Namnlös", "Formulär", "Sida 2 av 3")
