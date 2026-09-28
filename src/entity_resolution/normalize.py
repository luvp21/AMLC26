"""Normalize business_name/business_address text so records from different
sources (different scripts, abbreviation conventions, legal-suffix wording)
become comparable. The baseline functions (normalize_name/normalize_address)
never branch on `country`. The flagged Phase 2 path (normalize_record) adds
country-aware abbreviation and stopword tables; a country label without a
table (anything unseen) gets only the generic, unambiguous rules.
"""
from __future__ import annotations

import re

from anyascii import anyascii
from cleanco import basename

# Common street/unit abbreviations seen in the data (US, India, France address
# styles) — expand to a canonical long form so "Rd"/"Road" compare equal.
# Deliberately small and address-generic, not country-specific.
_ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "apt": "apartment", "ste": "suite",
    "bldg": "building", "fl": "floor", "nr": "near", "opp": "opposite",
    "sq": "square", "ter": "terrace", "cir": "circle", "pl": "place",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def _to_ascii_lower(text: str) -> str:
    """Transliterate any script to ASCII (Devanagari/Tamil/Kannada/accented
    Latin all included) then lowercase. Confirmed necessary: Source 2 has
    full Devanagari business names, Source 3 has Tamil/Kannada tokens mixed
    into otherwise-Latin text — plain string similarity can't compare across
    scripts without this step."""
    return anyascii(text).lower()


def normalize_name(raw: str | float | None) -> str:
    """Clean a business_name for comparison: transliterate, strip legal
    suffix (Ltd/Pvt/LLC/SARL/...) via cleanco, strip punctuation, collapse
    whitespace."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = _to_ascii_lower(raw)
    text = basename(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(raw: str | float | None) -> str:
    """Clean a business_address for comparison: transliterate, expand common
    abbreviations, strip punctuation, collapse whitespace. No legal-suffix
    stripping (addresses don't have those)."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = _to_ascii_lower(raw)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [_ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in text.split(" ")]
    return " ".join(tokens)


def tokenize(text: str, min_len: int = 2) -> list[str]:
    """Split normalized text into tokens for blocking/Jaccard features.
    Drops single-character tokens (mostly noise from address unit numbers)."""
    if not text:
        return []
    return [t for t in text.split(" ") if len(t) >= min_len]


# ---------------------------------------------------------------- Phase 2 (flagged)
# normalize_record() reproduces normalize_name/normalize_address + tokenize
# exactly when every NormalizationConfig flag is off (tested), and adds each
# Phase 2 fix behind its own flag. Country-aware steps key off the country
# label; a label with no table gets only the generic table.

# Unambiguous everywhere. Together with _EN_ABBREVIATIONS this is exactly
# _ADDRESS_ABBREVIATIONS, so US/India tokens don't change under country_abbrev.
_GENERIC_ABBREVIATIONS = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "hwy": "highway",
    "pkwy": "parkway", "apt": "apartment", "bldg": "building", "nr": "near", "opp": "opposite",
    "sq": "square",
}
# Ambiguous across languages (st = street / saint, ter = terrace / bis-ter).
_EN_ABBREVIATIONS = {
    "st": "street", "dr": "drive", "ln": "lane", "ct": "court", "fl": "floor", "ste": "suite",
    "ter": "terrace", "cir": "circle", "pl": "place",
}
_FR_ABBREVIATIONS = {
    "av": "avenue", "ave": "avenue", "bd": "boulevard", "boul": "boulevard", "blvd": "boulevard",
    "ch": "chemin", "chem": "chemin", "imp": "impasse", "pl": "place", "fg": "faubourg",
    "rte": "route", "st": "saint", "ste": "sainte", "all": "allee", "sq": "square", "crs": "cours",
}
_COUNTRY_ABBREVIATIONS = {
    "US": {**_GENERIC_ABBREVIATIONS, **_EN_ABBREVIATIONS},
    "India": {**_GENERIC_ABBREVIATIONS, **_EN_ABBREVIATIONS},
    "France": {**_GENERIC_ABBREVIATIONS, **_FR_ABBREVIATIONS},
}
# France only, and only in street position (see _expand_fr_street).
_FR_STREET_POSITION = {"r": "rue", "q": "quai"}
# "15 B" and "15 bis" are the same address; both become "15bis".
_FR_HOUSE_SUFFIXES = {"bis": "bis", "b": "bis", "ter": "ter", "t": "ter"}
_FR_DROP = frozenset({"cedex", "ndeg"})  # ndeg: anyascii of "N°"
_FR_NUMBER_MARKERS = frozenset({"no"})  # "NO 132 R ..." — dropped only right before a number

_STOPWORDS_GENERIC = frozenset({"the", "of", "and", "near", "opp", "opposite", "behind", "beside"})
_STOPWORDS_BY_COUNTRY = {"France": frozenset({"de", "du", "la", "le", "les", "des", "chez", "et"})}

_LANDMARK_WORDS = frozenset({"near", "nr", "nearby", "opp", "opposite", "behind", "beside"})

LEGAL_TOKENS = frozenset({
    "private", "pvt", "pvtltd", "pte", "limited", "ltd", "llp", "opc", "inc", "incorporated",
    "corp", "corporation", "llc", "co", "company", "cie", "sarl", "sas", "sasu", "sa", "eurl", "snc",
})
_LEGAL_TAIL = 3

_POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?:-\d{4})?(?!\d)")
_DIGITS_RE = re.compile(r"\d+")


def _has_digit(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)


def _filter_tokens(tokens: list[str], keep_digits: bool) -> list[str]:
    """Baseline: len >= 2. keep_digits: also digit tokens of any length and
    single letters next to a token containing a digit ("block a 12", "5 b")."""
    if not keep_digits:
        return [t for t in tokens if len(t) >= 2]
    out = []
    for i, t in enumerate(tokens):
        if len(t) >= 2 or t.isdigit():
            out.append(t)
        elif t and ((i > 0 and _has_digit(tokens[i - 1])) or (i + 1 < len(tokens) and _has_digit(tokens[i + 1]))):
            out.append(t)
    return out


def _split_tokens(text: str) -> list[str]:
    text = _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()
    return text.split(" ") if text else []


def _is_house_number(tokens: list[str], i: int) -> bool:
    """tokens[i] ends a house-number group: a digit token, optionally followed
    by bis/ter or a single letter ("5 b r du ...")."""
    if i < 0:
        return False
    if _has_digit(tokens[i]):
        return True
    return (tokens[i] in _FR_HOUSE_SUFFIXES or len(tokens[i]) == 1) and i > 0 and _has_digit(tokens[i - 1])


def _expand_fr_segment(tokens: list[str], table: dict[str, str]) -> list[str]:
    """French rules for one comma segment: r/q -> rue/quai only at segment
    start or right after a house number; "59 bis" / "59 b" -> "59bis"; drop
    cedex, "N°" and a "no" marker before a number."""
    out: list[str] = []
    for i, t in enumerate(tokens):
        if t in _FR_DROP or (t in _FR_NUMBER_MARKERS and i + 1 < len(tokens) and tokens[i + 1].isdigit()):
            continue
        if t in _FR_STREET_POSITION and (i == 0 or _is_house_number(tokens, i - 1)):
            out.append(_FR_STREET_POSITION[t])
        elif t in _FR_HOUSE_SUFFIXES and out and out[-1].isdigit():
            out[-1] = out[-1] + _FR_HOUSE_SUFFIXES[t]
        else:
            out.append(table.get(t, t))
    return out


def _split_landmark(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Everything from the first landmark word to the end of the segment is
    the landmark phrase ("near dindoshi bus depot")."""
    for i, t in enumerate(tokens):
        if t in _LANDMARK_WORDS:
            return tokens[:i], tokens[i + 1:]
    return tokens, []


def _strip_legal_tail(tokens: list[str]) -> list[str]:
    """Drop legal-form tokens among the last three, in any order ("sai tech
    limited private" -> "sai tech"); never empties the name."""
    head, tail = tokens[:-_LEGAL_TAIL], tokens[-_LEGAL_TAIL:]
    kept = head + [t for t in tail if t not in LEGAL_TOKENS]
    return kept if kept else tokens


def _drop_stopwords(tokens: list[str], country: str) -> list[str]:
    stop = _STOPWORDS_GENERIC | _STOPWORDS_BY_COUNTRY.get(country, frozenset())
    kept = [t for t in tokens if t not in stop]
    return kept if kept else tokens


def _address_numbers(text: str) -> tuple[str, list[str], list[str]]:
    """(postal, postal_all, house_nums) from the transliterated raw address:
    postal = standalone 5-6 digit runs (last one wins), house_nums = every
    other digit run."""
    postal_all = _POSTAL_RE.findall(text)
    rest = _POSTAL_RE.sub(" ", text)
    return (postal_all[-1] if postal_all else ""), postal_all, _DIGITS_RE.findall(rest)


def normalize_address_tokens(raw, country: str, cfg) -> tuple[list[str], list[str], str]:
    """(addr_tokens, addr_landmark, transliterated lowercase text)."""
    if not isinstance(raw, str) or not raw.strip():
        return [], [], ""
    text = _to_ascii_lower(raw)
    table = _COUNTRY_ABBREVIATIONS.get(country, _GENERIC_ABBREVIATIONS) if cfg.country_abbrev else _ADDRESS_ABBREVIATIONS
    tokens: list[str] = []
    landmark: list[str] = []
    for segment in text.split(","):
        seg = _split_tokens(segment)
        if cfg.landmarks:
            seg, lm = _split_landmark(seg)
            landmark.extend(lm)
        if cfg.country_abbrev and country == "France":
            seg = _expand_fr_segment(seg, table)
        else:
            seg = [table.get(t, t) for t in seg]
        tokens.extend(seg)
    tokens = _filter_tokens(tokens, cfg.keep_digits)
    if cfg.stopwords:
        tokens = _drop_stopwords(tokens, country)
    return tokens, _filter_tokens(landmark, cfg.keep_digits), text


def normalize_name_tokens(raw, country: str, cfg) -> list[str]:
    tokens = _filter_tokens(_split_tokens(normalize_name(raw)), cfg.keep_digits)
    if cfg.legal_bag:
        tokens = _strip_legal_tail(tokens)
    if cfg.stopwords:
        tokens = _drop_stopwords(tokens, country)
    return tokens


def normalize_record(name, addr, country: str, cfg) -> dict:
    """All normalized fields for one record under NormalizationConfig `cfg`.
    name_core = name tokens that don't also appear in the record's own
    address (drops city names like "bordeaux" from "Bordeaux Maison")."""
    name_tokens = normalize_name_tokens(name, country, cfg)
    addr_tokens, landmark, text = normalize_address_tokens(addr, country, cfg)
    out = {"name_tokens": name_tokens, "addr_tokens": addr_tokens}
    if cfg.landmarks:
        out["addr_landmark"] = landmark
    if cfg.extract_fields:
        postal, postal_all, house_nums = _address_numbers(text)
        addr_set = set(addr_tokens)
        out.update(postal=postal, postal_all=postal_all, house_nums=house_nums,
                   name_core=[t for t in name_tokens if t not in addr_set])
    return out
