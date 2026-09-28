"""v2 record normalization (names and addresses). Pure functions, one record
at a time; src/v2/prepare.py runs them in parallel and writes parquet.

Country-specific rules key off the label: the France seed table applies only
when country == "France". Learned rewrites and the native-script dictionary
(milestone 2) are passed in as plain dicts, so an unseen label simply gets none.
"""
from __future__ import annotations

import re

from anyascii import anyascii

from src.entity_resolution.normalize import _FR_ABBREVIATIONS, _expand_fr_segment, _filter_tokens

# France only (country == "France"): extra street/zone abbreviations, legal forms that
# may lead the name, and postal box / CEDEX routing codes dropped before postcode extraction.
FR_ADDRESS_TABLE = {**_FR_ABBREVIATIONS, "zi": "zone industrielle", "za": "zone artisanale",
                    "zac": "zone d amenagement concerte"}
FR_LEAD_LEGAL = {"sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "other", "snc": "other", "sci": "other",
                 "selarl": "other", "sa": "sa", "ste": "other", "societe": "other", "ets": "other",
                 "etablissements": "other"}
FR_TAIL_LEGAL = {"sci": "other", "selarl": "other", "cie": "other", "ei": "other"}
_FR_BOX_RE = re.compile(r"\b(?:b\.?\s?p|cs|tsa)\.?\s*\d+\b")

# ---------------------------------------------------------------- shared

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_DOTTED_ABBR_RE = re.compile(r"\b(?:[a-z]\.){2,}")


def is_non_latin(raw: str) -> bool:
    """True if any letter is outside the Latin blocks (checked before anyascii)."""
    for ch in raw:
        o = ord(ch)
        if o > 0x24F and ch.isalpha() and not 0x1E00 <= o <= 0x1EFF:
            return True
    return False


def _ascii(raw) -> str:
    return anyascii(raw).casefold() if isinstance(raw, str) else ""


def _clean(text: str) -> str:
    text = text.replace("&", " and ")
    text = _DOTTED_ABBR_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()


# ---------------------------------------------------------------- names

_ALIAS_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|a\.k\.a\.?|aka|f\s*/\s*k\s*/\s*a|f\.k\.a\.?|fka"
                       r"|t\s*/\s*a|trading\s+as)\b")
LEGAL_CLASS_OF = {
    "private": "private", "pvt": "private", "pvtltd": "private_limited", "pte": "private",
    "limited": "limited", "ltd": "limited", "llp": "llp", "opc": "other", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "llc": "llc", "co": "other", "company": "other", "plc": "other",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "other", "snc": "other", "gmbh": "other",
}
LEGAL_CLASSES = ["none", "private_limited", "limited", "llp", "inc", "llc", "corp", "sarl", "sas", "sa", "other"]
_HONORIFICS = ("the", "shri", "sri", "smt")


def _legal_class(removed: list[str]) -> str:
    classes = {LEGAL_CLASS_OF.get(t) or FR_LEAD_LEGAL.get(t, "other") for t in removed}
    if not classes:
        return "none"
    if "private_limited" in classes or {"private", "limited"} <= classes:
        return "private_limited"
    for c in ("limited", "llp", "inc", "llc", "corp", "sarl", "sas", "sa"):
        if c in classes:
            return c
    return "other"


def _strip_honorifics(tokens: list[str]) -> list[str]:
    while len(tokens) > 1:
        if tokens[0] in _HONORIFICS:
            tokens = tokens[1:]
        elif tokens[:2] == ["m", "s"] and len(tokens) > 2:
            tokens = tokens[2:]
        else:
            break
    return tokens


def _strip_legal(tokens: list[str], legal_extra: dict[str, str], extra_classes: dict | None = None
                 ) -> tuple[list[str], list[str]]:
    """Remove legal tokens among the last 3, any order; never empties the name."""
    classes = {**LEGAL_CLASS_OF, **(extra_classes or {})}

    def cls(t):
        return legal_extra.get(t, t)
    head, tail = tokens[:-3], tokens[-3:]
    kept = [t for t in tail if cls(t) not in classes]
    removed = [cls(t) for t in tail if cls(t) in classes]
    if not head and not kept:
        return tokens, []
    return head + kept, removed


def skeleton(text: str) -> str:
    """Consonant skeleton per token: ph->f, sh->s, c/q->k, v->w, z->j, m before a
    consonant -> n, drop non-leading vowels, collapse repeated letters."""
    out = []
    for tok in text.split():
        t = tok.replace("ph", "f").replace("sh", "s")
        t = t.translate(str.maketrans({"c": "k", "q": "k", "v": "w", "z": "j"}))
        t = re.sub(r"m(?=[bcdfghjklmnpqrstvwxz])", "n", t)
        t = t[:1] + re.sub(r"[aeiou]", "", t[1:])
        t = re.sub(r"(.)\1+", r"\1", t)
        if t:
            out.append(t)
    return " ".join(out)


def normalize_name(raw, word_dict: dict[str, str] | None = None, legal_extra: dict[str, str] | None = None,
                   country: str = "") -> dict:
    """name, name_alias, name_core, name_legal_class, name_skeleton, non_latin."""
    non_latin = is_non_latin(raw) if isinstance(raw, str) else False
    text = _ascii(raw)
    parts = _ALIAS_RE.split(text)
    main, aliases = parts[0], [_clean(p) for p in parts[1:]]
    tokens = _clean(main).split()
    if word_dict and non_latin:  # learned native-script dictionary: only names that were not Latin
        tokens = [w for t in tokens for w in word_dict.get(t, t).split()]
    tokens = _strip_honorifics(tokens)
    lead = []
    if country == "France" and len(tokens) > 1 and tokens[0] in FR_LEAD_LEGAL:
        lead, tokens = [tokens[0]], tokens[1:]
    core, removed = _strip_legal(tokens, legal_extra or {}, FR_TAIL_LEGAL if country == "France" else None)
    removed = lead + removed
    return {"name": " ".join(tokens), "name_alias": " | ".join(a for a in aliases if a),
            "name_core": " ".join(core), "name_legal_class": _legal_class(removed),
            "name_skeleton": skeleton(" ".join(core)), "non_latin": non_latin}


# ---------------------------------------------------------------- addresses

_JUNK_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|\b(?:null|none|n/a)\b")
_HOUSE_RE = re.compile(r"^(?:(?:h|house|plot|door|flat|shop|unit|office|room|block|sy|sr|survey)\.?\s*)?"
                       r"(?:no|number)?\.?\s*[-:.]?\s*#?\s*"
                       r"([a-z]{0,2}\s*[-/]?\s*\d+(?:\s*[-/]\s*\d+)?\s*(?:bis|ter|[a-z])?)\b")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
_POSTCODE_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_DIGITS_RE = re.compile(r"\d+")
_LANDMARK_WORDS = frozenset({"near", "nr", "nearby", "opp", "opposite", "behind", "beside"})


_FR_HOUSE_RE = re.compile(r"^(?:(?:ndeg|no|n)\.?\s*)?#?\s*(\d+)(?:\s*[-/]\s*(\d+))?(?:(bis|ter)\b|([a-z])\b|\s+(bis|ter|b|t)\b)?")


def _fr_house_number(parts: list[str], postcode: str) -> tuple[str, str]:
    """France: a spaced single letter is a suffix only for b/t (bis/ter), so the
    r/q of "73 Q. de Paludate" (quai) is never read as part of the number."""
    for part in parts:
        p = part.strip()
        if postcode and p.startswith(postcode):
            continue
        m = _FR_HOUSE_RE.match(p)
        if not m:
            continue
        num, end, attached, letter, spaced = m.groups()
        suffix = attached or letter or {"b": "bis", "t": "ter"}.get(spaced, spaced) or ""
        return num + suffix, (f"{num}-{end}" if end else "")
    return "", ""


def _house_number(parts: list[str], postcode: str) -> tuple[str, str]:
    """(house_num, house_range) from the first part that starts with a number
    (address parts are shuffled in this data, so not necessarily the first)."""
    for part in parts:
        if postcode and part.strip().startswith(postcode):
            continue
        m = _HOUSE_RE.match(part.strip())
        if not m:
            continue
        num = re.sub(r"[\s/-]", "", m.group(1))
        if _ORDINAL_RE.match(num) or not any(c.isdigit() for c in num):
            continue
        rng = re.match(r"^[a-z]{0,2}(\d+)[-/](\d+)$", re.sub(r"\s", "", m.group(1)))
        if rng:
            return rng.group(1), f"{rng.group(1)}-{rng.group(2)}"
        return num, ""
    return "", ""


_FR_POSTCODE_RE = re.compile(r"^(\d{5})\s+[a-z]")


def _postcode(parts: list[str], country: str) -> str:
    """A 5-6 digit number that does not start its part, or a part that is only
    that number; in France also "33000 bordeaux" (code before the city).
    Last one wins."""
    found = []
    for p in (q.strip() for q in parts):
        for m in _POSTCODE_RE.finditer(p):
            if m.start() > 0 or m.group(1) == p:
                found.append(m.group(1))
        fr = _FR_POSTCODE_RE.match(p) if country == "France" else None
        if fr and not re.search(r"\d", p[fr.end():]):
            found.append(fr.group(1))
    return found[-1] if found else ""


def normalize_address(raw, country: str, rewrites: dict[str, str] | None = None,
                      part_rewrites: dict[str, str] | None = None) -> dict:
    """addr, addr_parts, addr_landmark, house_num, house_range, postcode, num_tokens."""
    text = _JUNK_RE.sub(" ", _ascii(raw))
    if country == "France":
        text = _FR_BOX_RE.sub(" ", text)
    raw_parts = [p.strip() for p in text.split(",") if p.strip()]
    postcode = _postcode(raw_parts, country)
    house, house_range = (_fr_house_number if country == "France" else _house_number)(raw_parts, postcode)
    parts, landmark, tokens = [], [], []
    for p in raw_parts:
        seg = _clean(p)
        if part_rewrites and seg in part_rewrites:
            seg = part_rewrites[seg]
        seg_tokens = seg.split()
        for i, t in enumerate(seg_tokens):
            if t in _LANDMARK_WORDS:
                landmark.extend(seg_tokens[i + 1:])
                seg_tokens = seg_tokens[:i]
                break
        if rewrites:
            seg_tokens = [w for t in seg_tokens for w in rewrites.get(t, t).split()]
        if country == "France":
            seg_tokens = [w for t in _expand_fr_segment(seg_tokens, FR_ADDRESS_TABLE) for w in t.split()]
        seg_tokens = _filter_tokens(seg_tokens, keep_digits=True)
        if seg_tokens:
            parts.append(" ".join(seg_tokens))
            tokens.extend(seg_tokens)
    others = [d for d in _DIGITS_RE.findall(" ".join(tokens)) if d not in (postcode, re.sub(r"\D", "", house))]
    return {"addr": " ".join(tokens), "addr_parts": " | ".join(parts), "addr_landmark": " ".join(landmark),
            "house_num": house, "house_range": house_range, "postcode": postcode, "num_tokens": " ".join(others)}


def normalize_record(name, addr, country: str, word_dict=None, legal_extra=None, rewrites=None,
                     part_rewrites=None) -> dict:
    out = normalize_name(name, word_dict, legal_extra, country)
    out.update(normalize_address(addr, country, rewrites, part_rewrites))
    addr_tokens = set(out["addr"].split())
    out["name_core_minus_addr"] = " ".join(t for t in out["name_core"].split() if t not in addr_tokens)
    return out
