"""Language detection and Unicode script utilities for BhashaCore.

Detects the six supported languages from native-script text.  Bengali and
Assamese share a script block, as do Hindi and Nepali, so detection combines
Unicode range counting with high-signal function words.  Latin-script text is
English by default; a small romanised-Indic heuristic is reported as low
confidence so the UI can ask the user to pick a language manually.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "as": "Assamese",
    "ne": "Nepali",
}

LANGUAGE_CODES: Dict[str, str] = {name: code for code, name in LANGUAGE_NAMES.items()}

SUPPORTED_LANGUAGE_CODES = tuple(LANGUAGE_NAMES)

_SCRIPT_RANGES = {
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
}

# High-signal function words.  Overlapping forms are deliberately omitted so
# that a single shared word does not flip the decision.
_NEPALI_MARKERS = frozenset(
    {
        "छ", "छन्", "छु", "छौं", "हुन्छ", "हुन्", "होइन", "छैन", "गर्नुहोस्",
        "गर्नुपर्छ", "गर्न", "गरेको", "भएको", "लागि", "तपाईं", "तपाई", "मेरो",
        "पनि", "कसरी", "किन", "कहिले", "कुन", "अनि", "भन्नुहोस्", "दिनुहोस्",
        "सक्छ", "सक्छन्", "जस्तै", "बाट", "सम्म", "अथवा", "हरू", "हुन्थ्यो",
        "गर्दै", "गर्छ", "गर्छन्", "देखि", "सँग", "भित्र", "बाहिर", "लाई",
        "भयो", "दिएको", "भनेको", "कस्तो", "धेरै", "अहिले", "यो", "त्यो",
    }
)

_HINDI_MARKERS = frozenset(
    {
        "है", "हैं", "हूँ", "करें", "करना", "करने", "किया", "लिए", "आप", "मेरा",
        "मेरी", "मेरे", "नहीं", "क्या", "क्यों", "कैसे", "कब", "कहाँ", "था",
        "थी", "थे", "और", "भी", "बताइए", "बताएं", "समझाइए", "समझाएं", "दीजिए",
        "कृपया", "सकता", "सकती", "सकते", "चाहिए", "कौन", "कौन-कौन", "किस",
        "किसे", "जैसे", "उदाहरण", "हम", "हमारा", "आपका", "करता", "करती",
        "करते", "होगा", "होगी", "होंगे", "रहा", "रही", "रहे", "में", "हुआ",
        "हुई", "होता", "होती", "होते", "कितना", "कितनी",
    }
)

_ASSAMESE_MARKERS = frozenset(
    {
        "হৈছে", "হ'ল", "মোৰ", "আপুনি", "আপোনাৰ", "কৰিব", "কৰক", "কৰা", "কৰি",
        "দিয়ক", "বাবে", "কেনেকৈ", "কিয়", "কেতিয়া", "ক'ত", "নহয়", "লগত",
        "আছিল", "তেওঁ", "তেখেত", "মোক", "আমাৰ", "ইয়াত", "তাত", "এতিয়া",
        "অনুগ্ৰহ", "প্ৰশ্ন", "উত্তৰ", "জানিব", "বুজিব", "ক'ব", "দিব", "ল'ব",
        "হ'ব", "নাই", "কৰিছে", "কৰিছে", "সকলো", "কিবা", "কেনে", "কিমান",
        "দেখুৱাই", "ব্যৱহাৰ", "প্ৰক্ৰিয়া", "শিক্ষা", "ভাষা", "জন্তু", "আৰু",
        "বুলি", "কৰিবলৈ",
    }
)

_BENGALI_MARKERS = frozenset(
    {
        "হয়েছে", "আমার", "আপনি", "আপনার", "করা", "করুন", "জন্য", "কেন",
        "কীভাবে", "কোথায়", "কখন", "এবং", "সাথে", "নয়", "ছিল", "দিন", "প্রশ্ন",
        "উত্তর", "বিষয়ে", "জানতে", "বুঝতে", "এখানে", "সেখানে", "এখন", "আমি",
        "আমাদের", "তারা", "তাহলে", "কিন্তু", "অথবা", "কিছু", "সবাই", "কাজ",
        "ব্যবহার", "প্রক্রিয়া", "শিক্ষা", "ভাষা", "প্রাণী", "কী", "কি",
    }
)

# Phrase-level cues resolve short sentences where single function words are
# shared or absent.  Only multi-word phrases live here; single words are part
# of the marker sets above, so evidence is never double-counted.
_NEPALI_PHRASES = (
    "के हो", "के छ", "कहाँ छ", "कति हो", "कसरी गर्न", "के गर्न",
    "कहाँ छन्", "कति छ", "कति छन्",
)
_HINDI_PHRASES = (
    "क्या है", "क्या हैं", "क्या था", "क्या थी", "क्या होता", "क्या होती",
)
_ASSAMESE_PHRASES = (
    "কি হৈছে", "কি কৰ", "কি কৰিব", "ক'ত আছে", "কেনেকৈ কৰ",
)
_BENGALI_PHRASES = (
    "কী হয়", "কি হয়", "কীভাবে কর", "কেন হয়",
)

# Latin-script Indic ("Hinglish" and friends).  Two or more matches are needed
# before we report a low-confidence romanised text.
_ROMANISED_INDIC_MARKERS = frozenset(
    {
        "kya", "kaise", "kaun", "kyu", "kyun", "kripya", "mujhe", "chahiye",
        "kitna", "kab", "kahan", "nahi", "batao", "samjhao", "matlab", "hoga",
        "hogi", "karo", "karna", "raha", "rahi",
        "kemon", "kothay", "kivabe", "kobe", "amar", "apnar", "amake", "korbo",
        "koro", "hobe", "ache", "chilo", "keno", "bolun",
        "kivein", "kado", "kithe", "tuhada", "menu", "daso", "hunda", "vich",
        "layi", "kiven", "tuhade", "sada", "sanu",
        "kenekoi", "ketiya", "kio", "mur", "mor", "apuni", "koribo", "lage",
        "asil", "kotha", "koi", "bisoye",
        "kasto", "kasari", "kahile", "mero", "tapai", "malai", "dinus",
        "garnus", "chha", "chhan", "bhayo", "garne", "bata", "sanga", "huncha",
    }
)

_LATIN_LANGUAGE_ALIASES = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "bangla": "bn",
    "punjabi": "pa",
    "panjabi": "pa",
    "assamese": "as",
    "asamiya": "as",
    "nepali": "ne",
}

_NATIVE_LANGUAGE_ALIASES = {
    "अंग्रेज़ी": "en", "अंग्रेजी": "en", "ইংরেজি": "en", "ইংৰাজী": "en",
    "ਅੰਗਰੇਜ਼ੀ": "en", "अङ्ग्रेजी": "en", "अंग्रेजीमा": "en",
    "हिंदी": "hi", "हिन्दी": "hi", "হিন্দি": "hi", "হিন্দী": "hi", "ਹਿੰਦੀ": "hi",
    "बांग्ला": "bn", "बंगाली": "bn", "বাংলা": "bn", "বাঙালি": "bn",
    "ਬੰਗਾਲੀ": "pa",
    "पंजाबी": "pa", "ਪੰਜਾਬੀ": "pa", "পাঞ্জাবি": "pa", "पञ्जाबी": "pa",
    "असमिया": "as", "असमी": "as", "আসামি": "as", "অসমীয়া": "as",
    "ਅਸਾਮੀ": "as", "অসমীয়াত": "as",
    "नेपाली": "ne", "নেপালি": "ne", "নেপালী": "ne", "ਨੇਪਾਲੀ": "ne",
}

_WORD_RE = re.compile(r"[^\s.,;:!?।॥\"'“”‘’()\[\]{}<>/\\|—–\-]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageDetection:
    """Result of native-script language detection."""

    code: str
    confidence: float
    script: str
    mixed: bool
    evidence: str


def language_name(code: str) -> str:
    """Return the English display name for a language code."""
    return LANGUAGE_NAMES.get(code, LANGUAGE_NAMES["en"])


def language_code(name_or_code: Optional[str]) -> Optional[str]:
    """Accept either a display name or a code and return a code."""
    if not name_or_code:
        return None
    value = str(name_or_code).strip()
    lowered = value.lower()
    if lowered in LANGUAGE_NAMES:
        return lowered
    if value in LANGUAGE_NAMES.values():
        return LANGUAGE_CODES[value]
    return _LATIN_LANGUAGE_ALIASES.get(lowered) or _NATIVE_LANGUAGE_ALIASES.get(value)


def _char_script(ch: str) -> Optional[str]:
    point = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= point <= end:
                return script
    if ch.isalpha() and point < 0x0250:
        return "latin"
    return None


def _script_counts(text: str) -> Dict[str, int]:
    counts = {"latin": 0, "devanagari": 0, "bengali": 0, "gurmukhi": 0}
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        script = _char_script(ch)
        if script in counts:
            counts[script] += 1
    return counts


def _tokens(text: str) -> List[str]:
    return [token.casefold() for token in _WORD_RE.findall(text or "")]


def _marker_score(tokens: List[str], markers: frozenset) -> int:
    """Count exact marker tokens plus safe suffix-bearing forms.

    Devanagari languages inflect heavily (``तपाईं`` -> ``तपाईंले``), so markers
    of four or more characters also match token prefixes.  Short markers stay
    exact-only to avoid accidental substring matches.
    """
    score = 0
    for token in tokens:
        if token in markers:
            score += 1
        elif any(len(marker) >= 4 and token.startswith(marker) for marker in markers):
            score += 1
    return score


def _phrase_score(text: str, phrases: Sequence[str]) -> int:
    """Count whole-word phrase matches so short substrings do not inflate scores."""
    score = 0
    for phrase in phrases:
        score += len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))
    return score


def _margin_confidence(first: int, second: int, default: float = 0.6) -> float:
    total = first + second
    if total == 0:
        return default
    margin = first - second
    if margin >= 3:
        return 0.95
    if margin == 2:
        return 0.85
    if margin == 1:
        return 0.7
    return 0.5


def _calibrated_confidence(first: int, second: int, default: float = 0.6) -> float:
    """Cap confidence when the evidence base is tiny.

    Distinguishing Hindi from Nepali (or Bengali from Assamese) from one shared
    function word is not reliable, so weak evidence is reported as such.
    """
    confidence = _margin_confidence(first, second, default)
    total = first + second
    if total == 0:
        return 0.4
    if total == 1:
        return min(confidence, 0.55)
    if total == 2:
        return min(confidence, 0.75)
    return confidence



def detect_language(text: str) -> LanguageDetection:
    """Detect the language of ``text`` using scripts and function words."""
    if not text or not str(text).strip():
        return LanguageDetection("en", 0.0, "unknown", False, "empty input")

    raw = str(text)
    counts = _script_counts(raw)
    tokens = _tokens(raw)
    indic_total = counts["devanagari"] + counts["bengali"] + counts["gurmukhi"]
    latin_total = counts["latin"]
    total_letters = max(1, indic_total + latin_total)

    if indic_total == 0:
        romanised = sum(1 for token in tokens if token in _ROMANISED_INDIC_MARKERS)
        if latin_total > 0 and romanised >= 2:
            return LanguageDetection(
                "en", 0.55, "latin", False,
                f"Latin script with {romanised} romanised-Indic markers; select a language manually for best results",
            )
        return LanguageDetection("en", 0.95 if latin_total else 0.3, "latin", False, "Latin script")

    # Mostly-Latin text with a couple of Indic tokens is treated as English.
    if latin_total >= 4 * indic_total and indic_total < 6:
        return LanguageDetection(
            "en", 0.75, "latin", True,
            "Mostly Latin script with an isolated Indic phrase",
        )

    mixed = latin_total / total_letters >= 0.12
    dominant = max(
        ("devanagari", counts["devanagari"]),
        ("bengali", counts["bengali"]),
        ("gurmukhi", counts["gurmukhi"]),
        key=lambda item: item[1],
    )[0]

    if dominant == "gurmukhi":
        return LanguageDetection("pa", 0.97, "gurmukhi", mixed, "Gurmukhi script")

    if dominant == "bengali":
        # ৰ (U+09F0) and ৱ (U+09F1) are Assamese-only letters.
        assamese_letters = min(12, raw.count("\u09F0") * 4 + raw.count("\u09F1") * 4)
        assamese = (
            assamese_letters
            + _marker_score(tokens, _ASSAMESE_MARKERS)
            + _phrase_score(raw, _ASSAMESE_PHRASES)
        )
        bengali = _marker_score(tokens, _BENGALI_MARKERS) + _phrase_score(raw, _BENGALI_PHRASES)
        evidence = (
            f"Bengali script; Assamese markers={assamese}, Bengali markers={bengali}"
        )
        if assamese == bengali == 0:
            evidence += "; script only, language is ambiguous here"
        if assamese > bengali:
            return LanguageDetection(
                "as", _calibrated_confidence(assamese, bengali, 0.65), "bengali", mixed,
                evidence,
            )
        return LanguageDetection(
            "bn", _calibrated_confidence(bengali, assamese, 0.6), "bengali", mixed,
            evidence,
        )

    nepali = _marker_score(tokens, _NEPALI_MARKERS) + _phrase_score(raw, _NEPALI_PHRASES)
    hindi = _marker_score(tokens, _HINDI_MARKERS) + _phrase_score(raw, _HINDI_PHRASES)
    evidence = f"Devanagari script; Nepali markers={nepali}, Hindi markers={hindi}"
    if nepali == hindi == 0:
        evidence += "; script only, language is ambiguous here"
    if nepali > hindi:
        return LanguageDetection(
            "ne", _calibrated_confidence(nepali, hindi, 0.6), "devanagari", mixed, evidence,
        )
    return LanguageDetection(
        "hi", _calibrated_confidence(hindi, nepali, 0.6), "devanagari", mixed, evidence,
    )


def target_script(code: str) -> str:
    return {
        "hi": "devanagari",
        "ne": "devanagari",
        "bn": "bengali",
        "as": "bengali",
        "pa": "gurmukhi",
        "en": "latin",
    }.get(code, "latin")


def script_purity(text: str, code: str) -> float:
    """Fraction of letters written in the expected script for ``code``.

    Used as a coarse output-quality guard.  It cannot distinguish Hindi from
    Nepali (same script), but it does catch Latin/other-script corruption.
    """
    if not text:
        return 0.0
    expected = target_script(code)
    expected_count = 0
    total = 0
    for ch in str(text):
        if not unicodedata.category(ch).startswith("L"):
            continue
        total += 1
        if _char_script(ch) == expected:
            expected_count += 1
    if total == 0:
        return 0.0
    return expected_count / total


def contains_indic_script(text: str) -> bool:
    counts = _script_counts(text or "")
    return any(counts[script] for script in ("devanagari", "bengali", "gurmukhi"))


def parse_target_language(text: str) -> Optional[str]:
    """Find a requested target language name in English or native script."""
    if not text:
        return None
    lowered = text.casefold()
    for alias, code in _LATIN_LANGUAGE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return code
    for alias, code in _NATIVE_LANGUAGE_ALIASES.items():
        if alias in text:
            return code
    return None
