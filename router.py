"""Lightweight intent routing for BhashaCore.

Decides whether a turn needs general conversation, document retrieval, FAQ
lookup, translation, summarization, or clarification.  The router deliberately
uses transparent keyword/pattern heuristics rather than a multi-agent system.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from faq_engine import content_tokens
from language_utils import LANGUAGE_NAMES, parse_target_language

ROUTE_GENERAL = "general"
ROUTE_DOCUMENT = "document"
ROUTE_FAQ = "faq"
ROUTE_TRANSLATION = "translation"
ROUTE_SUMMARIZE = "summarize"
ROUTE_CLARIFY = "clarify"

ROUTE_LABELS = {
    ROUTE_GENERAL: "General knowledge",
    ROUTE_DOCUMENT: "Document-grounded",
    ROUTE_FAQ: "Curated FAQ",
    ROUTE_TRANSLATION: "Machine translation",
    ROUTE_SUMMARIZE: "Summary",
    ROUTE_CLARIFY: "Clarification",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    target_lang: Optional[str] = None
    requested_lang: Optional[str] = None
    translation_text: Optional[str] = None
    time_sensitive: bool = False
    document_reference: bool = False
    is_followup: bool = False
    domain_specific: bool = False


_DOCUMENT_CUES = [
    "according to the", "uploaded", "in the document", "in the file", "in the pdf",
    "in the report", "from the document", "from the file", "from the pdf",
    "from the report", "provided document", "provided file", "attachment",
    "the pdf", "the document", "the file", "in the text", "this document", "this file",
    "अपलोड", "फ़ाइल", "फाइल", "पीडीएफ", "के अनुसार", "रिपोर्ट में", "इस दस्तावेज़",
    "इस दस्तावेज", "दस्तावेज़ में", "दस्तावेज में", "फ़ाइल में", "फाइल में",
    "আপলোড", "ফাইল", "পিডিএফ", "অনুযায়ী", "রিপোর্টে", "এই নথি", "এই ডকুমেন্ট",
    "নথিতে", "ফাইলে", "ডকুমেন্টে",
    "আপলোড", "ফাইল", "পিডিএফ", "অনুসৰি", "ৰিপৰ্ট", "এই দস্তাবেজ", "দস্তাবেজত", "ফাইলত",
    "ਅੱਪਲੋਡ", "ਫਾਈਲ", "ਪੀਡੀਐਫ", "ਅਨੁਸਾਰ", "ਰਿਪੋਰਟ", "ਇਸ ਦਸਤਾਵੇਜ਼",
    "ਦਸਤਾਵੇਜ਼ ਵਿੱਚ", "ਫਾਈਲ ਵਿੱਚ",
    "अपलोड", "फाइल", "पीडीएफ", "अनुसार", "यो कागजात", "यो डकुमेन्ट",
    "कागजातमा", "फाइलमा",
]
_TRANSLATION_CUES = [
    "translate", "translation", "how do you say", "how to say", "meaning of",
    "translate this", "translate the",
    "अनुवाद", "अनुवाद कर", "में बदल", "में कहें", "का मतलब", "मतलब क्या",
    "অনুবাদ", "অনুবাদ কর", "অর্থ কী", "অর্থ কি", "বাংলায় বলুন", "বাংলায় বল",
    "অনুবাদ কৰ", "অসমীয়াত", "অৰ্থ কি",
    "ਤਰਜਮਾ", "ਤਰਜੁਮਾ", "ਅਨੁਵਾਦ", "ਵਿੱਚ ਬਦਲੋ", "ਦਾ ਮਤਲਬ", "ਦਾ ਅਰਥ",
    "अनुवाद", "नेपालीमा", "अर्थ के", "मतलब के",
]

_SUMMARY_CUES = [
    "summarize", "summarise", "summary", "tl;dr", "tldr", "short version", "key points",
    "सारांश", "संक्षेप", "संक्षिप्त", "सारांश दें", "संक्षेप में", "सार", "सार दें",
    "সারসংক্ষেপ", "সারমর্ম", "সংক্ষেপে", "সাৰাংশ", "সংক্ষেপ",
    "ਸੰਖੇਪ", "ਸਾਰ", "ਸੰਖੇਪ ਵਿੱਚ",
    "सारांश", "संक्षिप्त", "छोटकरीमा",
]

_TIME_SENSITIVE_CUES = [
    "deadline", "last date", "current", "today", "latest", "now", "price", "fee",
    "cost", "statistics", "how much", "2026", "2027",
    "अंतिम तिथि", "आखिरी तारीख", "समय सीमा", "आज", "वर्तमान", "नवीनतम", "कीमत", "शुल्क", "आंकड़े",
    "শেষ তারিখ", "সময়সীমা", "আজ", "বর্তমান", "সর্বশেষ", "মূল্য", "ফি", "পরিসংখ্যান",
    "শেষ তাৰিখ", "সময়সীমা", "আজি", "বৰ্তমান", "শেহতীয়া", "মূল্য", "ফি",
    "ਆਖਰੀ ਤਾਰੀਖ", "ਸਮਾਂ ਸੀਮਾ", "ਅੱਜ", "ਮੌਜੂਦਾ", "ਨਵੀਨਤਮ", "ਕੀਮਤ", "ਫੀਸ", "ਅੰਕੜੇ",
    "अन्तिम मिति", "समयसीमा", "आज", "वर्तमान", "नयाँ", "मूल्य", "शुल्क", "तथ्याङ्क",
]

_DOMAIN_CUES = [
    "scholarship", "scheme", "government", "eligibility", "documents", "apply",
    "application", "amount", "contact", "ration card", "pension", "subsidy",
    "benefit", "yojana", "public service",
    "छात्रवृत्ति", "योजना", "सरकारी", "पात्रता", "दस्तावेज़", "आवेदन", "राशन", "पेंशन", "सब्सिडी", "लाभ",
    "বৃত্তি", "প্রকল্প", "সরকারি", "যোগ্যতা", "নথি", "আবেদন", "রেশন", "পেনশন", "ভর্তুকি", "সুবিধা",
    "ਵਜ਼ੀਫ਼ਾ", "ਯੋਜਨਾ", "ਸਰਕਾਰੀ", "ਯੋਗਤਾ", "ਦਸਤਾਵੇਜ਼", "ਅਰਜ਼ੀ", "ਰਾਸ਼ਨ", "ਪੈਨਸ਼ਨ", "ਸਬਸਿਡੀ", "ਲਾਭ",
    "বৃত্তি", "আঁচনি", "চৰকাৰী", "যোগ্যতা", "আবেদন", "ৰেচন", "পেঞ্চন", "সুবিধা",
    "छात्रवृत्ति", "योजना", "सरकारी", "योग्यता", "कागजात", "आवेदन", "राशन", "पेन्शन", "अनुदान", "लाभ",
]

_FOLLOWUP_CUES = [
    "it", "this", "that", "that one", "the previous", "previous answer", "more",
    "again", "another", "example", "for example", "simpler", "explain", "and",
    "what about", "why", "how", "then", "also",
    "इसका", "उसका", "इसे", "उसे", "यह", "वह", "फिर", "अब", "और", "उदाहरण", "फिर से", "और बताएं",
    "এটা", "ওটা", "এটি", "সেটি", "এর", "ওর", "আরও", "এবার", "উদাহরণ", "আবার", "আরও বলুন",
    "এইটো", "সেইটো", "ইয়াৰ", "তাৰ", "আৰু", "এতিয়া", "উদাহৰণ", "আকৌ", "অধিক",
    "ਇਹ", "ਉਹ", "ਇਸ", "ਉਸ", "ਹੋਰ", "ਫਿਰ", "ਹੁਣ", "ਉਦਾਹਰਨ", "ਦੁਬਾਰਾ", "ਹੋਰ ਦੱਸੋ",
    "यो", "त्यो", "यस", "त्यस", "अझ", "फेरि", "अहिले", "उदाहरण", "अर्को", "थप",
]

_QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन", "কী", "কি", "কেন", "কীভাবে", "কোথায়", "কে", "ਕੀ", "ਕਿਉਂ", "ਕਿਵੇਂ", "ਕਿੱਥੇ", "ਕੌਣ", "कुन", "किन", "कसरी", "कहाँ", "कसले", "के", "कि"}

_ENGLISH_NAME_PATTERN = (
    r"(english|hindi|bengali|bangla|punjabi|panjabi|assamese|asamiya|nepali)"
)


def _wordish(char: str) -> bool:
    """Indic combining marks are word characters even when regex ``\\w`` says no."""
    return bool(char) and (
        char.isalnum() or unicodedata.category(char).startswith("M")
    )


def _bounded_match(text: str, candidate: str) -> bool:
    start = text.find(candidate)
    while start != -1:
        before = text[start - 1] if start > 0 else ""
        end = start + len(candidate)
        after = text[end] if end < len(text) else ""
        if not _wordish(before) and not _wordish(after):
            return True
        start = text.find(candidate, start + 1)
    return False


def _contains_any(text: str, phrases: Iterable[str]) -> Optional[str]:
    lowered = text.casefold()
    for phrase in phrases:
        candidate = phrase.casefold()
        # Short non-ASCII cues need boundaries too: Punjabi "ਸਾਰ" (summary)
        # otherwise matches inside "ਅਨੁਸਾਰ" (according to).
        if candidate.isascii():
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", lowered):
                return phrase
        elif len(candidate) <= 4:
            if _bounded_match(lowered, candidate):
                return phrase
        elif candidate in lowered:
            return phrase
    return None


def contains_document_reference(text: str) -> bool:
    return _contains_any(text, _DOCUMENT_CUES) is not None


def contains_translation_cue(text: str) -> bool:
    return _contains_any(text, _TRANSLATION_CUES) is not None


def contains_summary_cue(text: str) -> bool:
    return _contains_any(text, _SUMMARY_CUES) is not None


def is_time_sensitive(text: str) -> bool:
    return _contains_any(text, _TIME_SENSITIVE_CUES) is not None


def is_domain_specific(text: str) -> bool:
    return _contains_any(text, _DOMAIN_CUES) is not None


def is_followup(text: str, has_history: bool) -> bool:
    if not has_history:
        return False
    tokens = list(content_tokens(text))
    if len(tokens) <= 2:
        return True
    return _contains_any(text, _FOLLOWUP_CUES) is not None


def detect_requested_language(text: str) -> Optional[str]:
    """Detect an explicit 'answer in X' request without routing to translation."""
    lowered = text.casefold()
    match = re.search(rf"\bin\s+(?:simple\s+|easy\s+|plain\s+)?{_ENGLISH_NAME_PATTERN}\b", lowered)
    if match:
        return parse_target_language(match.group(1))
    match = re.search(rf"\b(?:answer|reply|respond|explain|tell|describe|write)\b.*?\b{_ENGLISH_NAME_PATTERN}\b", lowered)
    if match:
        return parse_target_language(match.group(1))

    native_patterns = [
        (r"(हिंदी|हिन्दी)\s*(?:में|मे)", "hi"),
        (r"(नेपाली)\s*(?:मा|म)", "ne"),
        (r"(বাংলা|বাংলায়)\s*", "bn"),
        (r"(অসমীয়া|অসমীয়াত)\s*", "as"),
        (r"(ਪੰਜਾਬੀ)\s*ਵਿੱਚ", "pa"),
        (r"(ਅੰਗਰੇਜ਼ੀ)\s*ਵਿੱਚ", "en"),
    ]
    for pattern, code in native_patterns:
        if re.search(pattern, text):
            return code
    return None


def _extract_quoted(text: str) -> Optional[str]:
    for pattern in (r'"([^"]{4,})"', r"“([^”]{4,})”", r"'([^']{8,})'"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_translation_text(text: str) -> Optional[str]:
    """Extract the payload the user wants translated, if included inline."""
    quoted = _extract_quoted(text)
    if quoted:
        return quoted

    lowered = text.casefold()
    patterns = [
        rf"translate\s+(?:this|the following|the text|the sentence)?\s*[:\-]\s*(.+)",
        rf"translate\s+(.+?)\s+(?:in|to|into)\s+{_ENGLISH_NAME_PATTERN}\s*[.!?]?$",
        rf"how do you say\s+(.+?)\s+in\s+{_ENGLISH_NAME_PATTERN}\s*[.!?]?$",
        rf"(?:meaning|translation)\s+of\s+(.+?)\s+in\s+{_ENGLISH_NAME_PATTERN}\s*[.!?]?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered if pattern.startswith("translate") else text, re.DOTALL)
        if match:
            payload = match.group(1).strip(" :-\n\t")
            if len(payload) >= 3:
                return payload

    if ":" in text:
        payload = text.split(":", 1)[1].strip()
        if len(payload) >= 8:
            return payload

    # Native-script commands usually refer to the previous answer.
    return None


def route(
    message: str,
    has_documents: bool = False,
    has_history: bool = False,
    last_route: Optional[str] = None,
) -> RouteDecision:
    """Classify the user's turn into a coarse route."""
    text = (message or "").strip()
    if not text:
        return RouteDecision(ROUTE_CLARIFY, "empty message")

    tokens = text.split()
    content = content_tokens(text)
    translation_cue = contains_translation_cue(text)
    summary_cue = contains_summary_cue(text)
    document_reference = contains_document_reference(text)

    followup = is_followup(text, has_history)
    time_sensitive = is_time_sensitive(text)
    domain_specific = is_domain_specific(text)

    if not has_history and len(tokens) <= 2 and not content:
        lowered_tokens = {token.casefold().strip("?？!.,।॥") for token in tokens}
        if lowered_tokens & _QUESTION_WORDS:
            return RouteDecision(ROUTE_CLARIFY, "bare question word without enough context")
        if not re.search(r"[?？]", text):
            return RouteDecision(ROUTE_CLARIFY, "very short message without a clear request")

    if translation_cue:
        target = parse_target_language(text)
        payload = extract_translation_text(text)
        return RouteDecision(
            ROUTE_TRANSLATION,
            "explicit translation request",
            target_lang=target,
            translation_text=payload,
            time_sensitive=time_sensitive,
            document_reference=document_reference,
            is_followup=followup,
            domain_specific=domain_specific,
        )

    if summary_cue:
        return RouteDecision(
            ROUTE_SUMMARIZE,
            "explicit summary request",
            target_lang=detect_requested_language(text),
            requested_lang=detect_requested_language(text),
            document_reference=document_reference,
            is_followup=followup,
            domain_specific=domain_specific,
        )

    if document_reference:
        if not has_documents:
            return RouteDecision(
                ROUTE_DOCUMENT,
                "document reference but no documents are loaded",
                target_lang=detect_requested_language(text),
                requested_lang=detect_requested_language(text),
                document_reference=True,
                is_followup=followup,
                time_sensitive=time_sensitive,
                domain_specific=domain_specific,
            )
        return RouteDecision(
            ROUTE_DOCUMENT,
            "explicit document reference",
            target_lang=detect_requested_language(text),
            requested_lang=detect_requested_language(text),
            document_reference=True,
            is_followup=followup,
            time_sensitive=time_sensitive,
            domain_specific=domain_specific,
        )

    if has_documents and last_route == ROUTE_DOCUMENT and followup:
        return RouteDecision(
            ROUTE_DOCUMENT,
            "follow-up to a document-grounded answer",
            target_lang=detect_requested_language(text),
            requested_lang=detect_requested_language(text),
            document_reference=True,
            is_followup=True,
            time_sensitive=time_sensitive,
            domain_specific=domain_specific,
        )

    return RouteDecision(
        ROUTE_GENERAL,
        "general conversational question",
        target_lang=detect_requested_language(text),
        requested_lang=detect_requested_language(text),
        is_followup=followup,
        time_sensitive=time_sensitive,
        domain_specific=domain_specific,
    )


def describe_route(route_name: str) -> str:
    return ROUTE_LABELS.get(route_name, route_name)
