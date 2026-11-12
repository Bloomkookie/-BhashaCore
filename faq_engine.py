"""BhashaCore FAQ Matching Engine.

Provides Unicode-aware text normalization, multilingual tokenization,
content-word scoring, and confidence-gated FAQ retrieval across
English, Hindi, Bengali, Punjabi, Assamese, and Nepali.

This engine uses only the Python standard library, making it lightweight,
fast, deterministic, and free of heavyweight machine learning dependencies.
"""

from __future__ import annotations

import json
import math
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAQ_PATH = BASE_DIR / "faq_data.json"

LANGUAGE_LABELS: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Assamese": "as",
    "Nepali": "ne",
}

LANGUAGE_CODES: Dict[str, str] = {code: name for name, code in LANGUAGE_LABELS.items()}

DEFAULT_MATCH_THRESHOLD = 0.40
DEFAULT_AMBIGUITY_MARGIN = 0.04

# Multilingual stopwords across all 6 supported languages.
# Filtering these prevents common grammatical words (such as "is", "what",
# "for", "का", "की", "কি", "জন্য", "ਦੀ", "বাবে", "लागि") from artificially
# triggering false-positive matches on completely unrelated questions.
STOP_WORDS: Set[str] = {
    # Domain terms common across all topics (treated as non-discriminating)
    "scholarship", "scholarships", "scheme",
    "छात्रवृत्ति", "छात्रवृत्तिका", "छात्रवृत्तिमा",
    "বৃত্তি", "বৃত্তির", "বৃত্তিতে", "বৃত্তিত",
    "ਵਜ਼ੀਫ਼ਾ", "ਵਜ਼ੀਫ਼ੇ", "ਵਜ਼ੀਫ਼ਿਆਂ", "ਵਜ਼ੀਫਾ", "ਵਜ਼ੀਫੇ",
    "वज़ीफ़ा", "वजीफा",
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "mine", "yours", "ours",
    "for", "of", "in", "to", "on", "at", "by", "with", "from", "about", "into",
    "through", "during", "before", "after", "above", "below",
    "this", "that", "these", "those", "there", "here",
    "please", "tell", "give", "show", "know", "want", "need",
    # Hindi
    "का", "की", "के", "को", "में", "से", "पर", "लिए", "है", "हैं", "हो", "था", "थी", "थे",
    "क्या", "कौन", "कौन-कौन", "किस", "किसे", "कहाँ", "कब", "कैसे", "क्यों",
    "मैं", "मुझे", "मेरा", "मेरी", "मेरे", "आप", "आपको", "आपका", "आपकी", "आपके",
    "यह", "इस", "वह", "उस", "इन", "उन", "एक", "दो", "सब", "सभी",
    "होगा", "होगी", "होंगे", "दें", "करें", "सकता", "सकती", "सकते", "कृपया",
    # Bengali
    "এর", "কে", "তে", "এ", "র", "জন্য", "হলো", "হয়", "আছে", "ছিল",
    "কী", "কি", "কোন", "কোন-কোন", "কারা", "কাকে", "কোথায়", "কখন", "কীভাবে", "কেন",
    "আমি", "আমাকে", "আমার", "আপনি", "আপনার", "এই", "সেই", "সব", "সকল",
    "দিন", "করুন", "দিতে", "হবে", "পারে", "দয়া", "করে",
    # Punjabi (Gurmukhi)
    "ਦਾ", "ਦੀ", "ਦੇ", "ਨੂੰ", "ਵਿੱਚ", "ਤੋਂ", "ਤੇ", "ਲਈ", "ਹੈ", "ਹਨ", "ਸੀ", "ਸਨ",
    "ਕੀ", "ਕਿਹੜੇ", "ਕਿਹੜੀ", "ਕੌਣ", "ਕਿੱਥੇ", "ਕਦੋਂ", "ਕਿਵੇਂ", "ਕਿਉਂ",
    "ਮੈਂ", "ਮੈਨੂੰ", "ਮੇਰਾ", "ਮੇਰੀ", "ਤੁਸੀਂ", "ਤੁਹਾਡਾ", "ਤੁਹਾਡੀ", "ਇਹ", "ਉਹ",
    "ਸਾਰੇ", "ਸਾਰੀ", "ਦੱਸੋ", "ਕਰੋ", "ਸਕਦਾ", "ਸਕਦੀ", "ਕਿਰਪਾ", "ਕਰਕੇ",
    # Assamese
    "ৰ", "লৈ", "ত", "বাবে", "হ’ল", "হয়", "হয়", "আছে", "আছিল",
    "কি", "কি-কি", "কোনে", "কাৰ", "ক’ত", "কেতিয়া", "কেনেকৈ", "কিয়",
    "মই", "মোক", "মোৰ", "আপুনি", "আপোনাৰ", "এই", "সেই", "সকলো",
    "দিয়ক", "কৰক", "লাগে", "পাৰে", "অনুগ্ৰহ",
    # Nepali
    "को", "का", "की", "लाई", "मा", "बाट", "लागि", "हो", "हुन्", "छ", "छन्", "थियो",
    "के", "कुन", "कुन-कुन", "कसले", "कसलाई", "कहाँ", "कहिले", "कसरी", "किन",
    "म", "मलाई", "मेरो", "तपाईं", "तपाईंको", "यो", "यस", "त्यो", "सबै",
    "दिनुहोस्", "गर्नुहोस्", "सक्छ", "सक्छन्", "कृपया",
}


def normalize(text: str) -> str:
    """Normalize text while carefully preserving Indic combining marks.

    Unlike standard regex [^\\w\\s] which strips Unicode combining marks (matras,
    halants, viramas) and shreds Indic words into disconnected consonants,
    this normalizer preserves:
      - Letters (category L: Lu, Ll, Lt, Lm, Lo)
      - Combining Marks (category M: Mn, Mc, Me) such as Indic matras and viramas
      - Numbers (category N: Nd, Nl, No)
      - Whitespace

    Punctuation (P) and Symbols (S) are converted to spaces.
    Unicode normalization form NFC is applied first so characters remain composed.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)).casefold()
    cleaned: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "M", "N")) or ch.isspace():
            cleaned.append(ch)
        elif cat == "Cf":
            # Zero-width joiner (ZWJ) or non-joiner (ZWNJ) used in Indic conjuncts
            continue
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def tokens(text: str) -> Set[str]:
    """Return the set of all normalized tokens in text."""
    norm = normalize(text)
    if not norm:
        return set()
    return set(norm.split())


def content_tokens(text: str) -> Set[str]:
    """Return non-stopword tokens that carry semantic intent."""
    return {t for t in tokens(text) if t not in STOP_WORDS}


def are_stem_match(w1: str, w2: str) -> bool:
    """Check if two words share a common root or grammatical inflection.

    Handles common Indic inflections (e.g. दस्तावेज़/दस्तावेज़ों, कागजात/कागजातहरू,
    নথি/নথিগুলি, বৃত্তি/বৃত্তির) and Latin plural/tense suffixes.
    """
    if w1 == w2:
        return True
    len1, len2 = len(w1), len(w2)
    if len1 >= 3 and len2 >= 3:
        if w1.startswith(w2) or w2.startswith(w1):
            return True
        common_len = 0
        for c1, c2 in zip(w1, w2):
            if c1 == c2:
                common_len += 1
            else:
                break
        max_len = max(len1, len2)
        if common_len >= 3 and (common_len / max_len) >= 0.70:
            return True
    return False


def soft_token_overlap(set1: Set[str], set2: Set[str]) -> Tuple[float, float]:
    """Compute soft Jaccard and Dice scores between two token sets.

    Accounts for morphological inflection matching using `are_stem_match`.
    Returns (jaccard, dice).
    """
    if not set1 or not set2:
        return 0.0, 0.0
    matched_1 = sum(1 for t1 in set1 if any(are_stem_match(t1, t2) for t2 in set2))
    matched_2 = sum(1 for t2 in set2 if any(are_stem_match(t1, t2) for t1 in set1))
    intersect = (matched_1 + matched_2) / 2.0
    union = len(set1) + len(set2) - intersect
    jaccard = intersect / max(1.0, union)
    dice = (2.0 * intersect) / (len(set1) + len(set2))
    return jaccard, dice


def similarity(question: str, example: str) -> float:
    """Calculate semantic similarity between user query and an FAQ example.

    Combines:
    1. Content-token Jaccard and Dice scoring (primary intent signal)
    2. Character SequenceMatcher (robust against typos and spelling variants)
    3. Conservative fallbacks to prevent unrelated queries from scoring high.
    """
    q_norm = normalize(question)
    e_norm = normalize(example)
    if not q_norm or not e_norm:
        return 0.0
    if q_norm == e_norm:
        return 1.0

    q_all = tokens(question)
    e_all = tokens(example)
    q_c = content_tokens(question)
    e_c = content_tokens(example)

    seq_ratio = SequenceMatcher(None, q_norm, e_norm).ratio()

    # Content-token comparison
    if q_c and e_c:
        jaccard, dice = soft_token_overlap(q_c, e_c)
        if len(q_c) == 1 and len(e_c) >= 3:
            word_score = jaccard
        else:
            word_score = (jaccard + dice) / 2.0

        if word_score >= 0.30:
            # Significant content overlap: combine with sequence ratio for typo/inflection tolerance
            return (0.65 * word_score) + (0.35 * (seq_ratio * 0.85))
        elif word_score > 0:
            # Incidental token overlap (1 common word out of many): prevent sequence ratio from inflating
            return min(word_score * 0.80, seq_ratio * 0.35)

    # If no content words matched:
    # Check if full tokens (including query question words) overlap
    if q_all and e_all:
        all_jaccard, _ = soft_token_overlap(q_all, e_all)
        if all_jaccard > 0:
            # Heavily discount when only function/stop words overlap
            return min(all_jaccard * 0.40, seq_ratio * 0.35)

    # Completely disjoint words: cap character sequence ratio strictly
    return seq_ratio * 0.30


def load_faqs(faq_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the FAQ knowledge base.

    Raises:
        FileNotFoundError: If the FAQ file does not exist.
        ValueError: If the file does not conform to the expected schema.
    """
    target_path = Path(faq_path) if faq_path else DEFAULT_FAQ_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Could not find {target_path.name}. Keep it at {target_path.resolve()}."
        )

    with target_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("faq_data.json must contain a top-level JSON object.")
    if not isinstance(payload.get("faqs"), list):
        raise ValueError("faq_data.json must contain a 'faqs' list.")
    if len(payload["faqs"]) == 0:
        raise ValueError("faq_data.json 'faqs' list cannot be empty.")

    return payload


def find_faq(
    question: str,
    payload: Dict[str, Any],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Find the best-matching FAQ entry for a given user question.

    Matches the user's question against examples across all supported languages.
    Returns (faq, confidence_score).

    If the top match is below `threshold` or within `ambiguity_margin` of the
    runner-up topic, returns (None, score) to avoid hallucinating or guessing.
    """
    if not question or not question.strip():
        return None, 0.0

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for faq in payload.get("faqs", []):
        best_score = 0.0
        for examples in faq.get("question_examples", {}).values():
            for example in examples:
                score = similarity(question, example)
                if score > best_score:
                    best_score = score
        scored.append((best_score, faq))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored or scored[0][0] < threshold:
        return None, (scored[0][0] if scored else 0.0)

    # Ambiguity check: if the top two topics have virtually indistinguishable
    # confidence scores, decline to guess rather than giving the wrong advice.
    if len(scored) > 1 and (scored[0][0] - scored[1][0] < ambiguity_margin):
        return None, scored[0][0]

    return scored[0][1], scored[0][0]


def get_answer(faq: Dict[str, Any], lang_code: str) -> Optional[str]:
    """Retrieve the answer string for an FAQ in the requested language."""
    if not faq or not isinstance(faq, dict):
        return None
    answers = faq.get("answer", {})
    if not isinstance(answers, dict):
        return None
    return answers.get(lang_code)


def format_fallback_message(reason: str = "no_match") -> str:
    """Return a standard fallback message when an answer cannot be provided."""
    if reason == "missing_translation":
        return (
            "A reviewed answer is not available in the selected language yet. "
            "Please choose another language or contact the project team."
        )
    return (
        "I couldn't confidently match that question to the current FAQ. "
        "Please rephrase it or choose one of the example questions below. "
        "I won't guess an answer."
    )
