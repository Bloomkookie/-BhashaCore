"""BhashaCore Extensible Knowledge Base Layer.

Manages curated FAQs, sampled RAG datasets, domains (education, public services, etc.),
and provides structured retrieval with full provenance, source metadata, and confidence scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DEFAULT_FAQ_PATH, FAQ_AMBIGUITY_MARGIN, FAQ_MATCH_THRESHOLD
from faq_engine import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    are_stem_match,
    content_tokens,
    find_faq,
    get_answer,
    load_faqs,
    normalize,
    similarity,
    tokens,
)


class KnowledgeEntry:
    """Represents a single verified or candidate QA knowledge record."""

    def __init__(
        self,
        entry_id: str,
        topic: str,
        domain: str = "education",
        question_examples: Optional[Dict[str, List[str]]] = None,
        answer: Optional[Dict[str, str]] = None,
        source: Optional[Dict[str, Any]] = None,
        fictional: bool = True,
        translation_status: str = "draft_unverified",
        facts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = entry_id
        self.topic = topic
        self.domain = domain
        self.question_examples = question_examples or {}
        self.answer = answer or {}
        self.source = source or {"type": "local_faq", "reference": "faq_data.json"}
        self.fictional = fictional
        self.translation_status = translation_status
        self.facts = facts or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry into JSON-serializable dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "domain": self.domain,
            "question_examples": self.question_examples,
            "answer": self.answer,
            "source": self.source,
            "fictional": self.fictional,
            "translation_status": self.translation_status,
            "facts": self.facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            entry_id=data.get("id", data.get("topic", "unknown")),
            topic=data.get("topic", data.get("id", "unknown")),
            domain=data.get("domain", "education"),
            question_examples=data.get("question_examples", {}),
            answer=data.get("answer", {}),
            source=data.get("source"),
            fictional=data.get("fictional", True),
            translation_status=data.get("translation_status", "draft_unverified"),
            facts=data.get("facts", []),
            metadata=data.get("metadata", {}),
        )


class KnowledgeBase:
    """Central store supporting multiple domains, provenance tracking and retrieval."""

    def __init__(self) -> None:
        self.entries: List[KnowledgeEntry] = []
        self.domains: Dict[str, List[KnowledgeEntry]] = {}
        self.metadata: Dict[str, Any] = {}

    def load_faq_file(self, path: Optional[Path] = None) -> int:
        """Load entries from a BhashaCore FAQ JSON file."""
        target = Path(path) if path else DEFAULT_FAQ_PATH
        payload = load_faqs(target)
        self.metadata = {
            "project": payload.get("project", "BhashaCore"),
            "schema_version": payload.get("schema_version", "1.0"),
            "purpose": payload.get("purpose", ""),
            "verification_policy": payload.get("verification_policy", {}),
            "languages": payload.get("languages", {}),
        }

        count = 0
        for item in payload.get("faqs", []):
            entry = KnowledgeEntry.from_dict(item)
            self.add_entry(entry)
            count += 1
        return count

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add a single entry and index by domain."""
        self.entries.append(entry)
        domain = entry.domain
        if domain not in self.domains:
            self.domains[domain] = []
        self.domains[domain].append(entry)

    def load_rag_entries(self, entries_list: List[Dict[str, Any]]) -> int:
        """Ingest preprocessed RAG sample entries."""
        count = 0
        for raw in entries_list:
            entry = KnowledgeEntry.from_dict(raw)
            self.add_entry(entry)
            count += 1
        return count

    def get_all_entries(self) -> List[KnowledgeEntry]:
        return list(self.entries)

    def get_entries_by_domain(self, domain: str) -> List[KnowledgeEntry]:
        return list(self.domains.get(domain, []))

    def get_domains(self) -> List[str]:
        return sorted(list(self.domains.keys()))

    def get_stats(self) -> Dict[str, Any]:
        """Return counts of entries, domains, and languages."""
        lang_set = set()
        for entry in self.entries:
            lang_set.update(entry.answer.keys())
            lang_set.update(entry.question_examples.keys())

        return {
            "total_entries": len(self.entries),
            "domains": {k: len(v) for k, v in self.domains.items()},
            "unique_domains_count": len(self.domains),
            "supported_languages": sorted(list(lang_set)),
        }

    def to_faq_payload(self) -> Dict[str, Any]:
        """Convert internal store to format compatible with legacy `find_faq`."""
        return {
            "schema_version": "1.0",
            "project": "BhashaCore",
            "faqs": [e.to_dict() for e in self.entries],
        }

    def search(
        self,
        question: str,
        threshold: float = FAQ_MATCH_THRESHOLD,
        ambiguity_margin: float = FAQ_AMBIGUITY_MARGIN,
        domain_filter: Optional[str] = None,
    ) -> Tuple[Optional[KnowledgeEntry], float, Dict[str, Any]]:
        """Search knowledge base with provenance and attribution metadata.
        
        Returns: (matched_entry or None, similarity_score, provenance_metadata)
        """
        candidate_entries = (
            self.get_entries_by_domain(domain_filter)
            if domain_filter
            else self.entries
        )

        if not question or not question.strip():
            return None, 0.0, {}

        scored: List[Tuple[float, KnowledgeEntry]] = []
        for entry in candidate_entries:
            best_score = 0.0
            for examples in entry.question_examples.values():
                for example in examples:
                    score = similarity(question, example)
                    if score > best_score:
                        best_score = score
            scored.append((best_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < threshold:
            best_sc = scored[0][0] if scored else 0.0
            return None, best_sc, {"status": "rejected_below_threshold"}

        if len(scored) > 1 and (scored[0][0] - scored[1][0] < ambiguity_margin):
            return None, scored[0][0], {
                "status": "rejected_ambiguous",
                "top_candidates": [
                    {"topic": scored[0][1].topic, "score": scored[0][0]},
                    {"topic": scored[1][1].topic, "score": scored[1][0]},
                ],
            }

        top_entry = scored[0][1]
        top_score = scored[0][0]
        provenance = {
            "status": "matched",
            "entry_id": top_entry.id,
            "topic": top_entry.topic,
            "domain": top_entry.domain,
            "source": top_entry.source,
            "fictional": top_entry.fictional,
            "translation_status": top_entry.translation_status,
        }
        return top_entry, top_score, provenance

    def compile_grounding_context(self, max_entries: int = 10) -> str:
        """Compile knowledge entries into a factual reference text for LLM grounding."""
        lines = [
            "### Verified Public Service & Test Reference Data ###",
            "Note: Test scholarship entries are fictional prototypes.",
            "",
        ]
        for entry in self.entries[:max_entries]:
            lines.append(f"Topic: {entry.topic} (Domain: {entry.domain})")
            en_ans = entry.answer.get("en", "")
            if en_ans:
                lines.append(f"Verified Fact / Answer (EN): {en_ans}")
            if entry.facts:
                lines.append(f"Key Facts: {', '.join(entry.facts)}")
            lines.append("")
        return "\n".join(lines)
