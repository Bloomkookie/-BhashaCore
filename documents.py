"""Document ingestion and hybrid multilingual retrieval for BhashaCore.

Supports PDF, TXT, Markdown, JSON, and CSV.  Chunks preserve document, page,
section, and detected-language metadata.  Retrieval fuses lexical BM25 with
bge-m3 multilingual embeddings through a weighted score, so a Hindi question
can retrieve an English passage without a separate translation step.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_ENABLED,
    EMBEDDING_MODEL,
    MAX_DOCUMENT_CHARS,
    RETRIEVAL_MIN_LEXICAL,
    RETRIEVAL_MIN_SEMANTIC,
    RETRIEVAL_TOP_K,
)
from faq_engine import content_tokens, normalize
from language_utils import detect_language


@dataclass
class DocumentInfo:
    doc_id: str
    name: str
    file_type: str
    size_bytes: int
    pages: Optional[int]
    chunks: int
    languages: List[str]
    added_at: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    lang: str = "en"
    char_start: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float = 0.0

    @property
    def doc_name(self) -> str:
        return self.chunk.doc_name

    @property
    def page(self) -> Optional[int]:
        return self.chunk.page

    @property
    def section(self) -> Optional[str]:
        return self.chunk.section

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class RetrievalResult:
    query: str
    chunks: List[RetrievedChunk]
    evidence_found: bool
    method: str
    warnings: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    best_lexical: float = 0.0
    best_semantic: float = 0.0


@dataclass
class IngestionResult:
    doc: Optional[DocumentInfo]
    chunks: List[Chunk] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class Citation:
    index: int
    doc_name: str
    page: Optional[int]
    section: Optional[str]
    excerpt: str
    chunk_id: str
    cited: bool = False

    def label(self) -> str:
        location = []
        if self.page:
            location.append(f"p. {self.page}")
        if self.section:
            location.append(self.section)
        suffix = f" ({', '.join(location)})" if location else ""
        return f"[{self.index}] {self.doc_name}{suffix}"


class _BM25Index:
    """IDF-weighted lexical index.

    Scores are absolute query-coverage values in ``[0, 1]`` rather than
    peak-normalised ranks: the share of the query's IDF mass that a chunk
    actually contains.  Peak normalisation made any partial match score 1.0 in
    a single-document corpus, defeating the evidence threshold.
    """

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokens) for tokens in tokenized_docs]
        self.doc_lens = [len(tokens) for tokens in tokenized_docs]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        doc_freq: Counter = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] += 1
        total = len(tokenized_docs)
        self.idf = {
            token: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        self.default_idf = math.log(1.0 + (total + 0.5) / 0.5)

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        if not self.term_freqs or not query_tokens:
            return [0.0] * len(self.term_freqs)
        unique = list(dict.fromkeys(query_tokens))
        total_weight = sum(self.idf.get(token, self.default_idf) for token in unique)
        if total_weight <= 0:
            return [0.0] * len(self.term_freqs)

        scores: List[float] = []
        for freqs, doc_len in zip(self.term_freqs, self.doc_lens):
            matched_weight = 0.0
            matched_terms = 0
            total_tf = 0
            for token in unique:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                matched_weight += self.idf.get(token, self.default_idf)
                matched_terms += 1
                total_tf += tf
            if not matched_terms:
                scores.append(0.0)
                continue
            coverage = matched_weight / total_weight
            length_penalty = (self.avgdl / max(1.0, doc_len)) ** 0.15
            intensity = min(1.0, 0.7 + 0.3 * min(1.0, total_tf / 3.0))
            scores.append(min(1.0, coverage * intensity * length_penalty))
        return scores


def _lexical_tokens(text: str) -> List[str]:
    terms = sorted(content_tokens(text))
    if terms:
        return terms
    # Fall back to all normalized tokens only when stopword filtering empties
    # the query (for example a single word such as "what").
    return [token for token in normalize(text).split() if len(token) >= 2]


class OllamaEmbeddingBackend:
    """bge-m3 embeddings through the local Ollama daemon."""

    def __init__(self, model: str = EMBEDDING_MODEL, enabled: bool = EMBEDDING_ENABLED):
        self.model = model
        self.enabled = enabled
        self.last_error: Optional[str] = None
        self._checked = False
        self._available = False

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._checked:
            return self._available
        from llm_client import embedding_model_available

        self._available = embedding_model_available(self.model)
        self._checked = True
        return self._available

    def embed(self, texts: List[str], keep_alive: str = "0s") -> Optional[np.ndarray]:
        if not texts or not self.available():
            return None
        from llm_client import embed_texts

        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            result = embed_texts(batch, model=self.model, keep_alive=keep_alive)
            if result["status"] != "success":
                self.last_error = result.get("error")
                return None
            vectors.extend(result["embeddings"])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class _NullEmbedder:
    def available(self) -> bool:
        return False

    def embed(self, texts: List[str], keep_alive: str = "0s") -> Optional[np.ndarray]:
        return None


def _chunk_sentences(
    text: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    base_offset: int = 0,
) -> List[Tuple[str, int]]:
    clean = re.sub(r"[ \t]+", " ", text or "").strip()
    if not clean:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?।॥])\s+|\n+", clean) if part.strip()]
    pieces: List[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:
            for start in range(0, len(sentence), max_chars):
                pieces.append(sentence[start : start + max_chars])

    chunks: List[Tuple[str, int]] = []
    current = ""
    current_start = base_offset
    cursor = base_offset
    for piece in pieces:
        if not current:
            current_start = cursor
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append((current, current_start))
            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else ""
            current = f"{tail} {piece}".strip() if tail else piece
            current_start = max(base_offset, cursor - len(tail))
        cursor += len(piece) + 1
    if current:
        chunks.append((current, current_start))
    return chunks


def _markdown_sections(text: str) -> List[Tuple[Optional[str], str, int]]:
    sections: List[Tuple[Optional[str], str, int]] = []
    title: Optional[str] = None
    buffer: List[str] = []
    start = 0
    cursor = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", line)
        if match:
            if buffer and "".join(buffer).strip():
                sections.append((title, "".join(buffer), start))
            title = match.group(1).strip()
            buffer = [line]
            start = cursor
        else:
            buffer.append(line)
        cursor += len(line)
    if buffer and "".join(buffer).strip():
        sections.append((title, "".join(buffer), start))
    return sections or [(None, text, 0)]


def _extract_pdf(data: bytes) -> List[Tuple[Optional[str], int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Encrypted PDF could not be opened: {exc}") from exc
    pages: List[Tuple[Optional[str], int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((None, number, text))
    return pages


def _extract_json(text: str) -> List[Tuple[Optional[str], Optional[int], str]]:
    payload = json.loads(text)
    collected: List[Tuple[str, str]] = []

    def walk(node: Any, section: str, depth: int = 0) -> None:
        if len(collected) > 2000 or depth > 8:
            return
        if isinstance(node, dict):
            if "question" in node and ("answer" in node or "answers" in node):
                question = str(node.get("question", "")).strip()
                answer = node.get("answer") or node.get("answers")
                if isinstance(answer, dict):
                    answer = " | ".join(str(v) for v in answer.values() if str(v).strip())
                collected.append((section, f"Question: {question}\nAnswer: {answer}"))
                return
            for key, value in node.items():
                child_section = section or str(key)
                walk(value, child_section, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, section, depth + 1)
        elif isinstance(node, (str, int, float)) and str(node).strip():
            collected.append((section, str(node).strip()))

    walk(payload, "")
    if not collected:
        collected = [("", json.dumps(payload, ensure_ascii=False, indent=1))]
    grouped: List[Tuple[Optional[str], Optional[int], str]] = []
    for section, line in collected:
        if grouped and grouped[-1][0] == (section or None):
            grouped[-1] = (grouped[-1][0], None, grouped[-1][2] + "\n" + line)
        else:
            grouped.append((section or None, None, line))
    return grouped


def _extract_csv(text: str) -> List[Tuple[Optional[str], Optional[int], str]]:
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    out: List[Tuple[Optional[str], Optional[int], str]] = []
    for index, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        pairs = []
        for key, value in zip(header, row):
            if value.strip():
                pairs.append(f"{key.strip()}: {value.strip()}")
        if pairs:
            out.append((None, None, f"Row {index}: " + "; ".join(pairs)))
    return out


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class DocumentStore:
    """Session-scoped document collection with citation-preserving retrieval."""

    def __init__(
        self,
        embedder: Optional[Any] = None,
        use_embeddings: bool = EMBEDDING_ENABLED,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: List[Chunk] = []
        self.embedder = embedder if embedder is not None else (OllamaEmbeddingBackend() if use_embeddings else _NullEmbedder())
        self.top_k = top_k
        self._embeddings: Optional[np.ndarray] = None
        self._bm25: Optional[_BM25Index] = None
        self._embeddings_disabled = False
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_document(
        self,
        name: str,
        data: bytes,
        sources: Optional[List[Tuple[Optional[str], Optional[int], str]]] = None,
    ) -> IngestionResult:
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return IngestionResult(
                None,
                error=f"Unsupported file type '{suffix or 'unknown'}'. Allowed: PDF, TXT, MD, JSON, CSV.",
            )
        if not data:
            return IngestionResult(None, error="The uploaded file is empty.")

        doc_id = hashlib.sha1(data).hexdigest()[:12]
        if doc_id in self.documents:
            return IngestionResult(
                self.documents[doc_id],
                warnings=["This exact file is already loaded; reusing the existing index."],
            )

        if sources is None:
            try:
                sources = self._extract(name, data)
            except Exception as exc:
                return IngestionResult(None, error=f"Could not read '{name}': {exc}")

        total_text = "\n".join(item[2] for item in sources)
        if not total_text.strip():
            return IngestionResult(
                None,
                error="No extractable text was found. Scanned/image-only PDFs are not supported.",
            )
        if len(total_text) > MAX_DOCUMENT_CHARS:
            return IngestionResult(
                None,
                error=f"Document is too large ({len(total_text):,} characters; limit {MAX_DOCUMENT_CHARS:,}).",
            )

        new_chunks: List[Chunk] = []
        pages = set()
        languages = set()
        for section, page, text in sources:
            if page:
                pages.add(page)
            for chunk_text, offset in _chunk_sentences(text, base_offset=0):
                if not chunk_text.strip():
                    continue
                lang = detect_language(chunk_text).code
                languages.add(lang)
                new_chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{len(new_chunks):04d}",
                        doc_id=doc_id,
                        doc_name=name,
                        text=chunk_text,
                        page=page,
                        section=section,
                        lang=lang,
                        char_start=offset,
                    )
                )
                if len(new_chunks) > 4000:
                    break
            if len(new_chunks) > 4000:
                self.warnings.append("Document truncated to the first 4000 chunks.")
                break

        if not new_chunks:
            return IngestionResult(None, error="Document text could not be split into chunks.")

        warnings: List[str] = []
        vectors = None
        if not self._embeddings_disabled:
            vectors = self.embedder.embed([chunk.text for chunk in new_chunks], keep_alive="30s")

        if vectors is not None and len(vectors) == len(new_chunks) and not self._embeddings_disabled:
            self._embeddings = (
                vectors
                if self._embeddings is None or len(self._embeddings) == 0
                else np.vstack([self._embeddings, vectors])
            )
        else:
            self._embeddings_disabled = True
            self._embeddings = None
            if self.embedder.available():
                error = getattr(self.embedder, "last_error", None)
                warnings.append(
                    "Semantic embeddings were unavailable for this document; lexical retrieval only."
                    + (f" ({error})" if error else "")
                )
            else:
                warnings.append("Embedding model bge-m3 is not installed; using lexical retrieval only.")

        self.chunks.extend(new_chunks)
        self._bm25 = _BM25Index([_lexical_tokens(chunk.text) for chunk in self.chunks])

        info = DocumentInfo(
            doc_id=doc_id,
            name=name,
            file_type=suffix.lstrip("."),
            size_bytes=len(data),
            pages=max(pages) if pages else None,
            chunks=len(new_chunks),
            languages=sorted(languages),
            added_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.documents[doc_id] = info
        return IngestionResult(info, chunks=new_chunks, warnings=warnings)

    def add_text(self, name: str, text: str, section: Optional[str] = None) -> IngestionResult:
        data = text.encode("utf-8")
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            name = f"{name}.txt"
        return self.add_document(name, data, sources=[(section, None, text)])

    def _extract(self, name: str, data: bytes) -> List[Tuple[Optional[str], Optional[int], str]]:
        suffix = Path(name).suffix.lower()
        if suffix == ".pdf":
            return _extract_pdf(data)
        text = _decode_text(data)
        if suffix in (".md", ".markdown"):
            return [(section, None, body) for section, body, _ in _markdown_sections(text)]
        if suffix == ".json":
            return _extract_json(text)
        if suffix == ".csv":
            return _extract_csv(text)
        return [(None, None, text)]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        extra_queries: Optional[Sequence[str]] = None,
        keep_alive: str = "0s",
    ) -> RetrievalResult:
        started = time.perf_counter()
        limit = top_k or self.top_k
        if not self.chunks or not self._bm25:
            return RetrievalResult(
                query=query,
                chunks=[],
                evidence_found=False,
                method="empty",
                warnings=["No documents are loaded."],
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        queries = [query] + [item for item in (extra_queries or []) if item and item != query]
        lexical = [0.0] * len(self.chunks)
        for candidate in queries:
            scores = self._bm25.scores(_lexical_tokens(candidate))
            lexical = [max(a, b) for a, b in zip(lexical, scores)]

        semantic: Optional[List[float]] = None
        warnings: List[str] = []
        if self._embeddings is not None:
            query_vectors = self.embedder.embed(queries, keep_alive=keep_alive)
            if query_vectors is not None and len(query_vectors) == len(queries):
                similarity = self._embeddings @ query_vectors.T
                semantic = [float(np.max(row)) for row in similarity]
            else:
                error = getattr(self.embedder, "last_error", None)
                warnings.append(
                    "Semantic query embedding failed; falling back to lexical retrieval."
                    + (f" ({error})" if error else "")
                )
                semantic = None

        scored: List[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            lex = lexical[index]
            sem = semantic[index] if semantic is not None else 0.0
            if semantic is not None:
                combined = 0.45 * lex + 0.55 * max(0.0, sem)
            else:
                combined = lex
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(combined, 4),
                    lexical_score=round(lex, 4),
                    semantic_score=round(sem, 4),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        best_lex = max((item.lexical_score for item in scored), default=0.0)
        best_sem = max((item.semantic_score for item in scored), default=0.0)
        if semantic is not None:
            # Semantic retrieval is the primary cross-lingual signal.  A lexical
            # match must cover a substantial share of the query on its own.
            evidence = best_sem >= RETRIEVAL_MIN_SEMANTIC or best_lex >= max(
                RETRIEVAL_MIN_LEXICAL, 0.35
            )
        else:
            evidence = best_lex >= max(RETRIEVAL_MIN_LEXICAL, 0.20)

        method = "hybrid(lexical+bge-m3)" if semantic is not None else "lexical"
        selected = scored[:limit] if evidence else []
        return RetrievalResult(
            query=query,
            chunks=selected,
            evidence_found=evidence,
            method=method,
            warnings=warnings,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            best_lexical=best_lex,
            best_semantic=best_sem,
        )

    def build_citations(
        self,
        retrieved: Sequence[RetrievedChunk],
        cited_indices: Optional[Iterable[int]] = None,
    ) -> List[Citation]:
        cited = set(cited_indices or [])
        citations: List[Citation] = []
        for index, item in enumerate(retrieved, start=1):
            text = item.text.strip().replace("\n", " ")
            excerpt = text[:240] + ("…" if len(text) > 240 else "")
            citations.append(
                Citation(
                    index=index,
                    doc_name=item.doc_name,
                    page=item.page,
                    section=item.section,
                    excerpt=excerpt,
                    chunk_id=item.chunk.chunk_id,
                    cited=(index in cited) if cited else False,
                )
            )
        return citations

    def summary_context(self, max_chars: int = 7000) -> str:
        if not self.chunks:
            return ""
        per_doc: Dict[str, List[Chunk]] = {}
        for chunk in self.chunks:
            per_doc.setdefault(chunk.doc_id, []).append(chunk)
        pieces: List[str] = []
        used = 0
        # Round-robin sample across documents so every upload is represented.
        cursors = {doc_id: 0 for doc_id in per_doc}
        while used < max_chars:
            progressed = False
            for doc_id, chunks in per_doc.items():
                cursor = cursors[doc_id]
                if cursor >= len(chunks):
                    continue
                progressed = True
                chunk = chunks[cursor]
                cursors[doc_id] = cursor + 1
                block = f"[{chunk.doc_name}] {chunk.text}\n"
                if used + len(block) > max_chars:
                    return "\n".join(pieces)
                pieces.append(block.rstrip())
                used += len(block)
            if not progressed:
                break
        return "\n\n".join(pieces)

    def all_chunks_for_doc(self, doc_id: str) -> List[Chunk]:
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self._embeddings = None
        self._bm25 = None
        self.warnings.clear()

    def stats(self) -> Dict[str, Any]:
        languages = sorted({chunk.lang for chunk in self.chunks})
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "languages": languages,
            "semantic_index": self._embeddings is not None,
            "embedding_model": getattr(self.embedder, "model", EMBEDDING_MODEL),
            "embedding_available": bool(self.embedder.available()),
            "warnings": list(dict.fromkeys(self.warnings)),
        }
