"""Statement-level Natural Language Inference (NLI) entailment and hallucination verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class EntailmentVerdict(str, Enum):
    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StatementVerification:
    statement: str
    cited_arxiv_id: str | None
    verdict: EntailmentVerdict
    confidence: float
    best_matching_premise: str
    reason: str


@dataclass(frozen=True)
class EntailmentReport:
    is_faithful: bool
    faithfulness_score: float
    total_statements: int
    entailed_statements: int
    contradicted_statements: int
    unsupported_statements: int
    statements: tuple[StatementVerification, ...]
    ungrounded_claims: tuple[str, ...]


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
CITATION_REGEX = re.compile(r"\[([A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+)?v\d+)\]")


def extract_statements(answer: str) -> list[tuple[str, str | None]]:
    """Split an answer into individual statements and extract their attached citation ID."""
    paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]
    statements: list[tuple[str, str | None]] = []

    for para in paragraphs:
        raw_sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(para) if s.strip()]
        for sentence in raw_sentences:
            citations = CITATION_REGEX.findall(sentence)
            cited_id = citations[0] if citations else None
            # Clean citations from the statement text for clean matching
            cleaned_text = CITATION_REGEX.sub("", sentence).strip()
            if cleaned_text:
                statements.append((cleaned_text, cited_id))

    return statements


class NLIJudge(Protocol):
    def verify_statement(
        self, statement: str, premise_text: str, cited_id: str | None
    ) -> StatementVerification:
        """Verify if statement is entailed by the premise text."""


class HeuristicNLIJudge:
    """Deterministic, zero-dependency token-overlap and negation-aware NLI judge."""

    NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "fails", "failed", "cannot", "unable", "without"}

    def _split_into_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]

    def verify_statement(
        self, statement: str, premise_text: str, cited_id: str | None
    ) -> StatementVerification:
        if not premise_text.strip():
            return StatementVerification(
                statement=statement,
                cited_arxiv_id=cited_id,
                verdict=EntailmentVerdict.UNSUPPORTED,
                confidence=1.0,
                best_matching_premise="",
                reason="No premise context available for citation.",
            )

        statement_words = set(re.findall(r"\w+", statement.lower()))
        informative_words = {w for w in statement_words if len(w) > 3}
        if not informative_words:
            return StatementVerification(
                statement=statement,
                cited_arxiv_id=cited_id,
                verdict=EntailmentVerdict.ENTAILED,
                confidence=0.8,
                best_matching_premise=premise_text[:100],
                reason="Statement contains no specific factual claims.",
            )

        premises = self._split_into_sentences(premise_text)
        best_overlap = 0.0
        best_sentence = ""

        statement_has_neg = bool(statement_words & self.NEGATION_WORDS)

        for premise in premises:
            premise_words = set(re.findall(r"\w+", premise.lower()))
            informative_premise = {w for w in premise_words if len(w) > 3}
            if not informative_premise:
                continue

            intersection = informative_words & informative_premise
            overlap = len(intersection) / len(informative_words)

            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = premise

        # Check for direct contradictions (polarity flip on high overlap)
        if best_overlap >= 0.60:
            best_words = set(re.findall(r"\w+", best_sentence.lower()))
            premise_has_neg = bool(best_words & self.NEGATION_WORDS)
            if statement_has_neg != premise_has_neg:
                return StatementVerification(
                    statement=statement,
                    cited_arxiv_id=cited_id,
                    verdict=EntailmentVerdict.CONTRADICTED,
                    confidence=best_overlap,
                    best_matching_premise=best_sentence,
                    reason="Contradiction detected: polarity mismatch on high token overlap.",
                )
            return StatementVerification(
                statement=statement,
                cited_arxiv_id=cited_id,
                verdict=EntailmentVerdict.ENTAILED,
                confidence=best_overlap,
                best_matching_premise=best_sentence,
                reason="Statement is directly entailed by the source premise.",
            )

        if best_overlap >= 0.35:
            return StatementVerification(
                statement=statement,
                cited_arxiv_id=cited_id,
                verdict=EntailmentVerdict.NEUTRAL,
                confidence=best_overlap,
                best_matching_premise=best_sentence,
                reason="Statement is partially mentioned but not fully substantiated.",
            )

        return StatementVerification(
            statement=statement,
            cited_arxiv_id=cited_id,
            verdict=EntailmentVerdict.UNSUPPORTED,
            confidence=1.0 - best_overlap,
            best_matching_premise=best_sentence,
            reason="Statement contains ungrounded claims absent from the source text.",
        )


def evaluate_entailment(
    answer: str,
    context_by_id: dict[str, str],
    *,
    judge: NLIJudge | None = None,
    min_faithfulness_threshold: float = 0.80,
) -> EntailmentReport:
    """Evaluate an entire answer against the retrieved chunk context."""
    if judge is None:
        judge = HeuristicNLIJudge()

    statements = extract_statements(answer)
    if not statements:
        return EntailmentReport(
            is_faithful=True,
            faithfulness_score=1.0,
            total_statements=0,
            entailed_statements=0,
            contradicted_statements=0,
            unsupported_statements=0,
            statements=(),
            ungrounded_claims=(),
        )

    verifications: list[StatementVerification] = []
    ungrounded: list[str] = []
    entailed_count = 0
    contradicted_count = 0
    unsupported_count = 0

    for stmt, cited_id in statements:
        premise = ""
        if cited_id and cited_id in context_by_id:
            premise = context_by_id[cited_id]
        elif context_by_id:
            # Aggregate all available premises if citation is missing or generalized
            premise = " ".join(context_by_id.values())

        ver = judge.verify_statement(stmt, premise, cited_id)
        verifications.append(ver)

        if ver.verdict == EntailmentVerdict.ENTAILED:
            entailed_count += 1
        elif ver.verdict == EntailmentVerdict.CONTRADICTED:
            contradicted_count += 1
            ungrounded.append(f"[Contradiction] {stmt}")
        elif ver.verdict == EntailmentVerdict.UNSUPPORTED:
            unsupported_count += 1
            ungrounded.append(f"[Unsupported] {stmt}")
        else:
            ungrounded.append(f"[Neutral/Unverified] {stmt}")

    faithfulness_score = entailed_count / len(statements)
    is_faithful = (
        faithfulness_score >= min_faithfulness_threshold and contradicted_count == 0
    )

    return EntailmentReport(
        is_faithful=is_faithful,
        faithfulness_score=faithfulness_score,
        total_statements=len(statements),
        entailed_statements=entailed_count,
        contradicted_statements=contradicted_count,
        unsupported_statements=unsupported_count,
        statements=tuple(verifications),
        ungrounded_claims=tuple(ungrounded),
    )
