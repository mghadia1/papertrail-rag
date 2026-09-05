"""Enforce the paraphrase rule: no content word from a paper's title may appear
in its paraphrase query.

Stopwords and tokens shorter than 3 characters are ignored (the rule targets
content words; titles unavoidably share articles and prepositions with queries).
Exits non-zero and prints every violation so it can gate the freeze.

    python eval/tools/check_no_title_words.py eval/questions-v3.draft.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from sqlalchemy import select

from papertrail.database import session_scope
from papertrail.models import Paper

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "via",
    "from", "into", "by", "at", "as", "is", "are", "be", "that", "this", "it",
    "its", "how", "what", "why", "when", "which", "who", "than", "then", "over",
    "under", "using", "use", "used", "can", "not", "no", "but", "we", "our",
    "their", "they", "them", "each", "one", "two", "three", "more", "most",
    "less", "up", "out", "off", "per", "so", "if", "does", "do", "about",
}

WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) >= 3 and w not in STOPWORDS}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/questions-v3.draft.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    paraphrase = [
        q for q in payload["retrieval_questions"] if q["type"] == "paraphrase"
    ]
    wanted = {aid for q in paraphrase for aid in q["relevant"]}
    with session_scope() as session:
        rows = session.execute(
            select(Paper.arxiv_id, Paper.title).where(Paper.arxiv_id.in_(wanted))
        ).all()
    titles = {row.arxiv_id: row.title for row in rows}

    violations = 0
    for q in paraphrase:
        (arxiv_id,) = q["relevant"].keys()
        title_words = content_words(titles[arxiv_id])
        query_words = content_words(q["query"])
        overlap = sorted(title_words & query_words)
        if overlap:
            violations += 1
            print(f"{q['id']} ({arxiv_id}) shares title words: {overlap}")
            print(f"    title: {titles[arxiv_id]}")
            print(f"    query: {q['query']}")
    if violations:
        print(f"\n{violations} paraphrase question(s) violate the no-title-word rule")
        return 1
    print(f"OK: {len(paraphrase)} paraphrase questions share no title content words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
