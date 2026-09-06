"""Phase 1b D2+D3: score gate signals and pick the best, then report held-out.

D2 (development only): per signal, AUROC separating answerable from negatives,
and the balanced-accuracy threshold via _choose_threshold. Chosen signal =
highest dev AUROC; ties broken by dev balanced accuracy, then by the cheaper
signal (vector-only cosine < hybrid/keyword < cross-encoder).

D3 (held-out, applied once with the frozen dev threshold): per signal, the
false-refusal rate on answerable, the false-answer rate split into negative_ood
and negative_near, and balanced accuracy. Headline: for the current rrf_top gate,
how many answerable held-out questions are refused although a relevant paper sat
at hybrid rank 1 or 2.

Reads docs/evidence/phase-8-gate-signals.json; writes
docs/evidence/phase-8-gate-selection.json (embeds the rows so the verifier is
self-contained).
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from papertrail.evaluation import auroc, _choose_threshold

ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ROOT / "docs" / "evidence" / "phase-8-gate-signals.json"
OUT = ROOT / "docs" / "evidence" / "phase-8-gate-selection.json"
SIGNAL_NAMES = ["rrf_top", "cos_top", "cos_margin", "cos_mean_top3", "kw_top",
                "ce_top", "ce_margin", "ce_sigmoid_top"]
# Lower cost = cheaper to compute; used only as a tie-break.
SIGNAL_COST = {"cos_top": 0, "cos_margin": 0, "cos_mean_top3": 0,
               "kw_top": 1, "rrf_top": 1,
               "ce_top": 2, "ce_margin": 2, "ce_sigmoid_top": 2}


def _dev_scores(rows, signal):
    pos = [r[signal] for r in rows if r["split"] == "development" and r["is_answerable"]]
    neg = [r[signal] for r in rows if r["split"] == "development" and not r["is_answerable"]]
    return pos, neg


def development_report(rows) -> dict:
    out = {}
    for sig in SIGNAL_NAMES:
        pos, neg = _dev_scores(rows, sig)
        chosen = _choose_threshold(pos, neg)
        out[sig] = {
            "auroc": auroc(pos, neg),
            "threshold": chosen["threshold"],
            "dev_accept_rate": chosen["development_positive_accept_rate"],
            "dev_abstain_rate": chosen["development_negative_abstain_rate"],
            "dev_balanced_accuracy": chosen["development_balanced_accuracy"],
        }
    return out


def choose_signal(development: dict) -> str:
    return sorted(
        SIGNAL_NAMES,
        key=lambda s: (-development[s]["auroc"], -development[s]["dev_balanced_accuracy"],
                       SIGNAL_COST[s], s),
    )[0]


def heldout_report(rows, development) -> dict:
    out = {}
    ho = [r for r in rows if r["split"] == "heldout"]
    pos = [r for r in ho if r["is_answerable"]]
    neg = [r for r in ho if not r["is_answerable"]]
    ood = [r for r in neg if r["type"] == "negative_ood"]
    near = [r for r in neg if r["type"] == "negative_near"]
    for sig in SIGNAL_NAMES:
        thr = development[sig]["threshold"]
        accept_pos = statistics.fmean(r[sig] >= thr for r in pos)
        abstain_neg = statistics.fmean(r[sig] < thr for r in neg)
        out[sig] = {
            "false_refusal_rate_answerable": 1.0 - accept_pos,
            "false_answer_rate_ood": statistics.fmean(r[sig] >= thr for r in ood) if ood else 0.0,
            "false_answer_rate_near": statistics.fmean(r[sig] >= thr for r in near) if near else 0.0,
            "balanced_accuracy": (accept_pos + abstain_neg) / 2,
        }
    # Headline: rrf_top refusals of answerable held-out despite a relevant paper at rank 1/2.
    thr = development["rrf_top"]["threshold"]
    refused_rank12 = sum(
        1 for r in pos
        if r["rrf_top"] < thr and r["retrieved_rank_of_relevant"] in (1, 2)
    )
    out["_headline_rrf_top_refused_at_rank_1or2"] = refused_rank12
    return out


def main() -> int:
    signals = json.loads(SIGNALS.read_text())
    rows = signals["rows"]
    development = development_report(rows)
    chosen = choose_signal(development)
    heldout = heldout_report(rows, development)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_set_frozen_at_utc": signals["evaluation_set_frozen_at_utc"],
        "corpus_arxiv_ids_sha256": signals["corpus_arxiv_ids_sha256"],
        "reranker_model": signals["reranker_model"],
        "protocol": {
            "signals": SIGNAL_NAMES,
            "selection_rule": "max dev AUROC; tie-break dev balanced accuracy, then cheaper signal",
            "selected_on": "development only",
        },
        "chosen_signal": chosen,
        "development": development,
        "heldout": heldout,
        "rows": rows,
        "claim_boundary": (
            "Signals selected on development, reported once on held-out. Model-vs-"
            "model retrieval scores, not calibrated probabilities; ce_* are logits, "
            "comparable across signals only via AUROC and error rates, not thresholds."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # compact console table
    print(f"chosen_signal: {chosen}")
    print(f"{'signal':14s} {'devAUROC':>9s} {'devBalAcc':>10s} {'HO_falseRefuse':>15s} {'HO_falseAns_near':>17s} {'HO_balAcc':>10s}")
    for s in SIGNAL_NAMES:
        d = development[s]; h = heldout[s]
        print(f"{s:14s} {d['auroc']:>9.3f} {d['dev_balanced_accuracy']:>10.3f} "
              f"{h['false_refusal_rate_answerable']:>15.3f} {h['false_answer_rate_near']:>17.3f} {h['balanced_accuracy']:>10.3f}")
    print(f"\nHeadline: rrf_top refuses {heldout['_headline_rrf_top_refused_at_rank_1or2']} "
          f"answerable held-out question(s) despite a relevant paper at hybrid rank 1 or 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
