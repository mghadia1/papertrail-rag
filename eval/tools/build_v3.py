"""Assemble the schema-3 v3 question set from authored queries.

Queries were drafted by an LLM (Claude); topical relevance grades are drafted by
Claude and are meant to be re-graded blind by a human before freeze. Left
unfrozen here (frozen_at_utc null).

Two disjoint samples:
- dev questions come from sample.json (seed 20260903).
- held-out questions come from sample-heldout.json (seed 20260905), authored
  AFTER the original held-out split was discarded for having been observed in a
  dry-run. No dry-run is run on held-out before freeze.

paraphrase/lexical questions carry one grade-2 relevant paper (the source).
topical questions carry a graded `relevant` map plus the full judged `pool`;
grades come from mode-blind shuffled pools (pool.py).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_SAMPLE = HERE / "sample.json"
HELDOUT_SAMPLE = HERE / "sample-heldout.json"
DEV_POOLS = HERE / "pools.json"
HELDOUT_POOLS = HERE / "pools-heldout.json"
OUT = ROOT / "eval" / "questions-v3.draft.json"
CORPUS_SHA = "7308d240f717df7c9b17a1dfb7140c68615a462298b42866b95bc17f93250146"
DEV_SEED = 20260903
HELDOUT_SEED = 20260905

# ---------------------------------------------------------------------------
# DEV retrieval questions, keyed by 1-based index into sample.json.
# ---------------------------------------------------------------------------
DEV: dict[int, tuple[str, str]] = {
    # paraphrase (24) — no title content word appears in the query
    1: ("paraphrase", "a parameter-efficient tuning method that stores each specialist as a small learnable code and drives a hypernetwork to emit per-input low-rank weight updates rather than one full stored adapter per specialist"),
    4: ("paraphrase", "hidden malicious behavior in cooperating language models that stays dormant until enough peers accumulate a shared signal, plus a clean-only test-time defense that quarantines anomalous updates before they spread"),
    7: ("paraphrase", "a label-free scheme where a region-aware multimodal model writes a description for an image area, verifies it by re-identifying that area among confusable candidates, and improves from unlabeled data"),
    10: ("paraphrase", "a robot control model that predicts future latent states in the representation used to choose actions, grounded with shape-and-depth cues and supervised by a moving-average teacher, trained with flow matching"),
    12: ("paraphrase", "auditing how skipping blocks in efficient extended-sequence transformers shifts which passages influence the output, using planted correct, wrong, and filler probe cards at several compression ratios"),
    13: ("paraphrase", "why chatbots cave and drop a correct clinical answer under user pushback, and evidence it depends on dialogue conditions like who asks and timing rather than a fixed per-model rate"),
    14: ("paraphrase", "how skewed, noisy, or unevenly sized local data affect which participants a privacy-preserving decentralized training scheme should include, plus a way to score each participant's contribution"),
    15: ("paraphrase", "upscaling low-quality thermal pictures so downstream detectors and scene parsers stay reliable, favoring authentic heat structure over artificial edge crispness"),
    16: ("paraphrase", "letting a coding assistant flag its own reasoning-phase breakpoints while it works, so one long recorded run becomes several supervised targets, including its failed-then-corrected stretches"),
    18: ("paraphrase", "a contrastive pretraining approach for scRNA-seq that splits each profile's genes into two co-expression halves and builds hard negatives by shuffling values to improve whole-profile embeddings"),
    19: ("paraphrase", "deriving per-point thermal-emission factors on mixed historic surfaces by segmenting a co-registered color photo and mapping each substance to a reference table before an inverse-Planck temperature solve, to avoid false subsurface flaws"),
    27: ("paraphrase", "spotting when a chatbot's step-by-step derivation is drifting toward a wrong answer by tracking how it evolves on satisfiability puzzles, then fixing errors with a targeted proof-search prompt"),
    28: ("paraphrase", "shrinking a multi-sensor object tracker by slimming its prediction head and closing the resulting gap with a two-stream teacher-student transfer that separates where-to-track from what-to-track"),
    29: ("paraphrase", "evidence that multimodal models lean on stale wording from an earlier chain of thought instead of re-reading the current picture, plus a training-free firewall that isolates fresh visual computation"),
    31: ("paraphrase", "what pretrained two-dimensional detectors implicitly encode about depth and camera-relative placement of things, recovered from their embeddings with linear and non-linear readouts despite no three-dimensional supervision"),
    32: ("paraphrase", "evidence that apparent spatial blindness in frozen visual encoders stems from the single global embedding readout, and that an attention-pooled query-conditioned readout restores most local feature-association signal"),
    34: ("paraphrase", "separating two failure modes when clips are queried about which event came first — whether the captured frames even contain the events versus whether they are trusted over the user — plus a forward-and-reversed scoring test"),
    37: ("paraphrase", "an onboard learned codec for Earth-observation spacecraft that spends bytes on cloud-free ground rather than clouds without sending a cloud map, plus a deadline-aware scheduler for brief intermittent ground contacts"),
    76: ("paraphrase", "estimating action values from logged interaction data by swapping the symmetric several-back temporal-difference loss for an asymmetric one, which counters the pessimistic bias that grows with longer horizons"),
    77: ("paraphrase", "predicting future scene descriptors for self-driving robots entirely in latent space and reading task outputs straight from those predictions, dropping the heavy module that usually maps states to tasks"),
    80: ("paraphrase", "a training-free way to shrink redundant audio and visual sequences for combined-sensory chat systems, pruning structurally before the network and consolidating semantically inside it to hold accuracy at tiny budgets"),
    81: ("paraphrase", "a review of patient-specific heart-and-vessel computer models that update with clinical measurements, contrasting mechanistic simulations with scalable learned and hybrid graph methods for diagnosis and therapy planning"),
    83: ("paraphrase", "a feed-forward Gaussian scene reconstructor that separates shape modeling from color detail into two branches, letting it work without known camera poses and render crisper novel views"),
    86: ("paraphrase", "toughening learned near-duplicate image matchers so adversarial tweaks cannot slip past, with no model re-fitting: randomized smoothing at match time plus an imperceptible mark added to reference images before release"),
    # lexical (16) — natural sentences carrying one or two rare terms
    6: ("lexical", "medical visual grounding trained on the LocAnyMed-200K dataset"),
    8: ("lexical", "billion-scale credit fraud detection on Weixin Pay using graph neural networks"),
    9: ("lexical", "next-speaker prediction on the TIDES bilingual multi-party conversation dataset"),
    11: ("lexical", "Kolmogorov-Arnold Networks for faster-than-Nyquist signaling detection"),
    17: ("lexical", "virtual contrast enhancement for breast MRI using a FLUX latent flow transformer"),
    22: ("lexical", "a physics-flavored network parametrizing contraction dynamics of engineered skeletal muscle tissues"),
    25: ("lexical", "Fourier motion modeling for dynamic 4D Gaussian Splatting on the N3V benchmark"),
    30: ("lexical", "efficient LLM distillation with offline top-K logits and a fused chunked KL loss"),
    36: ("lexical", "underwater 3D reconstruction results on the Barbados dataset compared with WaterSplatting"),
    38: ("lexical", "ultra-low-rank LoRA serving with subspace-aligned centroid-residual training in vLLM"),
    44: ("lexical", "score-aware low-rank key indexing for long-context KV cache compression (SAKI)"),
    46: ("lexical", "autoregressive-to-diffusion distillation for an efficient autonomous-driving VLA"),
    52: ("lexical", "coarse-to-fine cropping for GUI grounding with general-purpose VLMs"),
    62: ("lexical", "evaluating conditional diffusion for synthetic histopathology with pathology-specific FID"),
    70: ("lexical", "paired-recipient evaluation of survival prediction for deceased-donor kidney transplants using SRTR data"),
    78: ("lexical", "hybrid Android malware detection combining Random Forest and a multilayer perceptron"),
    # topical (12) — relevance decided by pooling
    2: ("topical", "speculative decoding methods that speed up large language model inference by drafting and verifying tokens"),
    3: ("topical", "preserving plasticity and avoiding catastrophic forgetting in continual or online learning"),
    5: ("topical", "many-shot in-context learning behavior and failure modes in vision-language models"),
    20: ("topical", "one-step or few-step distillation of diffusion models for faster image editing and object removal"),
    21: ("topical", "chain-of-thought monitoring for AI safety and its vulnerability to backdoors or poisoning"),
    23: ("topical", "detecting AI-generated video or images and generalizing to newly emerging generators"),
    26: ("topical", "efficient serving and inference of mixture-of-experts language models with layer skipping or early exit"),
    33: ("topical", "retaining prior capabilities when fine-tuning large language models, measuring and penalizing forgetting"),
    40: ("topical", "robustness and uncertainty in reinforcement learning objectives under misspecified rewards or utilities"),
    64: ("topical", "process-level reward and credit assignment for search-augmented or tool-using reasoning agents"),
    67: ("topical", "dataset distillation that compresses a training set into a small synthetic set for pretrained encoders"),
    82: ("topical", "super-resolution of medical or pathology images that preserves fine cellular structure"),
}

# ---------------------------------------------------------------------------
# HELD-OUT retrieval questions, keyed by 1-based index into sample-heldout.json.
# Authored fresh after the original held-out split was discarded. NOT dry-run.
# ---------------------------------------------------------------------------
HELDOUT: dict[int, tuple[str, str]] = {
    # paraphrase (12)
    3: ("paraphrase", "a message-passing operator for networks whose edges carry both a sign and an orientation, moving information only where node-potential gaps agree with the arrow, improving node classification and link prediction"),
    4: ("paraphrase", "testing the belief that shorter prompts help when the needed information is kept, by either trimming the middle or dropping only the irrelevant parts, and showing naive middle-removal merely measures how often it spares the answer"),
    5: ("paraphrase", "a formal account of when a metric reads healthy while the model is actually broken, spanning gamed reward models at fit time and unmonitored production faults, with a taxonomy validated on real incidents"),
    7: ("paraphrase", "a training-free method and dataset for judging which of several same-category objects sits nearest a reference item in one photo, estimating floor-plane distances with uncertainty-aware prompting"),
    8: ("paraphrase", "checking whether medical claims are true, false, or misleading by pulling trusted passages from WHO and a national disease-control agency and classifying with a fine-tuned encoder, tested on Nigerian fact-checks"),
    12: ("paraphrase", "scaling specific feed-forward units in a chatbot to reproduce dementia-like speech changes such as reduced idea density and worse recall, showing units found from clinical transcripts causally shape behavior"),
    14: ("paraphrase", "synthesizing audio-driven expressive portrait video that mixes implicit semantic features with explicit blendshape priors to give precise, continuous control over expression intensity without losing texture detail"),
    22: ("paraphrase", "a training-free fix for text-to-image models that merge or drop objects, correcting the initial attention allocation once so overlapping subjects separate, rather than steering the whole sampling path"),
    29: ("paraphrase", "learning end-to-end which higher-order structures like cycles and cliques to add to a graph model, instead of fixing them beforehand with an unsupervised rule, improving node and graph classification"),
    33: ("paraphrase", "predicting sustained high-water plateaus for early warning by combining readings from many monitoring gages with bounded corrections that keep each site's local time-series forecast as a stable anchor"),
    37: ("paraphrase", "correcting coarse- and fine-mesh numerical solutions with a small learned corrector to price multi-asset financial derivatives faster, demonstrated on Black-Scholes and Heston barrier contracts with little high-fidelity data"),
    38: ("paraphrase", "estimating how many overlapping decaying sinusoids fill a plate-reverb impulse response by predicting mode tallies in several spectral ranges with a tree regressor, then fitting decay and gain on grids via an all-pole model"),
    # lexical (8)
    1: ("lexical", "near-real-time object removal attacks on video perception evaluated on the South Carolina Connected Vehicle Testbed"),
    9: ("lexical", "monocular panoramic SLAM using a frozen geometry foundation model for loop closure (HALO-SLAM)"),
    13: ("lexical", "zero-shot humanoid motion tracking on the Unitree G1 with robot-native motion generation"),
    18: ("lexical", "the Observatorio Lazaro database tracking anglicism usage in the Spanish press"),
    21: ("lexical", "estimating plate-reverb parameters for Task A of the DAFx Parameter Estimation Challenge"),
    27: ("lexical", "neuro-symbolic biomedical relation extraction evaluated on DDI and ChemProt (ANCHOR-RE)"),
    35: ("lexical", "physics-guided deep learning for patient-specific microwave ablation planning of liver tumors"),
    36: ("lexical", "a machine-readable catalogue and handwriting-recognition accuracy measure for the Tsiolkovsky archive, fond 555"),
    # topical (6)
    2: ("topical", "graph-structured retrieval-augmented generation for complex multi-document question answering"),
    10: ("topical", "test-time reinforcement learning for LLM reasoning using majority-vote pseudo-labels"),
    15: ("topical", "evidence frame selection for long-video question answering under a fixed token budget"),
    19: ("topical", "ultra-low-bit KV-cache quantization for long-context LLM inference"),
    28: ("topical", "MLLM-based referring and reasoning image segmentation with efficient mask prediction"),
    31: ("topical", "online high-definition map construction for autonomous driving from multi-modal inputs"),
}

# Topical grades keyed by the question's seed paper. grade 2 = primary, 1 = partial.
# Any pooled id not listed is grade 0. DEV grades are Claude-drafted (to be
# re-graded blind by a human before freeze); HELD-OUT grades are added after
# pooling the fresh held-out topical questions.
# Adjudicated final grades. Every paper where the draft (Claude Opus 4.8) and the
# blind re-grade (Claude Fable 5.1) disagreed was decided by hand against the
# abstract (see eval/tools/adjudication.md). Kappa between the two gradings = 0.719
# (0.709 over the fully-blind pools). This is a Claude-vs-Claude second opinion.
GRADES: dict[str, dict[str, int]] = {
    # --- dev ---
    "2608.02123v1": {"2608.02123v1": 2, "2608.02989v1": 2, "2608.01651v1": 2, "2608.03447v1": 2, "2608.00881v1": 1},
    "2608.01475v1": {"2608.01475v1": 2, "2608.01252v1": 2, "2608.00630v1": 2, "2608.01518v1": 2, "2608.01184v1": 2, "2608.01743v1": 2, "2608.03123v1": 1, "2608.01630v1": 1, "2608.01672v1": 1, "2608.03855v1": 1, "2608.03874v1": 1},
    "2608.02830v1": {"2608.02830v1": 2, "2608.01575v1": 1},
    "2608.01288v1": {"2608.01288v1": 2, "2608.01035v1": 1, "2608.03059v1": 1, "2608.03974v1": 1},
    "2608.02820v1": {"2608.02820v1": 2, "2608.00583v1": 2, "2608.03291v1": 2, "2608.02089v1": 2, "2608.03745v1": 2, "2608.01388v1": 1, "2608.00732v1": 1, "2608.02271v1": 1, "2608.01085v1": 1},
    "2608.01334v1": {"2608.01334v1": 2, "2608.01988v1": 2, "2608.03096v1": 2, "2608.00559v1": 2, "2608.02160v1": 2, "2608.03008v1": 2, "2608.00716v1": 2, "2608.01258v1": 2, "2608.01046v1": 1},
    "2608.00573v1": {"2608.00573v1": 2, "2608.01784v1": 2, "2608.01536v1": 1, "2608.03036v1": 1, "2608.01651v1": 1},
    "2608.03887v1": {"2608.03887v1": 2, "2608.01743v1": 2, "2608.03874v1": 1, "2608.03855v1": 1},
    "2608.03562v1": {"2608.03562v1": 2, "2608.02509v1": 2, "2608.03069v1": 1, "2608.01556v1": 1, "2608.02034v1": 1, "2608.01130v1": 1, "2608.03875v1": 1},
    "2608.01321v1": {"2608.01321v1": 2, "2608.01597v1": 2, "2608.04007v1": 2, "2608.01359v1": 2, "2608.03223v1": 2, "2608.03467v1": 1, "2608.01867v2": 1},
    "2608.03218v1": {"2608.03218v1": 2, "2608.03269v1": 2},
    "2608.03664v1": {"2608.03664v1": 2, "2608.03540v1": 2, "2608.03107v1": 1},
    # --- held-out ---
    "2608.01565v1": {"2608.01565v1": 2, "2608.01269v2": 2, "2608.00585v1": 1, "2608.03527v1": 1},
    "2608.03545v1": {"2608.03545v1": 2, "2608.04001v1": 1, "2608.01014v1": 1, "2608.03204v1": 1},
    "2608.01660v1": {"2608.01660v1": 2, "2608.03918v1": 2, "2608.00714v1": 2, "2608.03083v1": 1, "2608.03112v1": 1, "2608.01169v1": 1, "2608.01271v1": 1},
    "2608.02691v1": {"2608.02691v1": 2, "2608.00528v1": 1, "2608.02901v1": 1, "2608.01247v1": 1, "2608.00902v1": 1, "2608.01631v1": 1},
    "2608.02791v1": {"2608.02791v1": 2, "2608.03147v1": 2, "2608.01354v1": 2, "2608.01663v1": 1, "2608.02284v1": 1, "2608.03911v1": 1, "2608.02470v1": 1},
    "2608.01338v1": {"2608.01338v1": 2, "2608.02449v1": 1},
}

# Recorded in the evidence file for transparency about how the topical labels were
# produced (item 4 of the review).
TOPICAL_GRADE_PROVENANCE = {
    "method": "two independent Claude gradings, then hand adjudication of every disagreement",
    "draft_grader": "Claude Opus 4.8",
    "blind_grader": "Claude Fable 5.1 (mode-blind shuffled sheet)",
    "adjudicator": "Claude Opus 4.8 (per-abstract, all 53 disagreements)",
    "cohen_kappa_3class_all_pools": 0.7186,
    "cohen_kappa_3class_blind_pools_only": 0.7089,
    "exact_label_agreement": 0.9232,
    "disagreeing_papers": 53,
    "non_blind_pools": ["v3q025", "v3q032", "v3q050"],
    "caveat": "Claude-vs-Claude second opinion, not human inter-annotator agreement.",
    "record": "eval/tools/adjudication.md",
}


def _neg(split: str, qtype: str, queries: list[str]) -> list[tuple[str, str, str]]:
    return [(split, qtype, q) for q in queries]


# Out-of-domain negatives. Dev reuses the frozen v1/v2 set; held-out uses fresh
# everyday questions (the previous held-out negatives were observed in a dry-run).
ABSTENTION = (
    _neg("development", "negative_ood", [
        "How do I replace a leaking kitchen faucet cartridge?",
        "What ingredients make a traditional sourdough starter?",
        "Who won the 1974 association football world championship?",
        "How should a violin bow be rehaired?",
        "Explain the rules for castling in tournament chess.",
        "What soil mixture is best for growing desert cacti at home?",
        "How is a residential property deed transferred?",
        "Give a step-by-step recipe for laminated croissant dough.",
        "Why did the Roman Republic replace its kings?",
        "How do I tune the carburetor on a vintage motorcycle?",
    ])
    + _neg("development", "negative_near", [
        "deep learning models for detecting sarcasm in social media text",
        "neural networks for fake news detection and misinformation classification",
        "transformer models for hate speech detection in online comments",
        "crowd counting and density estimation from surveillance images",
        "lane detection for autonomous driving perception",
        "sound event detection and tagging in audio recordings",
        "deep learning for stock price and stock market movement prediction",
        "short-term electricity load forecasting with neural networks",
    ])
    + _neg("heldout", "negative_ood", [
        "How do I prune an overgrown apple tree in late winter?",
        "What is the proper way to season a new cast-iron skillet?",
        "What are the regulation dimensions of a singles tennis court?",
        "How is a traditional garam masala spice blend made?",
        "What knots secure a canoe to a car roof rack?",
    ])
    + _neg("heldout", "negative_near", [
        "protein structure prediction with deep learning",
        "homomorphic encryption for privacy-preserving neural network inference",
        "gravitational-wave signal detection with neural networks",
        "exoplanet detection from transit light curves with machine learning",
    ])
)


def _pool_by_seed() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in (DEV_POOLS, HELDOUT_POOLS):
        if path.exists():
            for info in json.loads(path.read_text()).values():
                result[info["seed"]] = [p["arxiv_id"] for p in info["pool"]]
    return result


def main() -> int:
    dev_ids = json.loads(DEV_SAMPLE.read_text())["arxiv_ids"]
    heldout_ids = json.loads(HELDOUT_SAMPLE.read_text())["arxiv_ids"] if HELDOUT_SAMPLE.exists() else []
    pool_by_seed = _pool_by_seed()

    retrieval = []
    counts: dict[tuple[str, str], int] = {}
    rq = 0
    for split, table, ids in (("development", DEV, dev_ids), ("heldout", HELDOUT, heldout_ids)):
        for index in sorted(table):
            qtype, query = table[index]
            rq += 1
            arxiv_id = ids[index - 1]
            item = {"id": f"v3q{rq:03d}", "split": split, "type": qtype, "query": query}
            if qtype == "topical":
                item["seed"] = arxiv_id
                grades = GRADES.get(arxiv_id, {})
                item["relevant"] = grades
                pool = pool_by_seed.get(arxiv_id, [])
                item["pool"] = sorted(set(pool) | set(grades))
                missing = sorted(set(grades) - set(pool))
                if missing:
                    print(f"WARNING {item['id']} graded ids not in pool: {missing}")
            else:
                item["relevant"] = {arxiv_id: 2}
            retrieval.append(item)
            counts[(split, qtype)] = counts.get((split, qtype), 0) + 1

    abstention = []
    for n, (split, qtype, query) in enumerate(ABSTENTION, start=1):
        abstention.append({"id": f"v3n{n:03d}", "split": split, "type": qtype, "query": query})

    payload = {
        "schema_version": 3,
        "frozen_at_utc": None,
        "corpus_arxiv_ids_sha256": CORPUS_SHA,
        "sampling_seed": DEV_SEED,
        "heldout_sampling_seed": HELDOUT_SEED,
        "authorship": ("Queries LLM-authored (Claude). Topical grades were produced by two "
                       "independent Claude gradings with every disagreement hand-adjudicated "
                       "(see topical_grade_provenance). Held-out authored from a disjoint fresh "
                       "sample and never dry-run."),
        "topical_grade_provenance": TOPICAL_GRADE_PROVENANCE,
        "retrieval_questions": retrieval,
        "abstention_questions": abstention,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "retrieval": len(retrieval),
                      "abstention": len(abstention),
                      "counts": {f"{s}/{t}": c for (s, t), c in sorted(counts.items())}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
