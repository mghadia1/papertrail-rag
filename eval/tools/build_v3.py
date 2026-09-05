"""Assemble the schema-3 v3 question set from authored queries.

Queries were drafted by an LLM (Claude) from eval/tools/sample.md and are meant
for human review before the set is frozen (frozen_at_utc is left null here).

Indices below refer to the 1-based order in sample.json (seed 20260903).
- paraphrase / lexical questions carry one grade-2 relevant paper (the source).
- topical questions carry an empty `relevant` map plus a `seed` (the source
  paper) and are graded later by pooling (eval/tools/pool.py).
Out-of-domain negatives reuse the frozen v1/v2 set; near-miss negatives are real
CS/ML topics verified absent from the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLE = HERE / "sample.json"
POOLS = HERE / "pools.json"
OUT = ROOT / "eval" / "questions-v3.draft.json"
CORPUS_SHA = "7308d240f717df7c9b17a1dfb7140c68615a462298b42866b95bc17f93250146"
SEED = 20260903

# Topical relevance grades, drafted by an LLM (Claude) from the shuffled,
# mode-blind judgment pools (eval/tools/pools.json). Keyed by the question's seed
# paper. grade 2 = primary (squarely on-topic), 1 = partial (adjacent). Any pooled
# id not listed is grade 0. Review before freeze.
GRADES: dict[str, dict[str, int]] = {
    "2608.02123v1": {"2608.02123v1": 2, "2608.02989v1": 2, "2608.01651v1": 2, "2608.03447v1": 2, "2608.00881v1": 1},
    "2608.01475v1": {"2608.01475v1": 2, "2608.01252v1": 2, "2608.00630v1": 2, "2608.01518v1": 2, "2608.01184v1": 2, "2608.01743v1": 2, "2608.03123v1": 1, "2608.01314v1": 1},
    "2608.02830v1": {"2608.02830v1": 2, "2608.01575v1": 1},
    "2608.01288v1": {"2608.01288v1": 2, "2608.01035v1": 1, "2608.01113v1": 1, "2608.03059v1": 1},
    "2608.02820v1": {"2608.02820v1": 2, "2608.00583v1": 2, "2608.03291v1": 2, "2608.02089v1": 2, "2608.03745v1": 1, "2608.01388v1": 1, "2608.00732v1": 1, "2608.02271v1": 1, "2608.01085v1": 1},
    "2608.01334v1": {"2608.01334v1": 2, "2608.01988v1": 2, "2608.03096v1": 2, "2608.00559v1": 2, "2608.02160v1": 2, "2608.03008v1": 2, "2608.00716v1": 2, "2608.01258v1": 2, "2608.01046v1": 1},
    "2608.00573v1": {"2608.00573v1": 2, "2608.01784v1": 2, "2608.01536v1": 1, "2608.03036v1": 1, "2608.01651v1": 1},
    "2608.03887v1": {"2608.03887v1": 2, "2608.01743v1": 2, "2608.01314v1": 1, "2608.03874v1": 1},
    "2608.03562v1": {"2608.03562v1": 2, "2608.02509v1": 2, "2608.03069v1": 1, "2608.01556v1": 1, "2608.01151v1": 1, "2608.02034v1": 1},
    "2608.02603v1": {"2608.02603v1": 2, "2608.01127v2": 1, "2608.03084v1": 1, "2608.02953v1": 1, "2608.03211v1": 1, "2608.00617v1": 1},
    "2608.02206v1": {"2608.02206v1": 2, "2608.01588v1": 2, "2608.02191v1": 2, "2608.02145v1": 2, "2608.02437v2": 2, "2608.01659v1": 1, "2608.00950v1": 1, "2608.01958v1": 1, "2608.01186v1": 1, "2608.01178v1": 1},
    "2608.01321v1": {"2608.01321v1": 2, "2608.01597v1": 2, "2608.03467v1": 2, "2608.04007v1": 2, "2608.01867v2": 2, "2608.01359v1": 1, "2608.02585v1": 1, "2608.01913v1": 1, "2608.01358v1": 1},
    "2608.03218v1": {"2608.03218v1": 2, "2608.03269v1": 2},
    "2608.03817v1": {"2608.03817v1": 2, "2608.01207v1": 2, "2608.01021v1": 1, "2608.02790v1": 1, "2608.03966v1": 1, "2608.03720v1": 1},
    "2608.02688v1": {"2608.02688v1": 2, "2608.01734v1": 2, "2608.03855v1": 1, "2608.03260v1": 1, "2608.01007v1": 1, "2608.00985v1": 1, "2608.02027v1": 1},
    "2608.03664v1": {"2608.03664v1": 2, "2608.03540v1": 2},
    "2608.01370v1": {"2608.01370v1": 2, "2608.01356v1": 1, "2608.03079v1": 1, "2608.03508v1": 1},
    "2608.01743v1": {"2608.01743v1": 2, "2608.03573v1": 1},
}

# index -> (type, split, query). Topical also gets its seed index as relevance
# is decided by pooling.
RETRIEVAL: dict[int, tuple[str, str, str]] = {
    # ---- development paraphrase (24) ----
    1: ("paraphrase", "development", "a parameter-efficient tuning method that stores each specialist as a small learnable code and drives a hypernetwork to emit per-input low-rank weight updates rather than one full stored adapter per specialist"),
    4: ("paraphrase", "development", "hidden malicious behavior in cooperating language models that stays dormant until enough peers accumulate a shared signal, plus a clean-only test-time defense that quarantines anomalous updates before they spread"),
    7: ("paraphrase", "development", "a label-free scheme where a region-aware multimodal model writes a description for an image area, verifies it by re-identifying that area among confusable candidates, and improves from unlabeled data"),
    10: ("paraphrase", "development", "a robot control model that predicts future latent states in the representation used to choose actions, grounded with shape-and-depth cues and supervised by a moving-average teacher, trained with flow matching"),
    12: ("paraphrase", "development", "auditing how skipping blocks in efficient extended-sequence transformers shifts which passages influence the output, using planted correct, wrong, and filler probe cards at several compression ratios"),
    13: ("paraphrase", "development", "why chatbots cave and drop a correct clinical answer under user pushback, and evidence it depends on dialogue conditions like who asks and timing rather than a fixed per-model rate"),
    14: ("paraphrase", "development", "how skewed, noisy, or unevenly sized local data affect which participants a privacy-preserving decentralized training scheme should include, plus a way to score each participant's contribution"),
    15: ("paraphrase", "development", "upscaling low-quality thermal pictures so downstream detectors and scene parsers stay reliable, favoring authentic heat structure over artificial edge crispness"),
    16: ("paraphrase", "development", "letting a coding assistant flag its own reasoning-phase breakpoints while it works, so one long recorded run becomes several supervised targets, including its failed-then-corrected stretches"),
    18: ("paraphrase", "development", "a contrastive pretraining approach for scRNA-seq that splits each profile's genes into two co-expression halves and builds hard negatives by shuffling values to improve whole-profile embeddings"),
    19: ("paraphrase", "development", "deriving per-point thermal-emission factors on mixed historic surfaces by segmenting a co-registered color photo and mapping each substance to a reference table before an inverse-Planck temperature solve, to avoid false subsurface flaws"),
    27: ("paraphrase", "development", "spotting when a chatbot's step-by-step derivation is drifting toward a wrong answer by tracking how it evolves on satisfiability puzzles, then fixing errors with a targeted proof-search prompt"),
    28: ("paraphrase", "development", "shrinking a multi-sensor object tracker by slimming its prediction head and closing the resulting gap with a two-stream teacher-student transfer that separates where-to-track from what-to-track"),
    29: ("paraphrase", "development", "evidence that multimodal models lean on stale wording from an earlier chain of thought instead of re-reading the current picture, plus a training-free firewall that isolates fresh visual computation"),
    31: ("paraphrase", "development", "what pretrained two-dimensional detectors implicitly encode about depth and camera-relative placement of things, recovered from their embeddings with linear and non-linear readouts despite no three-dimensional supervision"),
    32: ("paraphrase", "development", "evidence that apparent spatial blindness in frozen visual encoders stems from the single global embedding readout, and that an attention-pooled query-conditioned readout restores most local feature-association signal"),
    34: ("paraphrase", "development", "separating two failure modes when clips are queried about which event came first — whether the captured frames even contain the events versus whether they are trusted over the user — plus a forward-and-reversed scoring test"),
    37: ("paraphrase", "development", "an onboard learned codec for Earth-observation spacecraft that spends bytes on cloud-free ground rather than clouds without sending a cloud map, plus a deadline-aware scheduler for brief intermittent ground contacts"),
    76: ("paraphrase", "development", "estimating action values from logged interaction data by swapping the symmetric several-back temporal-difference loss for an asymmetric one, which counters the pessimistic bias that grows with longer horizons"),
    77: ("paraphrase", "development", "predicting future scene descriptors for self-driving robots entirely in latent space and reading task outputs straight from those predictions, dropping the heavy module that usually maps states to tasks"),
    80: ("paraphrase", "development", "a training-free way to shrink redundant audio and visual sequences for combined-sensory chat systems, pruning structurally before the network and consolidating semantically inside it to hold accuracy at tiny budgets"),
    81: ("paraphrase", "development", "a review of patient-specific heart-and-vessel computer models that update with clinical measurements, contrasting mechanistic simulations with scalable learned and hybrid graph methods for diagnosis and therapy planning"),
    83: ("paraphrase", "development", "a feed-forward Gaussian scene reconstructor that separates shape modeling from color detail into two branches, letting it work without known camera poses and render crisper novel views"),
    86: ("paraphrase", "development", "toughening learned near-duplicate image matchers so adversarial tweaks cannot slip past, with no model re-fitting: randomized smoothing at match time plus an imperceptible mark added to reference images before release"),
    # ---- held-out paraphrase (12) ----
    39: ("paraphrase", "heldout", "a three-way cancer prognosis model that turns sparse tabular patient records into text embeddings and uses them as an anchor to align pathology and genomic signals through cross-attention and a distribution-matching objective"),
    41: ("paraphrase", "heldout", "a content-policy method for prompt-to-picture diffusion applied at inference that reads the predicted denoised frame to catch banned material and then tweaks a low-rank residual in the conditioning to suppress it, leaving weights unchanged"),
    42: ("paraphrase", "heldout", "a stochastic form of low-dimensional adapter adaptation that samples structured variations along the leading components of shared adapters to give reliable predictive uncertainty while matching the deterministic transform in expectation"),
    43: ("paraphrase", "heldout", "using a vision-language model's grounding to synthesize varied foregrounds and backgrounds and to inject representation noise, improving scarce-label recognition of items when training and evaluation distributions differ sharply"),
    48: ("paraphrase", "heldout", "an interactive guessing game revealing chatbots gather evidence poorly across dialogue rounds for explanation-forming reasoning: many commit before using clues, others exhaust their budget without converging"),
    50: ("paraphrase", "heldout", "making inter-assistant social relations explicit in prompts and finding they mainly push teams toward agreement, helping when consensus is rewarded but not reliably improving accuracy in objective question debates"),
    53: ("paraphrase", "heldout", "quantifying how well annotators agree when they give free-form open-ended labels for life-science passages, comparing embedding, large-language-model, and entailment-based soft reliability scores"),
    54: ("paraphrase", "heldout", "deriving gaze-like attention labels from finished operations by combining deformation-constrained organ tracking with instrument paths, powering an assistive camera that pre-frames relevant regions and eases the operator's mental load"),
    55: ("paraphrase", "heldout", "an analysis of how the training objective for image generators — raw pixels versus autoencoder latents versus self-supervised features — shifts the modeling burden across context inference and per-token denoising"),
    56: ("paraphrase", "heldout", "speeding up decoding of networks that blend full and linear-recurrent layers by rewriting the recurrence into a branch-structured closed form and a GPU kernel that checks all draft nodes at once with far less transient state memory"),
    57: ("paraphrase", "heldout", "a low-latency causal clip reviser that generates chunk by chunk, preserving source fidelity and long-horizon temporal consistency without future frames or a fixed duration, at roughly thirty frames per second"),
    65: ("paraphrase", "heldout", "a terminal benchmark that hides tool meanings so assistants must learn behavior by trial and error, showing they fall back to exhaustive probing under mapping drift despite recorded cues to the following action"),

    # ---- development lexical (16) ----
    6: ("lexical", "development", "LocAnyMed-200K medical visual grounding dataset F1@IoU 0.50"),
    8: ("lexical", "development", "Weixin Pay billion-scale credit fraud detection graph neural network overlapping subgraphs"),
    9: ("lexical", "development", "TIDES longitudinal bilingual English Korean dataset next-speaker prediction AMI Meeting Corpus"),
    11: ("lexical", "development", "Kolmogorov-Arnold Network versus MLP for faster-than-Nyquist BPSK detection bit error rate"),
    17: ("lexical", "development", "Predictive Enhancement Calibration virtual contrast breast MRI FLUX latent flow transformer MAMA100"),
    22: ("lexical", "development", "physics-flavored CNN-Transformer for engineered skeletal muscle contraction Duchenne muscular dystrophy force-time"),
    25: ("lexical", "development", "Fourier motion modeling 4D Gaussian Splatting N3V Google Immersive dynamic novel view"),
    30: ("lexical", "development", "offline top-K logits fused chunked KL loss knowledge distillation 32768 tokens H200 GPU"),
    36: ("lexical", "development", "Swimm3R underwater Beta Splitting medium-aware SfM Barbados dataset WaterSplatting PSNR"),
    38: ("lexical", "development", "SALT subspace-aligned centroid-residual ultra-low-rank LoRA serving vLLM Llama-3.2-3B PCIe"),
    44: ("lexical", "development", "SAKI score-aware low-rank key indexing KV cache LLaMA 3.1 8B Qwen 2.5 7B top-64 recall"),
    46: ("lexical", "development", "WAM-Diff2 autoregressive-to-diffusion distillation autonomous driving VLA FlashInfer CUDA Graphs speedup"),
    52: ("lexical", "development", "GUI-Lens coarse-to-fine cropping GUI grounding OCR UI components GPT-5.5"),
    62: ("lexical", "development", "conditional diffusion synthetic histopathology modified Frechet Inception Distance aggregated Jaccard index nuclei segmentation"),
    70: ("lexical", "development", "paired recipient evaluation deceased donor kidney transplant survival SRTR concordance index"),
    78: ("lexical", "development", "ShielDroid hybrid Android malware detection Random Forest Multilayer Perceptron 97.5% accuracy"),
    # ---- held-out lexical (8) ----
    45: ("lexical", "heldout", "LLM-Guided Retrieval molecular perturbation response Tahoe-100M single-cell atlas unseen cell line"),
    47: ("lexical", "heldout", "standalone DINOv3 DINO.txt training-free open-vocabulary segmentation remote sensing UDD5 DOTA LoveDA"),
    60: ("lexical", "heldout", "MARBERT emoji pragmatics Arabic digital discourse Facebook politeness respect solidarity F1"),
    66: ("lexical", "heldout", "writing-system-level tokenizer adaptation byte-level BPE Ukrainian Nemotron GPT-OSS merge ordering"),
    69: ("lexical", "heldout", "OSSDD OpenSARShip Sentinel-1 ship detection dataset VV VH polarization Faster R-CNN FCOS DETR"),
    75: ("lexical", "heldout", "nGPT normalized Transformer hypersphere Mamba-2 Mixture-of-Experts 14B GatedAdamW training recipe"),
    84: ("lexical", "heldout", "OliveGemma PaliGemma-2-3B LoRA Mediterranean European diet food recognition MedGR ODIN VIPPSTAR"),
    89: ("lexical", "heldout", "condition-number barrier sparse least squares Axiotis Sviridenko Small-Set Expansion Hypothesis Gemini agentic proof"),

    # ---- development topical (12) ----
    2: ("topical", "development", "speculative decoding methods that speed up large language model inference by drafting and verifying tokens"),
    3: ("topical", "development", "preserving plasticity and avoiding catastrophic forgetting in continual or online learning"),
    5: ("topical", "development", "many-shot in-context learning behavior and failure modes in vision-language models"),
    20: ("topical", "development", "one-step or few-step distillation of diffusion models for faster image editing and object removal"),
    21: ("topical", "development", "chain-of-thought monitoring for AI safety and its vulnerability to backdoors or poisoning"),
    23: ("topical", "development", "detecting AI-generated video or images and generalizing to newly emerging generators"),
    26: ("topical", "development", "efficient serving and inference of mixture-of-experts language models with layer skipping or early exit"),
    33: ("topical", "development", "retaining prior capabilities when fine-tuning large language models, measuring and penalizing forgetting"),
    40: ("topical", "development", "robustness and uncertainty in reinforcement learning objectives under misspecified rewards or utilities"),
    64: ("topical", "development", "process-level reward and credit assignment for search-augmented or tool-using reasoning agents"),
    67: ("topical", "development", "dataset distillation that compresses a training set into a small synthetic set for pretrained encoders"),
    82: ("topical", "development", "super-resolution of medical or pathology images that preserves fine cellular structure"),
    # ---- held-out topical (6) ----
    49: ("topical", "heldout", "benchmarks that evaluate controllable video generation models as world models beyond visual quality"),
    51: ("topical", "heldout", "sparse-view or few-shot 3D Gaussian Splatting reconstruction and super-resolution"),
    72: ("topical", "heldout", "black-box detection of hallucination in large vision-language models using consistency signals"),
    79: ("topical", "heldout", "multimodal molecular representation learning linking chemical structure to cellular phenotypes for drug discovery"),
    87: ("topical", "heldout", "combining or fusing multiple pathology foundation models for tile-level representations"),
    88: ("topical", "heldout", "KL regularization to retain base-model capabilities during reinforcement-learning post-training of language models"),
}

# Out-of-domain negatives reused from the frozen v1/v2 set.
NEG_OOD = [
    ("development", "How do I replace a leaking kitchen faucet cartridge?"),
    ("development", "What ingredients make a traditional sourdough starter?"),
    ("development", "Who won the 1974 association football world championship?"),
    ("development", "How should a violin bow be rehaired?"),
    ("development", "Explain the rules for castling in tournament chess."),
    ("development", "What soil mixture is best for growing desert cacti at home?"),
    ("development", "How is a residential property deed transferred?"),
    ("development", "Give a step-by-step recipe for laminated croissant dough."),
    ("development", "Why did the Roman Republic replace its kings?"),
    ("development", "How do I tune the carburetor on a vintage motorcycle?"),
    ("heldout", "What is the safest way to clean a wool overcoat?"),
    ("heldout", "How are points scored in competitive badminton?"),
    ("heldout", "Which herbs should be planted beside tomatoes?"),
    ("heldout", "How do fixed-rate home mortgages calculate monthly payments?"),
    ("heldout", "What caused the eruption of Mount Vesuvius in antiquity?"),
]

# Near-miss negatives: real CS/ML topics verified absent from this corpus.
NEG_NEAR = [
    ("development", "deep learning models for detecting sarcasm in social media text"),
    ("development", "neural networks for fake news detection and misinformation classification"),
    ("development", "transformer models for hate speech detection in online comments"),
    ("development", "crowd counting and density estimation from surveillance images"),
    ("development", "lane detection for autonomous driving perception"),
    ("development", "sound event detection and tagging in audio recordings"),
    ("development", "deep learning for stock price and stock market movement prediction"),
    ("development", "short-term electricity load forecasting with neural networks"),
    ("heldout", "reinforcement learning for adaptive traffic signal control"),
    ("heldout", "vision-based control for autonomous drone racing and quadrotors"),
    ("heldout", "crop yield prediction from satellite remote sensing"),
    ("heldout", "handwriting recognition for historical manuscripts"),
]


def main() -> int:
    sample = json.loads(SAMPLE.read_text())
    ids = sample["arxiv_ids"]  # 1-based indices map to ids[i-1]
    pools = json.loads(POOLS.read_text()) if POOLS.exists() else {}
    pool_by_seed = {info["seed"]: [p["arxiv_id"] for p in info["pool"]] for info in pools.values()}

    retrieval = []
    counts: dict[tuple[str, str], int] = {}
    rq = 0
    for index in sorted(RETRIEVAL):
        qtype, split, query = RETRIEVAL[index]
        rq += 1
        arxiv_id = ids[index - 1]
        item = {
            "id": f"v3q{rq:03d}",
            "split": split,
            "type": qtype,
            "query": query,
        }
        if qtype == "topical":
            item["seed"] = arxiv_id
            grades = GRADES.get(arxiv_id, {})
            item["relevant"] = grades
            pool = pool_by_seed.get(arxiv_id, [])
            # Guarantee every graded id is inside the judged pool.
            item["pool"] = sorted(set(pool) | set(grades))
            missing = sorted(set(grades) - set(pool))
            if missing:
                print(f"WARNING {item['id']} graded ids not in pool: {missing}")
        else:
            item["relevant"] = {arxiv_id: 2}
        retrieval.append(item)
        counts[(split, qtype)] = counts.get((split, qtype), 0) + 1

    abstention = []
    n = 0
    for split, query in NEG_OOD:
        n += 1
        abstention.append({"id": f"v3n{n:03d}", "split": split, "type": "negative_ood", "query": query})
    for split, query in NEG_NEAR:
        n += 1
        abstention.append({"id": f"v3n{n:03d}", "split": split, "type": "negative_near", "query": query})

    payload = {
        "schema_version": 3,
        "frozen_at_utc": None,  # set at freeze time, after review
        "corpus_arxiv_ids_sha256": CORPUS_SHA,
        "sampling_seed": SEED,
        "authorship": "Queries and (pending) topical grades drafted by an LLM (Claude); review before freeze.",
        "retrieval_questions": retrieval,
        "abstention_questions": abstention,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT.relative_to(ROOT)),
        "retrieval": len(retrieval),
        "abstention": len(abstention),
        "counts": {f"{s}/{t}": c for (s, t), c in sorted(counts.items())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
