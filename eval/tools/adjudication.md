# Topical grade adjudication (v3)

Two independent Claude sessions graded the 18 topical pools: the draft grades
(Claude Opus 4.8) and a blind re-grade (Claude Fable 5.1, from the shuffled
mode-blind sheet). Cohen's kappa (3-class) = **0.719** over 690 pooled papers
(**0.709** over the 573 papers in the 15 fully-blind pools; three pools —
v3q025, v3q032, v3q050 — were not fully blind for the second grader). This is a
Claude-vs-Claude second opinion, **not** human inter-annotator agreement.

Every one of the **53 disagreeing papers** was then adjudicated by a further model
pass (Claude Opus 4.8, this session) reading each abstract against the query. No
human has reviewed these decisions yet. Decisions
(d=draft, b=blind, →=final):

## v3q003 — continual/online plasticity & forgetting
- 2608.01314 Remember-R1 d1 b0 → **0**: long-context *visual* forgetting in CoT, not continual learning.
- 2608.01630 RING d0 b1 → **1**: continual large-scale knowledge injection.
- 2608.01672 Learning What to Remember d0 b1 → **1**: online test-time updates preserving future-relevant info.
- 2608.03855 CheMatE d0 b1 → **1**: abstract explicitly addresses catastrophic forgetting in domain adaptation.
- 2608.03874 ContinualSkillBench d0 b1 → **1**: in-context continual skill learning.

## v3q020 — one/few-step diffusion distillation for editing/removal
- 2608.01113 CoT-Edit d1 b0 → **0**: video editing with a CoT planner, no step-distillation.
- 2608.02841 Localize Don't Beautify d0 b1 → **0**: client-side API edit-confinement, not distillation.
- 2608.03974 JoyAI-Video-Edit d0 b1 → **1**: uses distribution-matching distillation for fast editing (video).

## v3q021 — CoT monitoring for safety, backdoors/poisoning
- 2608.02089 Observability Ladder d2 b1 → **2**: core CoT-trace monitoring for judging correctness.
- 2608.03291 Tell-Tale Trace d2 b1 → **2**: CoT-dynamics failure monitoring.
- 2608.03745 Risky Business d1 b2 → **2**: faithfulness-vs-safety tension in CoT monitoring.

## v3q023 — detecting AI-generated media
- 2608.03539 IRIS d0 b1 → **0**: forgery-resistant *watermarking*, not detection.

## v3q032 — retaining capabilities when fine-tuning (non-blind)
- 2608.01314 Remember-R1 d1 b0 → **0**: visual-context forgetting, not fine-tuning retention.
- 2608.03855 CheMatE d0 b1 → **1**: catastrophic forgetting during domain-adaptive pretraining.

## v3q037 — robustness/uncertainty in RL objectives under misspecified rewards
- 2608.01130 Surrogate Updates d0 b1 → **1**: surrogate-loss vs decision-utility mismatch.
- 2608.01151 Stochastic Optimal Control d1 b0 → **0**: chance-constrained control, not reward misspecification.
- 2608.01556 Personalized Reward Modeling d1 b0 → **1**: conflicting/heterogeneous preference (reward) signals.
- 2608.02519 Analytic Planning under Uncertainty d0 b1 → **0**: predictive-state uncertainty, not reward/utility.
- 2608.03875 SAFT VLM reward models d0 b1 → **1**: refines noisy/unreliable reward signals.

## v3q042 — process-level reward/credit for search/tool agents
- 2608.01358 HopRefusalBench d1 b0 → **0**: refusal/abstention benchmark, not credit assignment.
- 2608.01359 EviSD d1 b2 → **2**: per-action credit via evidence-conditioned distillation for search agents.
- 2608.01867 CRISP d2 b1 → **1**: critical-step perception for efficiency, credit-adjacent.
- 2608.01913 Diagnosing Search Behavior d1 b0 → **0**: diagnostic study, not a credit method.
- 2608.02101 Cross-Domain Hybrid OPD d0 b1 → **0**: generalization/alignment-tax, not credit.
- 2608.02585 GradCuit d1 b0 → **0**: credit for latent reasoning, not search/tool agents.
- 2608.03223 Self-Distilled Reward Shaping d0 b2 → **2**: dense per-step credit for agentic RL (draft missed it).
- 2608.03467 Rarity-Aware Credit Redistribution d2 b1 → **1**: GRPO credit, not search-agent-specific.

## v3q050 — medical/pathology super-resolution (non-blind)
- 2608.01823 Detail Continuation SR d0 b1 → **0**: general generative SR, not medical.
- 2608.03106 FaithIR d0 b1 → **0**: infrared SR, not medical.
- 2608.03107 Orthogonal Line-Scanning d0 b1 → **1**: microscopy resolution recovery (cellular imaging).

## v3q054 — graph-structured RAG for multi-document QA (biggest gap)
- 2608.00585 Multi-Hop RAG verification d2 b1 → **1**: multi-hop multi-doc RAG, but not graph-structured.
- 2608.00658 Select-And-Extract d1 b0 → **0**: generic RAG plugin.
- 2608.00712 Multi-Hop Question Generation d1 b0 → **0**: question generation, not RAG QA.
- 2608.00765 RAGOCR d1 b0 → **0**: retrieved-text compression, not graph/multi-doc QA.
- 2608.01311 RH-RAG d1 b0 → **0**: privacy-constrained long-form RAG, not graph.
- 2608.01468 Biomedical QA reranking d2 b0 → **0**: single-source PubMed RAG QA, not graph/multi-doc.
- 2608.01630 RING d1 b0 → **0**: retrieval-internalized generation, not graph RAG QA.
- 2608.02678 DenialRAG d1 b0 → **0**: RAG poisoning attack.
- 2608.03527 Docs Reranker with Search Rubrics d0 b1 → **1**: set-level reranking for complex multi-doc needs.
- 2608.03860 SciRet d1 b0 → **0**: generic RAG pipeline study.

## v3q061 — test-time RL with majority-vote pseudo-labels
- 2608.02585 GradCuit d1 b0 → **0**: test-time latent reasoning, not TTRL with pseudo-labels.
- 2608.02951 SP3O d1 b0 → **0**: preference-based RL, not test-time/majority-vote.
- 2608.03204 Test-Time Aligning LVLMs d0 b1 → **1**: test-time inference-time alignment.

## v3q065 — evidence frame selection for long-video QA
- 2608.01271 Video Token Compression d0 b1 → **1**: token budgeting for video-LLMs.
- 2608.01980 AdaThinkV d1 b0 → **0**: decoding-token efficiency, not frame selection.
- 2608.02078 CAVE d1 b0 → **0**: temporal grounding, not QA frame selection.

## v3q067 — ultra-low-bit KV-cache quantization
- 2608.01078 ScaleQ-1.58 d0 b1 → **0**: weight PTQ, not KV-cache.
- 2608.02901 AnchorKV d1 b2 → **1**: KV compression (anchor-residual), adjacent to KV quantization.
- 2608.03893 Cross-Model KV Transfer d1 b0 → **0**: KV reuse across models, not quantization.

## v3q071 — MLLM referring/reasoning segmentation
- 2608.01473 Slot2Text d0 b1 → **0**: object-centric tokens for surgical VQA, not mask prediction.

## v3q073 — online HD map construction
- 2608.01201 PRISM d1 b0 → **0**: E2E driving motion planning, not map construction.
- 2608.01535 STAR-VLM d1 b0 → **0**: motion/velocity estimation via radar.
- 2608.03084 SUV d1 b0 → **0**: future scene understanding via video generation.

Net: draft was too generous on generic-RAG (v3q054), SLAM/driving (v3q073), and
general-SR (v3q050); the blind grader caught two papers the draft missed
(2608.03855, 2608.03223) and one upgrade (2608.03745). The final labels are the
adjudicated ones; the raw kappa stands on the record.
