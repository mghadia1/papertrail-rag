# v3 question set — review draft (NOT frozen)

Queries and (pending) topical grades drafted by an LLM (Claude); review before freeze.

Sampling seed 20260903. 78 retrieval questions, 27 abstention questions.


## development · paraphrase (24)

**v3q001** — a parameter-efficient tuning method that stores each specialist as a small learnable code and drives a hypernetwork to emit per-input low-rank weight updates rather than one full stored adapter per specialist
  - answer: `2608.03275v1` — MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation

**v3q004** — hidden malicious behavior in cooperating language models that stays dormant until enough peers accumulate a shared signal, plus a clean-only test-time defense that quarantines anomalous updates before they spread
  - answer: `2608.01085v1` — When Collaboration Becomes a Trigger: Collective Evidence-Threshold Backdoors in Multi-Agent Systems

**v3q007** — a label-free scheme where a region-aware multimodal model writes a description for an image area, verifies it by re-identifying that area among confusable candidates, and improves from unlabeled data
  - answer: `2608.01354v1` — PixVL: Self-Supervised Training of Pixel-Level MLLMs via a Unified Mask--Text Consistency Cycle

**v3q010** — a robot control model that predicts future latent states in the representation used to choose actions, grounded with shape-and-depth cues and supervised by a moving-average teacher, trained with flow matching
  - answer: `2608.01397v1` — SG-WAM: Self-Guided World Modeling in Geometry-Aware Policy Space

**v3q012** — auditing how skipping blocks in efficient extended-sequence transformers shifts which passages influence the output, using planted correct, wrong, and filler probe cards at several compression ratios
  - answer: `2608.01676v1` — Understanding Sparse Attention Selectivity in Long-Context Foundation Models via Counterfactual Evaluation

**v3q013** — why chatbots cave and drop a correct clinical answer under user pushback, and evidence it depends on dialogue conditions like who asks and timing rather than a fixed per-model rate
  - answer: `2608.01017v1` — Why LLMs Give In: Conversational Factors and Reasoning Behind Medical Sycophancy

**v3q014** — how skewed, noisy, or unevenly sized local data affect which participants a privacy-preserving decentralized training scheme should include, plus a way to score each participant's contribution
  - answer: `2608.02250v1` — Assessing the Impacts of Imperfect Datasets on Client Selections in Federated Learning

**v3q015** — upscaling low-quality thermal pictures so downstream detectors and scene parsers stay reliable, favoring authentic heat structure over artificial edge crispness
  - answer: `2608.03106v1` — FaithIR: Rethinking Infrared Image Super-Resolution from Perceptual Sharpness to Task Relevant Fidelity

**v3q016** — letting a coding assistant flag its own reasoning-phase breakpoints while it works, so one long recorded run becomes several supervised targets, including its failed-then-corrected stretches
  - answer: `2608.02302v1` — Trajectories That Segment Themselves: Agent-Declared Boundaries as a Training Unit

**v3q018** — a contrastive pretraining approach for scRNA-seq that splits each profile's genes into two co-expression halves and builds hard negatives by shuffling values to improve whole-profile embeddings
  - answer: `2608.00985v1` — Beyond Gene Reconstruction: Learning Cell Representations through Complementary Transcriptomic Views

**v3q019** — deriving per-point thermal-emission factors on mixed historic surfaces by segmenting a co-registered color photo and mapping each substance to a reference table before an inverse-Planck temperature solve, to avoid false subsurface flaws
  - answer: `2608.02964v1` — Material-Segmented Per-Pixel Emissivity Correction for Thermographic Anomaly Detection in Cultural Heritage Digital Twins

**v3q026** — spotting when a chatbot's step-by-step derivation is drifting toward a wrong answer by tracking how it evolves on satisfiability puzzles, then fixing errors with a targeted proof-search prompt
  - answer: `2608.03291v1` — The Tell-Tale Trace: Detecting Reasoning Failures in LLMs Using Chain-of-Thought Dynamics

**v3q027** — shrinking a multi-sensor object tracker by slimming its prediction head and closing the resulting gap with a two-stream teacher-student transfer that separates where-to-track from what-to-track
  - answer: `2608.01488v1` — Towards Compact Unified Multimodal Tracking: Synergizing Knowledge Distillation with Structural Pruning

**v3q028** — evidence that multimodal models lean on stale wording from an earlier chain of thought instead of re-reading the current picture, plus a training-free firewall that isolates fresh visual computation
  - answer: `2608.01930v1` — Recompute or Reuse? Diagnosing and Mitigating Textual Shortcuts in VLM Self-Reflection

**v3q030** — what pretrained two-dimensional detectors implicitly encode about depth and camera-relative placement of things, recovered from their embeddings with linear and non-linear readouts despite no three-dimensional supervision
  - answer: `2608.01495v1` — Probing the 3D Object-Level Understanding of Pre-Trained Detection Transformers

**v3q031** — evidence that apparent spatial blindness in frozen visual encoders stems from the single global embedding readout, and that an attention-pooled query-conditioned readout restores most local feature-association signal
  - answer: `2608.00726v1` — Foveated Probes Recover Localized Binding Information in Vision Foundation Models

**v3q033** — separating two failure modes when clips are queried about which event came first — whether the captured frames even contain the events versus whether they are trusted over the user — plus a forward-and-reversed scoring test
  - answer: `2608.03160v1` — Caved or Convinced: Temporal Sampling Gates Claim Deference in Video Large Language Models

**v3q035** — an onboard learned codec for Earth-observation spacecraft that spends bytes on cloud-free ground rather than clouds without sending a cloud map, plus a deadline-aware scheduler for brief intermittent ground contacts
  - answer: `2608.01457v1` — Clear-Weighted Bit Allocation for Satellite Downlinks

**v3q066** — estimating action values from logged interaction data by swapping the symmetric several-back temporal-difference loss for an asymmetric one, which counters the pessimistic bias that grows with longer horizons
  - answer: `2608.02034v1` — Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning

**v3q067** — predicting future scene descriptors for self-driving robots entirely in latent space and reading task outputs straight from those predictions, dropping the heavy module that usually maps states to tasks
  - answer: `2608.02428v1` — DF$^3$: World Modeling via Decoder-Free Feature Forecasting in Autonomous Navigation

**v3q070** — a training-free way to shrink redundant audio and visual sequences for combined-sensory chat systems, pruning structurally before the network and consolidating semantically inside it to hold accuracy at tiny budgets
  - answer: `2608.03812v1` — OmniPack: Unified Token Compression for Efficient Omni-modal Large Language Models

**v3q071** — a review of patient-specific heart-and-vessel computer models that update with clinical measurements, contrasting mechanistic simulations with scalable learned and hybrid graph methods for diagnosis and therapy planning
  - answer: `2608.02135v1` — Cardiovascular Digital Twins from Physics Based to Data Driven Approaches

**v3q073** — a feed-forward Gaussian scene reconstructor that separates shape modeling from color detail into two branches, letting it work without known camera poses and render crisper novel views
  - answer: `2608.01186v1` — QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction

**v3q075** — toughening learned near-duplicate image matchers so adversarial tweaks cannot slip past, with no model re-fitting: randomized smoothing at match time plus an imperceptible mark added to reference images before release
  - answer: `2608.03101v1` — Double Down on Defense: Strengthening Deep Perceptual Hashes against Evasion Attacks without Retraining


## development · lexical (16)

**v3q006** — LocAnyMed-200K medical visual grounding dataset F1@IoU 0.50
  - answer: `2608.03322v1` — LocAnyMed: Vision-Language Grounding for Multimodal Medical Images

**v3q008** — Weixin Pay billion-scale credit fraud detection graph neural network overlapping subgraphs
  - answer: `2608.02168v1` — Empowering Credit Risk Detection in Weixin Pay with Billion-Scale Deep Graph Learning

**v3q009** — TIDES longitudinal bilingual English Korean dataset next-speaker prediction AMI Meeting Corpus
  - answer: `2608.01724v1` — TIDES: A Longitudinal Bilingual Dataset for Modeling Multi-Party Social Dynamics

**v3q011** — Kolmogorov-Arnold Network versus MLP for faster-than-Nyquist BPSK detection bit error rate
  - answer: `2608.02062v1` — A Comparative Analysis of MLP and Kolmogorov-Arnold Networks (KAN) for Faster-than-Nyquist (FTN) Signaling Detection

**v3q017** — Predictive Enhancement Calibration virtual contrast breast MRI FLUX latent flow transformer MAMA100
  - answer: `2608.03612v1` — Predictive Enhancement Calibration for Latent Breast MRI Virtual Contrast Enhancement

**v3q022** — physics-flavored CNN-Transformer for engineered skeletal muscle contraction Duchenne muscular dystrophy force-time
  - answer: `2608.03927v1` — A Physics-Flavored Transformer Network for Parametrizing Contraction Dynamics of Engineered Skeletal Muscle Tissues

**v3q024** — Fourier motion modeling 4D Gaussian Splatting N3V Google Immersive dynamic novel view
  - answer: `2608.01958v1` — FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis

**v3q029** — offline top-K logits fused chunked KL loss knowledge distillation 32768 tokens H200 GPU
  - answer: `2608.03796v1` — Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss

**v3q034** — Swimm3R underwater Beta Splitting medium-aware SfM Barbados dataset WaterSplatting PSNR
  - answer: `2608.00950v1` — Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction

**v3q036** — SALT subspace-aligned centroid-residual ultra-low-rank LoRA serving vLLM Llama-3.2-3B PCIe
  - answer: `2608.03579v1` — Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training for Efficient Ultra-LoRA Serving

**v3q042** — SAKI score-aware low-rank key indexing KV cache LLaMA 3.1 8B Qwen 2.5 7B top-64 recall
  - answer: `2608.03228v1` — SAKI: Score-Aware Low-Rank Key Indexing for Long-Context KV Retrieval

**v3q044** — WAM-Diff2 autoregressive-to-diffusion distillation autonomous driving VLA FlashInfer CUDA Graphs speedup
  - answer: `2608.01035v1` — WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA

**v3q050** — GUI-Lens coarse-to-fine cropping GUI grounding OCR UI components GPT-5.5
  - answer: `2608.03270v1` — GUI-Lens: Coarse-to-Fine Cropping for GUI Grounding with General-Purpose VLMs

**v3q057** — conditional diffusion synthetic histopathology modified Frechet Inception Distance aggregated Jaccard index nuclei segmentation
  - answer: `2608.03990v1` — Assessment of Conditional Diffusion Model for Synthetic Histopathology Image Generation

**v3q063** — paired recipient evaluation deceased donor kidney transplant survival SRTR concordance index
  - answer: `2608.03017v1` — Paired Recipient-based Evaluation of Survival Prediction for Deceased Donor Kidney Transplants

**v3q068** — ShielDroid hybrid Android malware detection Random Forest Multilayer Perceptron 97.5% accuracy
  - answer: `2608.03250v1` — ShielDroid: A Hybrid Approach Integrating Machine and Deep Learning for Android Malware Detection


## development · topical (12)

**v3q002** — speculative decoding methods that speed up large language model inference by drafting and verifying tokens
  - seed: `2608.02123v1` · pool size 42
  - grade 2: `2608.01651v1` — Bole: Efficient Tree Speculation for Hybrid-Attention Language Models
  - grade 2: `2608.02123v1` — From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding
  - grade 2: `2608.02989v1` — AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding
  - grade 2: `2608.03447v1` — Approximate Speculative Decoding
  - grade 1: `2608.00881v1` — AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving

**v3q003** — preserving plasticity and avoiding catastrophic forgetting in continual or online learning
  - seed: `2608.01475v1` · pool size 37
  - grade 2: `2608.00630v1` — Relative Parameter Importance in Task-Agnostic Replay-Free Continual Learning
  - grade 2: `2608.01184v1` — SAFE-Merge: Data-Free Continual Model Merging with General Knowledge Preservation
  - grade 2: `2608.01252v1` — AdaHAT: Adaptive Hard Attention to the Task in Task-Incremental Learning
  - grade 2: `2608.01475v1` — Plasticity of Growing and Elastic Neural Networks in Online Continual Learning
  - grade 2: `2608.01518v1` — UCBound-Net: Uncertainty-Guided Boundary-Aware Continual Learning for Domain-Incremental Ultrasound Segmentation
  - grade 2: `2608.01743v1` — Toward Plasticity-Preserving KL Regularization for Capability Retention in LLM Reinforcement Learning
  - grade 1: `2608.01314v1` — Remember-R1: Mitigating Long-Context Visual Forgetting through Reinforcement Learning
  - grade 1: `2608.03123v1` — Trajectory-Guided Forget-Recover Network for Continual LLM Unlearning

**v3q005** — many-shot in-context learning behavior and failure modes in vision-language models
  - seed: `2608.02830v1` · pool size 42
  - grade 2: `2608.02830v1` — In-Context Collapse in Vision-Language Models and How to Mitigate it?
  - grade 1: `2608.01575v1` — Measuring in-context algorithmic reasoning in language models against an exact Bayes-optimal standard

**v3q020** — one-step or few-step distillation of diffusion models for faster image editing and object removal
  - seed: `2608.01288v1` · pool size 39
  - grade 2: `2608.01288v1` — TurboClear: One-Step Object-Effect Removal via Region-Calibrated Distribution Matching and Fusion
  - grade 1: `2608.01035v1` — WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA
  - grade 1: `2608.01113v1` — CoT-Edit: Let CoT Guide Instruction Video Editing
  - grade 1: `2608.03059v1` — RIDGE: Re-Noising with Internal Dynamic Guidance for Image Editing

**v3q021** — chain-of-thought monitoring for AI safety and its vulnerability to backdoors or poisoning
  - seed: `2608.02820v1` · pool size 31
  - grade 2: `2608.00583v1` — A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense
  - grade 2: `2608.02089v1` — How Much Does a Reasoning Summary Reveal? An Observability Ladder for Large Language Models
  - grade 2: `2608.02820v1` — Evading Chain-of-Thought Monitoring Through Model Poisoning
  - grade 2: `2608.03291v1` — The Tell-Tale Trace: Detecting Reasoning Failures in LLMs Using Chain-of-Thought Dynamics
  - grade 1: `2608.00732v1` — Mitigating Backdoors via Decoy Shortcuts and Knowledge Decoupling
  - grade 1: `2608.01085v1` — When Collaboration Becomes a Trigger: Collective Evidence-Threshold Backdoors in Multi-Agent Systems
  - grade 1: `2608.01388v1` — Why Formal Monitors Fail: Attack Distribution Entropy as a Coverage Bound for LTL-Based LLM Agent Safety
  - grade 1: `2608.02271v1` — Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures
  - grade 1: `2608.03745v1` — Risky Business: Measuring The Faithfulness-Safety Tension

**v3q023** — detecting AI-generated video or images and generalizing to newly emerging generators
  - seed: `2608.01334v1` · pool size 40
  - grade 2: `2608.00559v1` — Test-Time Curriculum for Open-Set AIGC Detection
  - grade 2: `2608.00716v1` — Generated Images Are Easier to Forget: A Machine Unlearning Perspective for Synthetic Image Detection
  - grade 2: `2608.01258v1` — A Benchmark Dataset for MLLM-Generated Image Detection: GPT Image2 & Nano Banana2
  - grade 2: `2608.01334v1` — SphereVideo: Prototype-anchored Hyperspherical Boundary for Continual AI-generated Video Detection
  - grade 2: `2608.01988v1` — Grounding and Explaining Visual Evidence for AI-Generated Image Detection in Human-Centric Scenes
  - grade 2: `2608.02160v1` — AdaForensics: Learning A Characteristic-aware Adaptive Deepfake Detector
  - grade 2: `2608.03008v1` — V-FIND: Revealing the Intrinsic Forgery Knowledge Encoded in Video Forgery Detectors
  - grade 2: `2608.03096v1` — FakeI2V-Bench: Benchmarking the Applicability of Image-level Deepfake Detectors for Deepfake Video Detection
  - grade 1: `2608.01046v1` — DeBERTa-Sentinel: Toward Transparent and Trustworthy Detection of AI-Generated Text

**v3q025** — efficient serving and inference of mixture-of-experts language models with layer skipping or early exit
  - seed: `2608.00573v1` · pool size 40
  - grade 2: `2608.00573v1` — TrimMoE A communication aware and adaptive depth framework for distributed edge inference
  - grade 2: `2608.01784v1` — REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models
  - grade 1: `2608.01536v1` — Celty: SpMspV GPU Kernel and SIMT Co-Design for Efficient Dual-Sparse LLM Inference
  - grade 1: `2608.01651v1` — Bole: Efficient Tree Speculation for Hybrid-Attention Language Models
  - grade 1: `2608.03036v1` — LLM Serving in the Wild: An Empirical Study of Frameworks, Methods, and System Designs

**v3q032** — retaining prior capabilities when fine-tuning large language models, measuring and penalizing forgetting
  - seed: `2608.03887v1` · pool size 37
  - grade 2: `2608.01743v1` — Toward Plasticity-Preserving KL Regularization for Capability Retention in LLM Reinforcement Learning
  - grade 2: `2608.03887v1` — Omega-S: A Functional Resilience Index for LLM Fine-Tuning
  - grade 1: `2608.01314v1` — Remember-R1: Mitigating Long-Context Visual Forgetting through Reinforcement Learning
  - grade 1: `2608.03874v1` — ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?

**v3q038** — robustness and uncertainty in reinforcement learning objectives under misspecified rewards or utilities
  - seed: `2608.03562v1` · pool size 40
  - grade 2: `2608.02509v1` — Optimizing Minimax Regret in Uncertain MDPs with Small Sets of Policies
  - grade 2: `2608.03562v1` — Robust General Utility for Reinforcement Learning
  - grade 1: `2608.01151v1` — Learning-Based Stochastic Optimal Control with Infinite-Horizon Probabilistic Constraints
  - grade 1: `2608.01556v1` — Rethinking Personalized Reward Modeling for LLMs under Preference Heterogeneity via Group-Debiased Federated Learning
  - grade 1: `2608.02034v1` — Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning
  - grade 1: `2608.03069v1` — Revisiting TD Target Aggregation under Uncertainty in Q-Learning

**v3q058** — process-level reward and credit assignment for search-augmented or tool-using reasoning agents
  - seed: `2608.01321v1` · pool size 37
  - grade 2: `2608.01321v1` — BiCAA: Bidirectional Credit Assignment for Search-Augmented Agent
  - grade 2: `2608.01597v1` — HindSearch: Trajectory-Level Hindsight Critique for Search-Augmented Reinforcement Learning
  - grade 2: `2608.01867v2` — CRISP: Critical Step Perception for Training Efficient Deep Search Agents
  - grade 2: `2608.03467v1` — When Correct Solutions Repeat: Rarity-Aware Credit Redistribution for GRPO
  - grade 2: `2608.04007v1` — TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning
  - grade 1: `2608.01358v1` — HopRefusalBench: Diagnosing Refusal Failures in Search-Augmented Agents for Multi-Hop Reasoning
  - grade 1: `2608.01359v1` — EviSD: Evidence-Conditioned Self-Distillation for Search-Augmented Agents
  - grade 1: `2608.01913v1` — Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents
  - grade 1: `2608.02585v1` — GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning

**v3q061** — dataset distillation that compresses a training set into a small synthetic set for pretrained encoders
  - seed: `2608.03218v1` · pool size 36
  - grade 2: `2608.03218v1` — Self-Supervised Representation-Guided Generative Dataset Distillation
  - grade 2: `2608.03269v1` — Efficient Video Dataset Distillation via Cluster-Guided Prototype Blending

**v3q072** — super-resolution of medical or pathology images that preserves fine cellular structure
  - seed: `2608.03664v1` · pool size 40
  - grade 2: `2608.03540v1` — S$^3$-Diff: Structural Semantic Synergy Diffusion Model for High Fidelity Super Resolution of Pathological Images
  - grade 2: `2608.03664v1` — Morphology-Aware Implicit Super-Resolution Network for Pathological Images


## heldout · paraphrase (12)

**v3q037** — a three-way cancer prognosis model that turns sparse tabular patient records into text embeddings and uses them as an anchor to align pathology and genomic signals through cross-attention and a distribution-matching objective
  - answer: `2608.03247v1` — CIGTSurv: Clinical Information Guided Tri-modal Survival Prediction with Local Prototype Association and Global Feature Alignment

**v3q039** — a content-policy method for prompt-to-picture diffusion applied at inference that reads the predicted denoised frame to catch banned material and then tweaks a low-rank residual in the conditioning to suppress it, leaving weights unchanged
  - answer: `2608.03284v1` — Test-Time Scaling for Safe Text-Guided Image Generation via Intermediate Clean Estimates

**v3q040** — a stochastic form of low-dimensional adapter adaptation that samples structured variations along the leading components of shared adapters to give reliable predictive uncertainty while matching the deterministic transform in expectation
  - answer: `2608.01142v2` — EulerLoRA: Rank-Driven Jump Dynamics for Calibrated Parameter-Efficient Fine-Tuning

**v3q041** — using a vision-language model's grounding to synthesize varied foregrounds and backgrounds and to inject representation noise, improving scarce-label recognition of items when training and evaluation distributions differ sharply
  - answer: `2608.01348v1` — Prompt-Driven Simulation with Feature Perturbation for Cross-Domain Few-Shot Object Detection

**v3q046** — an interactive guessing game revealing chatbots gather evidence poorly across dialogue rounds for explanation-forming reasoning: many commit before using clues, others exhaust their budget without converging
  - answer: `2608.03388v1` — Don't Let Me Ask for It: LLMs Show Deficiencies in Active Multi-Turn Information Acquisition for Abductive Inference

**v3q048** — making inter-assistant social relations explicit in prompts and finding they mainly push teams toward agreement, helping when consensus is rewarded but not reliably improving accuracy in objective question debates
  - answer: `2608.03239v1` — Relational Priors as Convergence Pressure in LLM-Based Multi-Agent Systems

**v3q051** — quantifying how well annotators agree when they give free-form open-ended labels for life-science passages, comparing embedding, large-language-model, and entailment-based soft reliability scores
  - answer: `2608.03529v1` — Consensus Measures for Unstructured Biomedical Text Annotations

**v3q052** — deriving gaze-like attention labels from finished operations by combining deformation-constrained organ tracking with instrument paths, powering an assistive camera that pre-frames relevant regions and eases the operator's mental load
  - answer: `2608.02471v1` — Action-grounded tissue affordance enables anticipatory auto-framing that lowers surgeon cognitive workload during laparoscopic surgery

**v3q053** — an analysis of how the training objective for image generators — raw pixels versus autoencoder latents versus self-supervised features — shifts the modeling burden across context inference and per-token denoising
  - answer: `2608.00626v1` — Where Does Generative Difficulty Reside? An Empirical Study of Target Representations

**v3q054** — speeding up decoding of networks that blend full and linear-recurrent layers by rewriting the recurrence into a branch-structured closed form and a GPU kernel that checks all draft nodes at once with far less transient state memory
  - answer: `2608.01651v1` — Bole: Efficient Tree Speculation for Hybrid-Attention Language Models

**v3q055** — a low-latency causal clip reviser that generates chunk by chunk, preserving source fidelity and long-horizon temporal consistency without future frames or a fixed duration, at roughly thirty frames per second
  - answer: `2608.03974v1` — JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion

**v3q059** — a terminal benchmark that hides tool meanings so assistants must learn behavior by trial and error, showing they fall back to exhaustive probing under mapping drift despite recorded cues to the following action
  - answer: `2608.02358v1` — ScrambleToolBench: Agents Search Exhaustively Even When Their Own Map Points to the Next Step


## heldout · lexical (8)

**v3q043** — LLM-Guided Retrieval molecular perturbation response Tahoe-100M single-cell atlas unseen cell line
  - answer: `2608.01734v1` — LLM-Guided Retrieval for Prediction of Molecular Perturbation Responses

**v3q045** — standalone DINOv3 DINO.txt training-free open-vocabulary segmentation remote sensing UDD5 DOTA LoveDA
  - answer: `2608.03023v1` — Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing

**v3q056** — MARBERT emoji pragmatics Arabic digital discourse Facebook politeness respect solidarity F1
  - answer: `2608.01174v1` — Does Machine "know" interpersonal pragmatics? Evidence from MARBERT's learning of emoji pragmatics in Arabic digital discourse

**v3q060** — writing-system-level tokenizer adaptation byte-level BPE Ukrainian Nemotron GPT-OSS merge ordering
  - answer: `2608.00582v1` — Writing-System-Level Tokenizer Adaptation for Byte-Level BPE

**v3q062** — OSSDD OpenSARShip Sentinel-1 ship detection dataset VV VH polarization Faster R-CNN FCOS DETR
  - answer: `2608.01963v1` — OSSDD - a New Open Dataset for Sentinel-1 Ship Detection

**v3q065** — nGPT normalized Transformer hypersphere Mamba-2 Mixture-of-Experts 14B GatedAdamW training recipe
  - answer: `2608.01284v1` — Training nGPT

**v3q074** — OliveGemma PaliGemma-2-3B LoRA Mediterranean European diet food recognition MedGR ODIN VIPPSTAR
  - answer: `2608.03428v1` — OliveGemma: A 3 Billion Visual Language Model for Recognising the Mediterranean & European Diet

**v3q078** — condition-number barrier sparse least squares Axiotis Sviridenko Small-Set Expansion Hypothesis Gemini agentic proof
  - answer: `2608.02588v1` — The Condition-Number Barrier in Sparse Least Squares


## heldout · topical (6)

**v3q047** — benchmarks that evaluate controllable video generation models as world models beyond visual quality
  - seed: `2608.02603v1` · pool size 38
  - grade 2: `2608.02603v1` — WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity
  - grade 1: `2608.00617v1` — Diagnosing Under-Development of Irreversible Processes in Video Generation
  - grade 1: `2608.01127v2` — MiniWorld: Democratizing the Training of Video World Models from Scratch
  - grade 1: `2608.02953v1` — RealWeather: Realistic and Scene-Faithful Weather Translation with Driving World Models
  - grade 1: `2608.03084v1` — SUV: Future Scene Understanding as Video Generation for End-to-End Driving
  - grade 1: `2608.03211v1` — CrossScope: A Role-Asymmetric World Model for Joint Dual-Scope Surgical Video Prediction

**v3q049** — sparse-view or few-shot 3D Gaussian Splatting reconstruction and super-resolution
  - seed: `2608.02206v1` · pool size 27
  - grade 2: `2608.01588v1` — D^2-4DGS: Dual-Depth Guided Sparse-Camera 4D Gaussian Splatting
  - grade 2: `2608.02145v1` — UniqueSplat: View-conditioned 3D Gaussian Splatting for Generalizable 3D Reconstruction
  - grade 2: `2608.02191v1` — DerainSplat: Feed-Forward Clean 3D Gaussian Splatting from Sparse Rainy Views
  - grade 2: `2608.02206v1` — CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution
  - grade 2: `2608.02437v2` — InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis
  - grade 1: `2608.00950v1` — Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction
  - grade 1: `2608.01178v1` — DynActiveGS: Active Gaussian Splatting for Dynamic Scene Reconstruction
  - grade 1: `2608.01186v1` — QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction
  - grade 1: `2608.01659v1` — StreamSplat: Streaming Feed-Forward 3D Gaussian Splatting
  - grade 1: `2608.01958v1` — FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis

**v3q064** — black-box detection of hallucination in large vision-language models using consistency signals
  - seed: `2608.03817v1` · pool size 42
  - grade 2: `2608.01207v1` — It's the Decoding Format, Not the Perturbation: Auditing Consistency-Based Selection for Vision-Language Test-Time Scaling
  - grade 2: `2608.03817v1` — UHP Detection: LVLMs have their Unique Hallucination Pattern in the Consistency Space
  - grade 1: `2608.01021v1` — Can Humans Dream of Electric Sheep? Human-Written Samples for Fine-Grained Vision-and-Language Hallucination Benchmarking
  - grade 1: `2608.02790v1` — Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI
  - grade 1: `2608.03720v1` — Detecting Hallucinations and Recovering Verified Answers in Arabic Islamic Question Answering
  - grade 1: `2608.03966v1` — HalluTruthQA-4K: A Fine-Grained Corpus and Annotation Process for Arabic Hallucination Detection and Truth Verification

**v3q069** — multimodal molecular representation learning linking chemical structure to cellular phenotypes for drug discovery
  - seed: `2608.02688v1` · pool size 34
  - grade 2: `2608.01734v1` — LLM-Guided Retrieval for Prediction of Molecular Perturbation Responses
  - grade 2: `2608.02688v1` — Learning Molecular Representations from Cellular Phenotypes with Structure Preservation
  - grade 1: `2608.00985v1` — Beyond Gene Reconstruction: Learning Cell Representations through Complementary Transcriptomic Views
  - grade 1: `2608.01007v1` — Fused Bayesian Flow Networks for Dual-Target Molecular Design
  - grade 1: `2608.02027v1` — Scikit-fingerprints: Python library for scikit-learn compatible molecular fingerprints and chemoinformatics
  - grade 1: `2608.03260v1` — ED-DiT: Physics-Guided Diffusion Pretraining for Transferable Molecular Representations from Electron Density
  - grade 1: `2608.03855v1` — Bi-semantic Chemical Embedder for Joint Representation Learning of SMILES and Natural Language

**v3q076** — combining or fusing multiple pathology foundation models for tile-level representations
  - seed: `2608.01370v1` · pool size 35
  - grade 2: `2608.01370v1` — Understanding Synergistic Interactions among Pathology Foundation Models via Adaptive Fusion
  - grade 1: `2608.01356v1` — Harnessing Adversarial Distillation to Customise Debiased, Disease-Specific Pathology Foundation Models for Breast Cancer
  - grade 1: `2608.03079v1` — CorePath: A Breast-Specialized Pathology Foundation Model for Core Needle Biopsy Diagnosis and Risk-Controlled Report Generation
  - grade 1: `2608.03508v1` — From Multi-Resolution Cells to Gigapixel Whole Slide Images Foundation Model for Computational Pathology

**v3q077** — KL regularization to retain base-model capabilities during reinforcement-learning post-training of language models
  - seed: `2608.01743v1` · pool size 43
  - grade 2: `2608.01743v1` — Toward Plasticity-Preserving KL Regularization for Capability Retention in LLM Reinforcement Learning
  - grade 1: `2608.03573v1` — SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs


## abstention — out-of-domain negatives

- (development) How do I replace a leaking kitchen faucet cartridge?
- (development) What ingredients make a traditional sourdough starter?
- (development) Who won the 1974 association football world championship?
- (development) How should a violin bow be rehaired?
- (development) Explain the rules for castling in tournament chess.
- (development) What soil mixture is best for growing desert cacti at home?
- (development) How is a residential property deed transferred?
- (development) Give a step-by-step recipe for laminated croissant dough.
- (development) Why did the Roman Republic replace its kings?
- (development) How do I tune the carburetor on a vintage motorcycle?
- (heldout) What is the safest way to clean a wool overcoat?
- (heldout) How are points scored in competitive badminton?
- (heldout) Which herbs should be planted beside tomatoes?
- (heldout) How do fixed-rate home mortgages calculate monthly payments?
- (heldout) What caused the eruption of Mount Vesuvius in antiquity?

## abstention — near-miss negatives (real ML topics verified absent)

- (development) deep learning models for detecting sarcasm in social media text
- (development) neural networks for fake news detection and misinformation classification
- (development) transformer models for hate speech detection in online comments
- (development) crowd counting and density estimation from surveillance images
- (development) lane detection for autonomous driving perception
- (development) sound event detection and tagging in audio recordings
- (development) deep learning for stock price and stock market movement prediction
- (development) short-term electricity load forecasting with neural networks
- (heldout) reinforcement learning for adaptive traffic signal control
- (heldout) vision-based control for autonomous drone racing and quadrotors
- (heldout) crop yield prediction from satellite remote sensing
- (heldout) handwriting recognition for historical manuscripts
