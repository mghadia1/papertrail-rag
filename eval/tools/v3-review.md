# v3 question set — review draft (NOT frozen)

Queries LLM-authored (Claude). Dev topical grades are Claude-drafted, to be re-graded blind by a human before freeze. Held-out authored from a disjoint fresh sample and never dry-run.

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

**v3q045** — estimating action values from logged interaction data by swapping the symmetric several-back temporal-difference loss for an asymmetric one, which counters the pessimistic bias that grows with longer horizons
  - answer: `2608.02034v1` — Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning

**v3q046** — predicting future scene descriptors for self-driving robots entirely in latent space and reading task outputs straight from those predictions, dropping the heavy module that usually maps states to tasks
  - answer: `2608.02428v1` — DF$^3$: World Modeling via Decoder-Free Feature Forecasting in Autonomous Navigation

**v3q048** — a training-free way to shrink redundant audio and visual sequences for combined-sensory chat systems, pruning structurally before the network and consolidating semantically inside it to hold accuracy at tiny budgets
  - answer: `2608.03812v1` — OmniPack: Unified Token Compression for Efficient Omni-modal Large Language Models

**v3q049** — a review of patient-specific heart-and-vessel computer models that update with clinical measurements, contrasting mechanistic simulations with scalable learned and hybrid graph methods for diagnosis and therapy planning
  - answer: `2608.02135v1` — Cardiovascular Digital Twins from Physics Based to Data Driven Approaches

**v3q051** — a feed-forward Gaussian scene reconstructor that separates shape modeling from color detail into two branches, letting it work without known camera poses and render crisper novel views
  - answer: `2608.01186v1` — QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction

**v3q052** — toughening learned near-duplicate image matchers so adversarial tweaks cannot slip past, with no model re-fitting: randomized smoothing at match time plus an imperceptible mark added to reference images before release
  - answer: `2608.03101v1` — Double Down on Defense: Strengthening Deep Perceptual Hashes against Evasion Attacks without Retraining


## development · lexical (16)

**v3q006** — medical visual grounding trained on the LocAnyMed-200K dataset
  - answer: `2608.03322v1` — LocAnyMed: Vision-Language Grounding for Multimodal Medical Images

**v3q008** — billion-scale credit fraud detection on Weixin Pay using graph neural networks
  - answer: `2608.02168v1` — Empowering Credit Risk Detection in Weixin Pay with Billion-Scale Deep Graph Learning

**v3q009** — next-speaker prediction on the TIDES bilingual multi-party conversation dataset
  - answer: `2608.01724v1` — TIDES: A Longitudinal Bilingual Dataset for Modeling Multi-Party Social Dynamics

**v3q011** — Kolmogorov-Arnold Networks for faster-than-Nyquist signaling detection
  - answer: `2608.02062v1` — A Comparative Analysis of MLP and Kolmogorov-Arnold Networks (KAN) for Faster-than-Nyquist (FTN) Signaling Detection

**v3q017** — virtual contrast enhancement for breast MRI using a FLUX latent flow transformer
  - answer: `2608.03612v1` — Predictive Enhancement Calibration for Latent Breast MRI Virtual Contrast Enhancement

**v3q022** — a physics-flavored network parametrizing contraction dynamics of engineered skeletal muscle tissues
  - answer: `2608.03927v1` — A Physics-Flavored Transformer Network for Parametrizing Contraction Dynamics of Engineered Skeletal Muscle Tissues

**v3q024** — Fourier motion modeling for dynamic 4D Gaussian Splatting on the N3V benchmark
  - answer: `2608.01958v1` — FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis

**v3q029** — efficient LLM distillation with offline top-K logits and a fused chunked KL loss
  - answer: `2608.03796v1` — Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss

**v3q034** — underwater 3D reconstruction results on the Barbados dataset compared with WaterSplatting
  - answer: `2608.00950v1` — Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction

**v3q036** — ultra-low-rank LoRA serving with subspace-aligned centroid-residual training in vLLM
  - answer: `2608.03579v1` — Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training for Efficient Ultra-LoRA Serving

**v3q038** — score-aware low-rank key indexing for long-context KV cache compression (SAKI)
  - answer: `2608.03228v1` — SAKI: Score-Aware Low-Rank Key Indexing for Long-Context KV Retrieval

**v3q039** — autoregressive-to-diffusion distillation for an efficient autonomous-driving VLA
  - answer: `2608.01035v1` — WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA

**v3q040** — coarse-to-fine cropping for GUI grounding with general-purpose VLMs
  - answer: `2608.03270v1` — GUI-Lens: Coarse-to-Fine Cropping for GUI Grounding with General-Purpose VLMs

**v3q041** — evaluating conditional diffusion for synthetic histopathology with pathology-specific FID
  - answer: `2608.03990v1` — Assessment of Conditional Diffusion Model for Synthetic Histopathology Image Generation

**v3q044** — paired-recipient evaluation of survival prediction for deceased-donor kidney transplants using SRTR data
  - answer: `2608.03017v1` — Paired Recipient-based Evaluation of Survival Prediction for Deceased Donor Kidney Transplants

**v3q047** — hybrid Android malware detection combining Random Forest and a multilayer perceptron
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

**v3q037** — robustness and uncertainty in reinforcement learning objectives under misspecified rewards or utilities
  - seed: `2608.03562v1` · pool size 40
  - grade 2: `2608.02509v1` — Optimizing Minimax Regret in Uncertain MDPs with Small Sets of Policies
  - grade 2: `2608.03562v1` — Robust General Utility for Reinforcement Learning
  - grade 1: `2608.01151v1` — Learning-Based Stochastic Optimal Control with Infinite-Horizon Probabilistic Constraints
  - grade 1: `2608.01556v1` — Rethinking Personalized Reward Modeling for LLMs under Preference Heterogeneity via Group-Debiased Federated Learning
  - grade 1: `2608.02034v1` — Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning
  - grade 1: `2608.03069v1` — Revisiting TD Target Aggregation under Uncertainty in Q-Learning

**v3q042** — process-level reward and credit assignment for search-augmented or tool-using reasoning agents
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

**v3q043** — dataset distillation that compresses a training set into a small synthetic set for pretrained encoders
  - seed: `2608.03218v1` · pool size 36
  - grade 2: `2608.03218v1` — Self-Supervised Representation-Guided Generative Dataset Distillation
  - grade 2: `2608.03269v1` — Efficient Video Dataset Distillation via Cluster-Guided Prototype Blending

**v3q050** — super-resolution of medical or pathology images that preserves fine cellular structure
  - seed: `2608.03664v1` · pool size 40
  - grade 2: `2608.03540v1` — S$^3$-Diff: Structural Semantic Synergy Diffusion Model for High Fidelity Super Resolution of Pathological Images
  - grade 2: `2608.03664v1` — Morphology-Aware Implicit Super-Resolution Network for Pathological Images


## heldout · paraphrase (12)

**v3q055** — a message-passing operator for networks whose edges carry both a sign and an orientation, moving information only where node-potential gaps agree with the arrow, improving node classification and link prediction
  - answer: `2608.00836v1` — Nonlinear Laplacians Improve Signed-Directed Graph Learning

**v3q056** — testing the belief that shorter prompts help when the needed information is kept, by either trimming the middle or dropping only the irrelevant parts, and showing naive middle-removal merely measures how often it spares the answer
  - answer: `2608.03297v1` — Distractor-Aware Truncation: Disentangling Context-Length Effects from Signal Loss in Long-Context LLM Benchmarks

**v3q057** — a formal account of when a metric reads healthy while the model is actually broken, spanning gamed reward models at fit time and unmonitored production faults, with a taxonomy validated on real incidents
  - answer: `2608.02786v1` — Evaluation Blindness: How Silent Measurement Failures Corrupt AI Systems from Training to Deployment

**v3q058** — a training-free method and dataset for judging which of several same-category objects sits nearest a reference item in one photo, estimating floor-plane distances with uncertainty-aware prompting
  - answer: `2608.01709v1` — SpatialQuery: Benchmarking Geometry-Grounded Multi-Instance Spatial Reasoning in Vision-Language Models

**v3q059** — checking whether medical claims are true, false, or misleading by pulling trusted passages from WHO and a national disease-control agency and classifying with a fine-tuned encoder, tested on Nigerian fact-checks
  - answer: `2608.02310v1` — An Evidence-Grounded Retrieval-Augmented Transformer Framework for Health Misinformation Verification

**v3q062** — scaling specific feed-forward units in a chatbot to reproduce dementia-like speech changes such as reduced idea density and worse recall, showing units found from clinical transcripts causally shape behavior
  - answer: `2608.03067v1` — Activation-Guided Neuron Intervention to Induce Alzheimer's-Related Computational Language Phenotypes in a Large Language Model

**v3q064** — synthesizing audio-driven expressive portrait video that mixes implicit semantic features with explicit blendshape priors to give precise, continuous control over expression intensity without losing texture detail
  - answer: `2608.00663v1` — Geometry-guided Emotion Modulation for Controllable and Photorealistic Emotional Talking Face Generation

**v3q069** — a training-free fix for text-to-image models that merge or drop objects, correcting the initial attention allocation once so overlapping subjects separate, rather than steering the whole sampling path
  - answer: `2608.03135v1` — Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds

**v3q072** — learning end-to-end which higher-order structures like cycles and cliques to add to a graph model, instead of fixing them beforehand with an unsupervised rule, improving node and graph classification
  - answer: `2608.01160v1` — Differentiable Lifting for Topological Neural Networks

**v3q074** — predicting sustained high-water plateaus for early warning by combining readings from many monitoring gages with bounded corrections that keep each site's local time-series forecast as a stable anchor
  - answer: `2608.01775v1` — Multi-Source Dynamic Graph Learning for Compound-Flood Forecasting in Managed Coastal Systems

**v3q077** — correcting coarse- and fine-mesh numerical solutions with a small learned corrector to price multi-asset financial derivatives faster, demonstrated on Black-Scholes and Heston barrier contracts with little high-fidelity data
  - answer: `2608.02778v1` — Neural Networks with Local Converging Inputs for Efficient Options Pricing Models

**v3q078** — estimating how many overlapping decaying sinusoids fill a plate-reverb impulse response by predicting mode tallies in several spectral ranges with a tree regressor, then fitting decay and gain on grids via an all-pole model
  - answer: `2608.00667v1` — Band-Count Dense Modal Estimation with Fixed-Frequency Differentiable Resonator Refinement


## heldout · lexical (8)

**v3q053** — near-real-time object removal attacks on video perception evaluated on the South Carolina Connected Vehicle Testbed
  - answer: `2608.02806v1` — Fast Object Removal Attacks on Safety-Critical Video-based Perception Systems

**v3q060** — monocular panoramic SLAM using a frozen geometry foundation model for loop closure (HALO-SLAM)
  - answer: `2608.00925v1` — Look Up and Look Back: Hidden Attention and Latent Orientation in a Frozen Foundation Model for Panoramic SLAM

**v3q063** — zero-shot humanoid motion tracking on the Unitree G1 with robot-native motion generation
  - answer: `2608.01410v1` — GenTrack: Physical Alignment for Robot-Native Motion Generation and Zero-Shot Humanoid Tracking

**v3q066** — the Observatorio Lazaro database tracking anglicism usage in the Spanish press
  - answer: `2608.00713v1` — Observatorio Lazaro: A self-populating database of anglicism usage in the Spanish press

**v3q068** — estimating plate-reverb parameters for Task A of the DAFx Parameter Estimation Challenge
  - answer: `2608.00656v1` — Simulation-Based Plate-Reverb Parameter Estimation from a Single Impulse Response

**v3q070** — neuro-symbolic biomedical relation extraction evaluated on DDI and ChemProt (ANCHOR-RE)
  - answer: `2608.03154v1` — ANCHOR-RE: An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction

**v3q075** — physics-guided deep learning for patient-specific microwave ablation planning of liver tumors
  - answer: `2608.03086v1` — Automatic Patient-Specific Microwave Ablation Planning Accelerated by a Physics-Guided Deep Learning Model

**v3q076** — a machine-readable catalogue and handwriting-recognition accuracy measure for the Tsiolkovsky archive, fond 555
  - answer: `2608.03617v1` — A machine-readable catalogue of the Tsiolkovsky papers (fond 555, Archive of the Russian Academy of Sciences), and a way to measure how well its handwriting can be read


## heldout · topical (6)

**v3q054** — graph-structured retrieval-augmented generation for complex multi-document question answering
  - seed: `2608.01565v1` · pool size 39
  - grade 2: `2608.00585v1` — Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG, and Decomposition Repairs It
  - grade 2: `2608.01269v2` — ACE-GraphRAG: Agentic Context Engineering for Hierarchical GraphRAG
  - grade 2: `2608.01468v1` — Retrieval Augmented Biomedical Question Answering with Weak Question Recovery and Neural Reranking for BioASQ Task 14b
  - grade 2: `2608.01565v1` — DocNavRAG: Document-Structured Graph RAG with Stateful Evidence Construction for Complex Document Question Answering
  - grade 1: `2608.00658v1` — Select-And-Extract: A Lightweight Plugin for Retrieval-Augmented Generation
  - grade 1: `2608.00712v1` — Exploiting Intrinsic Duality for Multi-Hop Question Generation
  - grade 1: `2608.00765v1` — RAGOCR: Optical Compression of Retrieval-Augmented Text via Visual Representation
  - grade 1: `2608.01311v1` — RH-RAG: Trustworthy Long-Form Generation for Privacy-Constrained Settings
  - grade 1: `2608.01630v1` — RING: Retrieval-Internalized Generation for Continual Large-Scale Knowledge Injection
  - grade 1: `2608.02678v1` — DenialRAG: Single-Document RAG Poisoning via Embedded Parametric Denial
  - grade 1: `2608.03860v1` — SciRet: A Compute-Aware Empirical Study of Retrieval and Reranking for Scientific RAG

**v3q061** — test-time reinforcement learning for LLM reasoning using majority-vote pseudo-labels
  - seed: `2608.03545v1` · pool size 39
  - grade 2: `2608.03545v1` — Hi-TTRL: Regulating Consensus with Hints for Test-Time Reinforcement Learning
  - grade 1: `2608.01014v1` — Cloud-ScPO: Hidden-State Geometry for Semi-Supervised Preference Optimization in LLM Reasoning
  - grade 1: `2608.02585v1` — GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning
  - grade 1: `2608.02951v1` — SP3O: Reinforcement Learning from Segment Preferences without Reward Modeling
  - grade 1: `2608.04001v1` — Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility

**v3q065** — evidence frame selection for long-video question answering under a fixed token budget
  - seed: `2608.01660v1` · pool size 36
  - grade 2: `2608.00714v1` — Coverage-Driven Adaptive Keyframe Selection for Video Understanding
  - grade 2: `2608.01660v1` — Ground, Cover, and Refine: Evidence-Centric Frame Selection for Long-Video Question Answering
  - grade 2: `2608.03918v1` — When and Where to Look: Adaptive Visual Evidence Scheduling for Efficient Long Video Understanding
  - grade 1: `2608.01169v1` — Think in Sets for Streaming Video Token Compression
  - grade 1: `2608.01980v1` — AdaThinkV: Adaptive Thinking for Token-Efficient Video Reasoning
  - grade 1: `2608.02078v1` — CAVE: Competence-Aware Visual Boundary Evidence Alignment for Video Temporal Grounding
  - grade 1: `2608.03083v1` — GSTEP: Global Spatio-Temporal Density-Driven Visual Token Pruning for Efficient Video Large Language Models
  - grade 1: `2608.03112v1` — Adaptive Two-Stage Visual Token Pruning for Efficient Inference in Video-Language Models

**v3q067** — ultra-low-bit KV-cache quantization for long-context LLM inference
  - seed: `2608.02691v1` · pool size 36
  - grade 2: `2608.02691v1` — Output-Aware Rotation for INT2 KV-Cache Quantization
  - grade 1: `2608.00528v1` — S$^4$R: Selective Sampling, Subspaces, and Sparse Reconstruction for Compressed Long-Context KV Caching
  - grade 1: `2608.00902v1` — Practical Online KV Cache Compaction for LLM Agents: An Empirical Study
  - grade 1: `2608.01247v1` — RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Cache Eviction
  - grade 1: `2608.01631v1` — Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression
  - grade 1: `2608.02901v1` — AnchorKV: Anchor-Residual KV Cache Compression
  - grade 1: `2608.03893v1` — Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

**v3q071** — MLLM-based referring and reasoning image segmentation with efficient mask prediction
  - seed: `2608.02791v1` · pool size 39
  - grade 2: `2608.01354v1` — PixVL: Self-Supervised Training of Pixel-Level MLLMs via a Unified Mask--Text Consistency Cycle
  - grade 2: `2608.02791v1` — Better, Stronger, Faster, and Broader: Structured All-Mask Prediction for MLLM-Based Segmentation
  - grade 2: `2608.03147v1` — CROSS: Cascaded Distillation and Dual-Constraint Grounding for Remote Sensing Referring Segmentation
  - grade 1: `2608.01663v1` — Few-Shot Concept Prompt Learning for Segmentation Foundation Models via Visual Grounding
  - grade 1: `2608.02284v1` — EOVSAM: Efficient Open-Vocabulary Segmentation with SAM 3 in One Pass
  - grade 1: `2608.02470v1` — Grounding Agentic VLMs with Dedicated Segmentation for Fine-Grained Vehicle Damage Assessment
  - grade 1: `2608.03911v1` — UniEvo-RS: Omni-Prompt Unified Remote Sensing Segmentation with Representative Exemplar-Driven Prototype Evolution

**v3q073** — online high-definition map construction for autonomous driving from multi-modal inputs
  - seed: `2608.01338v1` · pool size 40
  - grade 2: `2608.01338v1` — Driver2Map: Imitating Human Driving for Online High-Definition Map Construction
  - grade 1: `2608.01201v1` — PRISM: Privileged Probabilistic Latent Supervision for End-to-End Autonomous Driving Motion Planning
  - grade 1: `2608.01535v1` — STAR-VLM: Spatiotemporal Grounding Vision-Language Models for Motion and Velocity Estimation via Automotive Radar Supervision
  - grade 1: `2608.02449v1` — MoRAL: Sensor-Grounded BEV Reasoning for Compact VLMs toward Edge-Oriented Autonomous Driving
  - grade 1: `2608.03084v1` — SUV: Future Scene Understanding as Video Generation for End-to-End Driving


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
- (heldout) How do I prune an overgrown apple tree in late winter?
- (heldout) What is the proper way to season a new cast-iron skillet?
- (heldout) What are the regulation dimensions of a singles tennis court?
- (heldout) How is a traditional garam masala spice blend made?
- (heldout) What knots secure a canoe to a car roof rack?

## abstention — near-miss negatives (real ML topics verified absent)

- (development) deep learning models for detecting sarcasm in social media text
- (development) neural networks for fake news detection and misinformation classification
- (development) transformer models for hate speech detection in online comments
- (development) crowd counting and density estimation from surveillance images
- (development) lane detection for autonomous driving perception
- (development) sound event detection and tagging in audio recordings
- (development) deep learning for stock price and stock market movement prediction
- (development) short-term electricity load forecasting with neural networks
- (heldout) protein structure prediction with deep learning
- (heldout) homomorphic encryption for privacy-preserving neural network inference
- (heldout) gravitational-wave signal detection with neural networks
- (heldout) exoplanet detection from transit light curves with machine learning
