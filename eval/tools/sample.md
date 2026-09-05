# v3 labeling sample — 90 papers (seed 20260903)

One query per paper. Do NOT look at any retrieval output while writing queries. For paraphrase questions, no word from the title may appear in the query (enforced later by a script).

## 1. `2608.03275v1`

**MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation**

Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of large language models, but existing MoE-based PEFT methods typically improve capacity by storing multiple full LoRA experts, causing adapter storage to grow linearly with the number of experts and restricting adaptation to a fixed expert pool. We ask whether MoE-based PEFT can produce instance-specific adaptations without explicitly storing a separate LoRA module for each expert. To address this gap, we propose MoEGen, an adaptation framework that shifts MoE-based PEFT from expert selection to expert-conditioned parameter generation. Instead of storing each expert as a full LoRA adapter, MoEGen represents each expert as a small learnable vector, termed an expert code. It routes each input over these vectors and uses their weighted combination to condition a lightweight hypernetwork that generates input-specific low-rank updates. This design decouples expert capacity from adapter storage while enabling instance-conditioned adaptation. Experiments on eight commonsense reasoning benchmarks show consistent improvements over strong static and MoE-based PEFT baselines across three backbones. MoEGen also performs strongly in joint medical and legal-domain adaptation.

- type: 
- query: 

---

## 2. `2608.02123v1`

**From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding**

Speculative decoding accelerates LLM inference only when drafted continuations survive target-model verification. Semi-autoregressive drafters such as DSpark predict an entire token block with one backbone forward and refine it with a lightweight Markov head. However, DSpark decodes this block as a single chain, so an early mismatch invalidates the remaining suffix and limits the benefit of large draft blocks. We show that the conditional structure already learned by DSpark can support multiple parent-consistent continuations without retraining or additional backbone passes. We introduce Parent-Conditioned Drafting Tree (PCTree), which uses the pretrained Markov head to score alternative children separately for each concrete parent and allocates a fixed verification budget to the most probable paths. This converts DSpark's linear draft into a tree while preserving its one-pass parallel backbone. Across Qwen3-{4B,8B,14B} and nine benchmarks, at $B{=}7$, measured speedup gains over autoregressive (AR) decoding, relative to matched DSpark, range from $3.1\%$ to $29.5\%$. On Qwen3-4B GSM8K at $B{=}16$, PCTree increases mean acceptance length from $9.41$ to $11.16$ and three-run mean AR speedup from $6.14{\times}$ to $6.60{\times}$. These show that parent-conditioned branching can turn conditional capacity already present in a semi-autoregressive drafter into end-to-end inference gains through an inference-only change.

- type: 
- query: 

---

## 3. `2608.01475v1`

**Plasticity of Growing and Elastic Neural Networks in Online Continual Learning**

Neural networks that can grow or both grow and shrink during learning, referred to as growing neural networks and elastic neural networks, respectively, have recently been explored in offline continual learning with a particular focus on catastrophic forgetting. Driven by the observations that 1) online continual learning closely resembles how animals learn; 2) loss of plasticity---the progressive decline in a learning network's ability to learn---is another crucial challenge facing continual learning; and 3) incremental introduction of randomly initialized hidden units was recently shown to help preserve plasticity, in this paper, we study the plasticity of several foundational growing and elastic networks in online continual learning. Our experiments in supervised learning settings show that adaptive growing networks, which incrementally incorporate new, randomly initialized units to the network while keeping all existing connections adaptive, can maintain high prediction accuracy without losing plasticity despite the continuous increase in the dead hidden unit proportion. Furthermore, we demonstrate that adaptive elastic networks, which in addition to progressively adding new hidden units also prune estimated dead hidden units at the beginning of each new task, can achieve excellent accuracy without loss of plasticity while simultaneously maintaining a near-constant, compact size. Our results suggest that growing and elastic networks, which exhibit the ability to adapt its structure to the relevant learning objectives, can be a promising class of algorithms also for preserving high plasticity in online continual learning.

- type: 
- query: 

---

## 4. `2608.01085v1`

**When Collaboration Becomes a Trigger: Collective Evidence-Threshold Backdoors in Multi-Agent Systems**

LLM-based multi-agent systems (MAS) extend LLM capabilities through iterative communication and shared contexts. However, this collaboration introduces a vulnerability: backdoor behavior can be activated when peer evidence reaches a hidden threshold, rather than being determined by any single message. We introduce a collective evidence-threshold backdoor paradigm for MAS and Boundary-Conditioned Backdoor Injection (BCBI), which constructs counterfactual boundary pairs to separate benign behavior before the threshold from the adversarial objective after it, and learns latent progression aligned with evidence. To mitigate this threat, we propose LAtent Transition Test-time Evaluation (LATTE), a clean-only latent-transition defense that learns benign communication dynamics and quarantines anomalous agent updates before their responses propagate. Across several benchmarks, BCBI yields selective activation with little premature activation; without knowing the attack target or trigger, LATTE limits propagation with minimal disruption.

- type: 
- query: 

---

## 5. `2608.02830v1`

**In-Context Collapse in Vision-Language Models and How to Mitigate it?**

Many-shot in-context learning (ICL) lets vision-language models (VLMs) adapt from image--label demonstrations without weight updates, and is widely assumed to improve as more demonstrations are supplied. We show the opposite: as demonstrations accumulate, a subset of VLMs undergo an \emph{in-context collapse}, a sharp, sometimes catastrophic accuracy drop spanning synthetic classification, natural-image classification, and VQA benchmarks, in some models falling below chance while outputs remain well-formed. Across an open VLM panel ($0.5$B--$11$B) and a frontier model (Claude Sonnet 4.5), the collapse is graded. Two capabilities turn out to be dissociable: robustness to accumulating demonstrations and the ability to learn a novel rule in context, their combinations yield three reproducible regimes. A parameter-matched lesion-and-rescue causally localizes the collapse to the vision-language integration pathway: an adapter on the connector and early/mid layers restores genuine learning (remap accuracy $0.39!\rightarrow!0.91$ at 16 shots), while an equal-capacity adapter on the late readout does not. We propose \textsc{CircA}, whose core is a one-time integration vaccine: trained once on one synthetic task, it transfers collapse-resistance to unseen task families (chance$\rightarrow$$0.71$/$0.60$ on CIFAR/Fashion). The layers best for in-context integration are not the layers best for weight-based consolidation, the late readout achieves higher accuracy and less forgetting at fewer parameters. The collapse is an integration failure at the vision--language interface, correctable by a lightweight, transferable intervention.

- type: 
- query: 

---

## 6. `2608.03322v1`

**LocAnyMed: Vision-Language Grounding for Multimodal Medical Images**

Medical visual grounding connects free-form clinical queries to spatial evidence in medical images and is an important component of interpretable medical artificial intelligence. However, general-purpose grounding models are predominantly trained on natural images, while existing medical localization resources remain fragmented across imaging modalities, datasets, and task formulations. To address this gap, we construct LocAnyMed-200K, a multimodal medical visual grounding dataset containing approximately 200K image-query-answer examples across computed tomography, optical medical imaging, ultrasound, and X-ray. We harmonize heterogeneous detection and localization resources into a unified free-form instruction format that supports one or multiple bounding boxes, point coordinates, and no-target outputs for negative queries. Full-parameter fine-tuning of LocateAnything-3B on LocAnyMed-200K improves F1@IoU 0.50 from 10.64 to 85.59 on a held-out evaluation split, demonstrating that large-scale domain-specific supervision can equip a general grounding model with effective medical localization capabilities. Beyond spatial coordinates, a clinically interpretable grounding system should also communicate the evidence supporting its prediction. We therefore derive LocAnyMed-CoT-20K, a rationale-augmented subset that connects anatomical context, visual observations, and spatial conclusions through structured reasoning and further improves cross-source generalization through fine-tuning. Together, these resources provide a unified foundation for studying both localization accuracy and rationale quality across heterogeneous medical imaging modalities. The code is publicly available at https://github.com/MiliLab/LocAnyMed.

- type: 
- query: 

---

## 7. `2608.01354v1`

**PixVL: Self-Supervised Training of Pixel-Level MLLMs via a Unified Mask--Text Consistency Cycle**

Recent studies develop pixel-level multimodal large language models (MLLMs) that support both Region Segmentation and Region Understanding, extending multimodal interaction from whole images to specific objects and regions. However, these methods face two fundamental challenges. First, the scarcity of high-quality mask--text pairs leaves abundant mask annotations without corresponding language supervision. Second, discrepancies in supervision formats and learning-signal densities induce optimization interference between Region Segmentation and Region Understanding. To address these challenges, we propose PixVL, a self-supervised post-training framework that introduces a unified Mask--Text Consistency Cycle, enabling pixel-level MLLMs to generate and self-verify regional descriptions and learn from unlabeled data. We found that direct cycle based solely on geometric reconstruction is unreliable because re-segmentation IoU does not faithfully reflect the semantic quality and referring sufficiency. PixVL therefore introduces confuser-aware semantic verification, which uses the model's confidence when it correctly chooses the target among highly similar candidate regions, and assigns zero reward to an incorrect choice. Meanwhile, PixVL performs cross-view verification using temporally separated video frames or geometrically transformed image views, preventing cyclic learning from collapsing to positional and shape shortcuts. Finally, a quality-coupled bidirectional learning strategy uses the highest-reward description to guide Text-to-Mask learning. This strategy transforms Region Understanding and Region Segmentation from competing tasks into mutual generators and verifiers. Experiments demonstrate that PixVL improves both region understanding task and segmentation task.

- type: 
- query: 

---

## 8. `2608.02168v1`

**Empowering Credit Risk Detection in Weixin Pay with Billion-Scale Deep Graph Learning**

Credit risk detection, particularly mitigating individual fraud, is crucial for maintaining the stability of digital financial ecosystems. Accurately identifying credit fraud among billions of users is critical for minimizing financial losses and safeguarding the sustainability of inclusive financial services. Given that credit fraud risks are often concealed within heterogeneous user-risk graphs, Graph Neural Networks (GNNs) have emerged as an effective tool for risk mining by capturing complex dependencies. To address the scalability bottleneck of industrial GNNs, distributed training based on subgraphs is indispensable. However, existing strategies often compromise topological integrity for load balancing. This can be catastrophic for risk detection, as it indiscriminately severs the long-tail evidence chains essential for risk propagation. Overlapping subgraphs can restore severed risk contexts but inevitably introduce redundancy and noise, while overlooking the representation alignment across different local subgraphs. In this paper, we propose a risk-aware overlapping subgraph learning framework for large-scale credit risk detection. We first construct base partitions to ensure load balance. Then, we perform budget-constrained sampling that selects informative long-tail nodes, thereby preserving critical risk diffusion patterns while filtering out noise. To mitigate representation inconsistency, we design a cross-subgraph consistency alignment mechanism. By enforcing alignment constraints on the overlapping nodes, we harmonize the local representations into a globally consistent latent space. Extensive experiments on Weixin Pay's production dataset demonstrate that our model significantly outperforms existing strategies for risk detection, offering a scalable and effective solution for industrial graph learning.

- type: 
- query: 

---

## 9. `2608.01724v1`

**TIDES: A Longitudinal Bilingual Dataset for Modeling Multi-Party Social Dynamics**

Group conversations are fundamental to human collaboration, yet standard large language models (LLMs) still struggle with the complexities of multi-party interaction. This challenge persists in part because existing group conversation datasets are often limited to short-term lab settings with contrived tasks, failing to capture the long-term social dynamics of real-world teams. To bridge this gap, we introduce TIDES, a high-resolution longitudinal dataset tracking 12 university project teams over a full semester. Comprising 75,971 utterances in both English and Korean from in-person meetings, TIDES provides a naturalistic record of teams working on self-managed projects. Our socio-structural annotations-covering interaction types, emergent roles, and development stages-allow for modeling of team evolution over months. Experiments show that fine-tuning on TIDES improves next-speaker prediction by 13.8 percentage points over a bigram baseline (64.53%) and yields performance comparable to strong proprietary zero-shot models. The model also comes within 2.1 percentage points of the published state of the art on the AMI Meeting Corpus while using approximately 42% less training data. However, human evaluations suggest that better next-speaker prediction does not necessarily yield more natural or coherent utterances, as fine-tuned models were generally less preferred than vanilla models. This potential mismatch motivates further study of how structural modeling can support natural multi-party generation.

- type: 
- query: 

---

## 10. `2608.01397v1`

**SG-WAM: Self-Guided World Modeling in Geometry-Aware Policy Space**

World Action Models (WAMs) couple action generation with prediction of future states. Their effectiveness depends on whether future dynamics are modeled in a space that is both aligned with action generation and sufficiently geometry-aware to capture where and how actions change the scene. Existing WAMs typically satisfy only part of this requirement, relying on either perceptually heavy observation-space targets or auxiliary latent spaces that are not jointly structured for action relevance and geometry. We propose SG-WAM, a self-guided framework that learns geometry-aware action-conditioned dynamics directly in the policy-derived representation space. SG-WAM introduces learnable dynamics tokens and a Self-Guided World Predictor that forecasts their future latent states conditioned on intervening robot actions. Prediction targets are generated by an exponential moving average copy of the same policy backbone, providing stable supervision within the representation family used by the action expert. Geometric supervision further structures the policy image-token representations, providing spatially grounded context for the dynamics tokens and yielding a future-alignment space that is both action-relevant and geometry-aware. Latent future prediction, geometric grounding, and flow-matching action generation are jointly optimized end-to-end in a unified framework. Built on a 0.9B model without large-scale embodied pretraining, SG-WAM achieves 98.5% average success on LIBERO and 73% on LIBERO-Plus, while outperforming strong baselines in both in-distribution and out-of-distribution real-world evaluations.

- type: 
- query: 

---

## 11. `2608.02062v1`

**A Comparative Analysis of MLP and Kolmogorov-Arnold Networks (KAN) for Faster-than-Nyquist (FTN) Signaling Detection**

Faster-than-Nyquist signaling improves spectral ef- ficiency by deliberately introducing inter-symbol interference. Classical sequence detectors such as BCJR can approach optimal performance, but their computational cost grows rapidly with channel memory. This paper investigates data-driven FTN BPSK detection under AWGN through a direct comparison between multilayer perceptrons and Kolmogorov Arnold Networks. A large-scale Monte Carlo dataset containing nearly four million labeled windows is generated for a time-packing factor of zero point eight and signal-to-noise ratio values from seven to ten decibels. The best MLP obtained from width sweeping uses hidden width thirty two, whereas the selected KAN uses hidden width four with spline grid size five. At ten decibels, the MLP produces a bit error rate of one point three times ten to the minus four, while the KAN reaches seven times ten to the minus six. This corresponds to an eighteen point six times lower bit error rate while using only one eighth of the MLP hidden width. The results show that KAN provides a more effective and more parameter-efficient neural decision model than the MLP baseline for FTN BPSK detection.

- type: 
- query: 

---

## 12. `2608.01676v1`

**Understanding Sparse Attention Selectivity in Long-Context Foundation Models via Counterfactual Evaluation**

Sparse attention is widely deployed in long-context serving stacks, yet no framework audits how discarding blocks changes the influence of specific content on model output. We first establish that the phenomenon is real and causal: Block Sparse Flash Attention (BSFA) route replay across four architectures changes output decisions in 13 of 16 cells, with zero identity-replay label flips. We then introduce a dense-calibrated counterfactual audit using matched probe cards---Gold (carrying the correct answer label), Poison (carrying a target wrong label), and Benign (filler only)---under six-layout position symmetry, isolating the sparsification-specific effect. Two patterns compete. Signal concentration: the selector preserves Gold and Poison blocks far above filler-matched Benign blocks (G$\approx$P$\gg$B across all model--task pairs). Integration loss: discarding blocks severs cross-block attention---confirmed by an ablation where isolating the probe block collapses its influence from 4.48 logits to zero. Compression ratio governs the balance: a full sweep from mild ($c=0.25$) to aggressive ($c=0.75$) compression across four model--task pairs reveals that three of four cells move toward stronger sparse amplification at higher compression, with two exhibiting sign reversals. Three independent arms---BSFA route replay, controlled block-top-$k$, and KV-cache eviction---converge: sparsification changes content influence in ways aggregate accuracy cannot detect. We provide an open measurement framework deployable on any model exposing block identities.

- type: 
- query: 

---

## 13. `2608.01017v1`

**Why LLMs Give In: Conversational Factors and Reasoning Behind Medical Sycophancy**

A language model that abandons a correct medical answer under user pushback is more dangerous than one that was simply wrong, because it lends the credibility of a correct answer to the user's misinformation. Such model behavior, described as medical sycophancy, is usually reported as a single rate per model, but we find it is a property of the conversation, not the model. We study medical sycophancy in language models with a fully crossed factorial design over four conversational factors, user role, the evidence behind a false claim, whether the challenge precedes or follows the model's answer, and whether the correct answer is grounded in the prompt, across five open-weight models and 500 MedQuAD questions (1.2M trials). The factors interact sharply: fabricated sources raise sycophancy 2.0x when they accompany the question but halve it once the model has answered, so the same evidence helps or hurts depending only on timing. Sycophancy varies far more across questions than across models (67x vs. 3x), so a single rate reflects the conversation and the questions sampled as much as the model. Chain-of-thought traces explain why. Models that re-examine their own prior answer concede, while those that reason about the medical facts hold, and only a model that has already answered can spend a round auditing the fabricated source.

- type: 
- query: 

---

## 14. `2608.02250v1`

**Assessing the Impacts of Imperfect Datasets on Client Selections in Federated Learning**

Federated learning (FL) is a popular distributed learning framework where multiple clients perform local training and a server aggregates the locally updated models. FL enables decentralized training while preserving the privacy of clients' datasets. However, non-independent and identically distributed (non-IID) or noisy datasets can lead to low model accuracy or high convergence latency. Precluding these clients through client selection may mitigate the problem, but heavily biased client selections may also degrade the learning performance. In this study, we first experimentally measure the impact of non-IID data (including skews in data quantity and label distribution), noisy data, and fairness in client selection on model accuracy and convergence. We then propose a privacy-preserving scoring method to assess each client's contribution in FL, with experiments conducted to demonstrate the effectiveness of the proposed assessment.

- type: 
- query: 

---

## 15. `2608.03106v1`

**FaithIR: Rethinking Infrared Image Super-Resolution from Perceptual Sharpness to Task Relevant Fidelity**

Infrared image super-resolution (IISR) is important for downstream tasks such as object detection and semantic segmentation. Existing IISR methods often produce artificial textures, over-sharpened edges, and spurious high-frequency details that distort authentic thermal structures and semantic information. To address this issue, we propose FaithIR, a faithful infrared super-resolution framework for reliable machine perception. FaithIR consists of a patch-level conditioning branch that captures global thermal and structural information and a pixel-level restoration branch that performs dense local reconstruction under structural guidance. The entire restoration process is performed directly in the pixel domain to preserve infrared-specific structures and task-relevant information. Extensive experiments on FLIR-IISR, M3FD, and FMB demonstrate strong reconstruction fidelity, cross-dataset generalization, and superior performance in object detection and semantic segmentation. These results show that demonstrate that preserving faithful infrared structure preservations is more important for reliable machine perception than merely pursuing perceptual sharpness alone.

- type: 
- query: 

---

## 16. `2608.02302v1`

**Trajectories That Segment Themselves: Agent-Declared Boundaries as a Training Unit**

Long-horizon coding-agent trajectories are poorly matched to the credit units available to train on: a single action has no stable value, an episode label merges productive exploration with abandoned directions, and a fixed window cuts where the logging mechanics fall. We introduce collection-time semantic self-segmentation, in which a declarative contract has the acting agent expose its own boundaries while the trajectory is generated. Instantiated with falsifiable causal hypotheses, successive adoptions expose variable-length semantic phases, and no milestone vocabulary, gold patch, environment replay, teacher logits, or retrospective segmenter places a boundary. Because the agent names its conjecture, a reviewer can negate it by name, which lets our protocol manufacture wrong-cause-then-correction transitions that recorded work rarely contains; one collection then yields four supervised targets, including audit supervision from exactly the failed regions an episode label discards. We then ask what survives deleting the declaration. Given the cut points but not the hypothesis, a model attributes action blocks to their governing hypothesis at over twice chance, beating equal-length blocks over the same trajectories (paired sign test $p = 0.0002$), surviving a lexical control and collapsing under label permutation. Asked instead to place boundaries, a code-blind annotator matches 24 of 40 where random placement matches 11.5, while a mechanical test-event rule beats chance at neither end of a strict-to-permissive sweep. The segments are therefore coherent and not cheaply reproducible. Downstream, DPO on 2,551 phase-boundary pairs changes no decision on 91 adversarial held-out items, while four of 60 change on matched-construction items, all wrong to right, where two controls change none: with 1,825 pairs from one generator, the variable to vary next is corpus diversity, not the boundary.

- type: 
- query: 

---

## 17. `2608.03612v1`

**Predictive Enhancement Calibration for Latent Breast MRI Virtual Contrast Enhancement**

Virtual contrast enhancement (VCE) synthesizes enhanced breast MR images from pre-contrast acquisitions. Modern latent generators offer strong image priors, but their bounded natural-image autoencoders conflict with the non-canonical intensity scale of MRI. We show that the upper bound can alter radiomic fidelity before generation, while scaling source and target independently creates a coordinate inconsistency. We propose Predictive Enhancement Calibration (PEC), which represents each pair in a shared, case-adaptive coordinate during training and predicts its unavailable upper endpoint from the pre-contrast image at inference. We integrate PEC with a pretrained FLUX latent flow transformer via parameter-efficient reference conditioning. Target round trips first isolate representation loss before generation; near-matched conditional models then compare PEC with fixed-wide and separate coordinates under comparable training budgets and backbone settings. On the fixed internal MAMA100 development cohort, PEC improves all eight point estimates in this source-only VCE setting, with paired evidence strongest for MSE and LPIPS.\noindent\textbf{Code:} https://github.com/tanlei0/pec-breast-mri-vce

- type: 
- query: 

---

## 18. `2608.00985v1`

**Beyond Gene Reconstruction: Learning Cell Representations through Complementary Transcriptomic Views**

The rapid growth of single-cell transcriptomic data has enabled the development of foundation models pretrained primarily by reconstructing masked expression values. This objective encourages these models to learn gene dependencies but does not directly optimize whole-cell representations, which are essential for many downstream tasks. To bridge this gap, we propose a contrastive pretraining framework that learns cell representations through complementary transcriptomic views. Since standard contrastive learning is not readily applicable to single-cell pretraining, we introduce specific adaptations along three dimensions --- co-expression-guided gene partitioning, expression-aware contrast-set construction, and competence-gated contrastive onset. Specifically, we first construct two complementary views of each cell by partitioning its genes according to their co-expression structure. Then, to prevent the model from using gene-set identity as a shortcut, we construct hard negatives by permuting expression values while keeping gene identities unchanged. Finally, we introduce a competence-aware controller to determine how the contrastive objective is applied. Experiments on cell-type annotation and gene regulatory network inference demonstrate competitive transfer under the evaluated protocols. In the six-network GRN evaluation, our method records the highest mean AUROC and AUPRC point estimates among the compared variants, while the highest-scoring variant differs across individual networks. These results establish complementary-view contrastive learning as an effective direction for single-cell pretraining beyond gene reconstruction.

- type: 
- query: 

---

## 19. `2608.02964v1`

**Material-Segmented Per-Pixel Emissivity Correction for Thermographic Anomaly Detection in Cultural Heritage Digital Twins**

Quantitative longwave thermography of heritage surfaces is limited by the global-constant emissivity assumption in inverse-Planck temperature retrieval; on heterogeneous surfaces emissivity varies within one field of view, producing apparent-temperature artifacts that mimic and mask subsurface anomalies. We present a training-free pipeline that derives per-pixel emissivity by applying SAM 3.1 open-vocabulary segmentation to a colocated, co-calibrated RGB channel, mapping segments to a material-keyed LWIR emissivity table compiled from primary measurement literature, and propagating the field into a per-pixel inverse-Planck solve on raw radiometric data. Lacking any public dataset with raw radiometry, a temperature reference, and a colocated RGB camera, we evaluate on a physics-based synthetic benchmark and four real datasets. On the benchmark, under a palette spanning the low-emissivity exceptions, the correction cuts mean absolute error from 1.97 K to 0.91 K at 20 K contrast and, with an accurate table, beats the best fitted global constant on every layout; on a heritage-realistic emissivity distribution it does not. We contribute a quantified operating-regime map, and a measurement-backed finding that tempers the heritage claim: weathered outdoor heritage emissivities cluster near the conventional default, so the correction is small on typical surfaces and concentrated on genuine low-emissivity exceptions. We characterize the dominant failure mode, in which open-vocabulary segmentation matches appearance rather than material, and the contraindicated regime in which emissivity-defined anomalies are suppressed.

- type: 
- query: 

---

## 20. `2608.01288v1`

**TurboClear: One-Step Object-Effect Removal via Region-Calibrated Distribution Matching and Fusion**

Recently, diffusion-based removal methods have achieved promising visual quality in removing both target objects and their associated effects. However, they typically rely on multi-step denoising, leading to high inference cost. Directly applying existing one-step distillation methods is also suboptimal, since their global objectives lack explicit region-wise calibration and may weaken the asymmetric edit-and-preserve behavior required by object-effect removal. To address these challenges, we propose TurboClear, a one-step SDXL-based object-effect removal model. During training, we design Region-Calibrated Distribution Matching (RDM) for region-aware distillation to preserve the teacher model's asymmetric edit-and-preserve behavior. Furthermore, we propose Learnable Spatial Fusion (LSF) for lightweight inference-time fusion. Extensive experiments show that TurboClear significantly improves inference efficiency while maintaining competitive visual quality. TurboClear reduces the computational overhead by up to $40.04\times$ compared to ObjectClear, and by up to $665\times$ against the Flux-based method OmniPaint, all while maintaining comparable or better visual removal quality. Code is available at https://github.com/GuoCalix/TurboClear.

- type: 
- query: 

---

## 21. `2608.02820v1`

**Evading Chain-of-Thought Monitoring Through Model Poisoning**

Chain-of-thought (CoT) monitoring is an increasingly important component of AI safety stacks but relies on the assumption that a model's reasoning trace is informative about its actions. This work studies the limits of CoT monitoring through the lens of model poisoning. We demonstrate that backdoors can be implanted into reasoning models to elicit an attacker-chosen behavior while their CoT traces appear entirely benign. We find that these CoT-Hidden backdoors can be induced through simple fine-tuning recipes across reasoning-model architectures and sizes. When direct poisoning is ineffective, we introduce a curriculum training approach that progressively teaches the model to produce an attacker-chosen output while concealing the behavior from its reasoning traces. These findings suggest that CoT monitoring may be better framed as a question about the consistency between a model's reasoning trace and its final response than as anomaly detection within a trace. We further examine the mechanisms that allow models to suppress evidence of the target behavior from their reasoning traces. Causal interventions locate a trigger-conditioned activation pathway that does not depend on the visible reasoning, and residual stream verbalizations provide an anomaly warning near answer generation, but do not identify the trigger, target, or backdoor mechanism.

- type: 
- query: 

---

## 22. `2608.03927v1`

**A Physics-Flavored Transformer Network for Parametrizing Contraction Dynamics of Engineered Skeletal Muscle Tissues**

Engineered Skeletal Muscle Tissues (ESMs) have become a key structure for biomedical disease modeling and pharmacological screening, yet their functional characterization often relies on simplistic metrics like peak force, discarding critical kinetic information. This is partially due to the high level of mathematical complexity which mechanistic models introduce to capture these dynamics. Hence, exactly the complexity prevents scalable application and widespread adaptation in the field. Here we present a Physics-Flavored Neural Network (PFNN) that automates the kinetic phenotyping of ESMs. Our architecture integrates a stretched-exponential physical model into a CNN-Transformer, enabling the extraction of physically meaningful parameters directly from force-time profiles. To address the scarcity of labeled biological data, we employ a hybrid training paradigm: the model develops a "physical intuition" on synthetic data before undergoing unsupervised self-alignment on unlabeled real-world measurements. Our results demonstrate that this physics-flavored approach achieves high-fidelity parameterization across diverse contractile phenotypes and cell lines, including Duchenne Muscular Dystrophy models. Our scalable, self-improving pipeline bridges the gap between idealized biophysics and noisy \emph{in vitro} data, providing a robust tool for high-throughput biophysical research.

- type: 
- query: 

---

## 23. `2608.01334v1`

**SphereVideo: Prototype-anchored Hyperspherical Boundary for Continual AI-generated Video Detection**

AI-generated video (AIGV) detection aims to distinguish real videos from AI-generated ones. In practice, detectors trained on existing data often fail to generalize to newly emerging generative models, making this task challenging. Therefore, continual learning (CL) is essential for improving the adaptability. However, CL frameworks for this task remain underexplored. To this end, we propose SphereVideo, a novel CL framework for AIGV detection built on two key observations. First, real videos exhibit a compact feature distribution. Based on this, we encourage real video features to cluster around a real prototype on a hypersphere while repelling AI-generated samples, thereby establishing a decision boundary. This prototype serves as a stable anchor for CL, regulating boundary evolution and mitigating catastrophic forgetting. Second, existing methods tend to rely solely on spatial artifacts as shortcuts. To enhance temporal modeling, we introduce a strategy that models the temporal dynamics of real data at both frame and clip levels. By strengthening real data modeling, this strategy further facilitates learning a real prototype and forming a stable decision boundary. Moreover, we construct a comprehensive and challenging benchmark. Extensive experiments demonstrate that SphereVideo achieves an improved plasticity-stability trade-off, outperforming prior methods by 3.08% on seen data and 4.00% on unseen AI-generated data.

- type: 
- query: 

---

## 24. `2608.02930v1`

**Hypercubes, Hyperplanes, and Constraint-Induced Complexity Collapse in Atomic Concept Learning**

We revisit higher-arity atomic concept learning through the geometry of hypercubes and hyperplanes of ground instances. Our starting point is the observation that the ambient r-dimensional hypercube of ground atoms is not structurally uniform. Its logical complexity is organized by hyperplanes: every hyperplane other than the full diagonal collapses into finitely many elementary-equivalence classes, with a bound independent of the term depth, while the full diagonal is exceptional and its class count grows without bound. This asymmetry is not merely geometric. It reflects the reduction-theoretic structure of the concepts themselves. Building on a higher-dimensional framework developed in the author's earlier work, we reinterpret these results through canonical simple concepts, minimal orderings, and representative reductions. This yields a taxonomy of hyperplane behavior in higher dimensions and shows that complexity is localized rather than spread uniformly through the instance space. The paper includes a fully worked binary case, an explicit treatment of the ternary hypercube, and an unpacked account of the reduction machinery that drives the collapse. The three-dimensional case already exhibits the essential phenomenon of orthogonal families, partial diagonals, and the exceptional full diagonal. This geometric-logical perspective clarifies where complexity is concentrated in atomic concept learning and suggests a modern interpretation in terms of constrained hypothesis spaces and structured classification.

- type: 
- query: 

---

## 25. `2608.01958v1`

**FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis**

4D Gaussian Splatting (4DGS) excels in dynamic 3D reconstruction and real-time novel view synthesis via efficient 4D Gaussian representations and parallelizable rendering. However, existing 4DGS approaches rely on a single polynomial to model motion, which limits performance in complex dynamic scenes where high-frequency motion components are prevalent, and fails to ensure long-term stability due to cumulative trajectory drift. To address these issues, we propose a Fourier Motion Modeling module: this paradigm decomposes motion into frequency-based sinusoidal components, capturing both low-frequency global trajectories and high-frequency local details to model complex motion patterns accurately. It retains the real-time rendering capability of 4DGS while improving complex motion fitting and long-term coherence. Additionally, we integrate a motion-aware regularization strategy into the loss function: it uses frequency-dependent weights to suppress high-frequency jitter while preserving low-frequency motion coherence. Extensive experiments on N3V and Google Immersive datasets from multiple scenarios demonstrate the effectiveness of our method.

- type: 
- query: 

---

## 26. `2608.00573v1`

**TrimMoE A communication aware and adaptive depth framework for distributed edge inference**

Serving Mixture-of-Experts (MoE) large language models across distributed edge servers is bottlenecked by the cross-server expert transmission. The existing approaches mainly focus on how to reach a remote expert faster. However, in this paper, we instead consider whether a given layer, and the layers after it, need to be executed at all. To this end, a communication-aware adaptive-depth framework is proposed in this paper, termed TrimMoE, which couples layer skipping and confidence-based early exit with substitute execution and server-expert selection under a unified quality budget. Specifically, in the offline stage, TrimMoE freezes the backbone, trains the lightweight per-layer exit heads, calibrates the per-layer importance thresholds, and allocates the expert replicas by a skip/exit-aware redundancy benefit. In the online stage, a transition-aware look-ahead anticipates the token movement, so that the depth reduction targets the costliest transmissions, and besides, two feedback rules adapt the delay-quality weights and the exit threshold. Moreover, we prove that the substitution-and-skipping proxy degradation never exceeds the configured budget, and that the early exit is admitted only under a calibrated confidence gate. On a heterogeneous 10-server testbed with Switch-Base-8E, Qwen-MoE-A2.7B, and Mixtral-8x7B, TrimMoE reduces the average latency by up to 62.8%, lowers the cross-server traffic and the remote-execution ratio, and sustains high throughput under load, while keeping the task-quality degradation within a 2% bound.

- type: 
- query: 

---

## 27. `2608.03291v1`

**The Tell-Tale Trace: Detecting Reasoning Failures in LLMs Using Chain-of-Thought Dynamics**

Chain-of-thought (CoT) reasoning improves large language model (LLM) performance while also providing an observable interface to the model's reasoning process. Existing approaches that leverage verbalized CoTs to monitor reasoning correctness, however, largely evaluate the semantic correctness or consistency of individual intermediate steps, rather than how the reasoning process evolves across the trace. As a result, failures distributed across the reasoning trajectory, rather than those localized to a single incorrect step, remain comparatively underexplored. Furthermore, verbalized CoTs need not faithfully reflect the model's internal reasoning, motivating analyses that do not treat individual statements as literal accounts of internal computation. In this work, we therefore ask whether the dynamics of visible CoT can be leveraged to systematically distinguish successful from failed reasoning without assuming such semantic faithfulness. We study a range of LLMs on verifiable Boolean satisfiability tasks with variable complexity, enabling controlled comparisons near each model's capability frontier. Tagging CoT sentences by reasoning function reveals premature verification collapse on SAT problems: incorrect traces enter clause checking earlier, repeat similar operations, and finalize sooner. On UNSAT problems, models presumptuously move towards incorrect SAT conclusions, checking candidate assignments rather than deriving contradictions across constructed cases. Subsequently, a targeted proof-search prompt intervention raises Llama3-70B accuracy from 13.3% to 85%, correcting 84.6% of these errors. These results show that capability failures can manifest as distributed, task-dependent changes in the structure of visible reasoning, and that CoT dynamics agnostic to whether the verbalized trace reflects the model's internal computations can help diagnose and correct failures.

- type: 
- query: 

---

## 28. `2608.01488v1`

**Towards Compact Unified Multimodal Tracking: Synergizing Knowledge Distillation with Structural Pruning**

Unified multimodal object tracking has achieved remarkable robustness by leveraging complementary sensor data (e.g., RGB, Thermal, Depth), yet the heavy computational burden of state-of-the-art models hinders their deployment on resource-constrained edge devices. In this work, we identify the prediction head as a critical but often overlooked efficiency bottleneck. By strategically streamlining the decoder architecture, we unlock the potential for real-time inference but simultaneously introduce a capacity gap between the lightweight student and the heavy teacher. To resolve this, we conduct a systematic analysis of 17 distillation strategies and introduce a Dual-Alignment Distillation framework. Our key insight is that effective compression requires decoupling knowledge transfer into two complementary streams: (1) Spatial Representation Alignment, which employs feature distillation to sharpen the student's spatial focus on foreground targets ("Where to track"); and (2) Semantic Distribution Alignment, which utilizes logit-based distillation to align decision boundaries and transfer discriminative dark knowledge ("What to track"). Extensive experiments across five benchmarks demonstrate that our approach significantly outperforms complex state-of-the-art methods. Notably, our distilled model achieves 91.5% MPR on RGBT234 and operates at 54 FPS on a single RTX 4090, representing a 5x speedup over the teacher model while maintaining superior accuracy.

- type: 
- query: 

---

## 29. `2608.01930v1`

**Recompute or Reuse? Diagnosing and Mitigating Textual Shortcuts in VLM Self-Reflection**

Vision-language models (VLMs) are expected to revise their reasoning when visual evidence changes. Failures to do so are often attributed to insufficient visual attention or contextual inertia, leaving unclear what models reuse instead of recomputing from the current image. We show that evidence-bearing reasoning in a prior chain of thought (CoT) can form a textual shortcut that competes behaviorally with visual recomputation. Across 16 VLMs, a matched counterfactual analysis identifies evidence-bearing content as the most robust carrier of prior-CoT influence. Removing this evidence-bearing content shifts answer preference more than removing length-matched non-evidence context or the final-answer span, with prior control weakening progressively as more stale evidence is removed. Reordering this evidence also weakens prior control, showing that its organization modulates shortcut strength. Beyond the immediate answer, the shortcut can retain residual influence after answer correction: weakening current-image support shifts preference back toward the prior answer, while repeated prior answers and reused premises arise mainly when the shortcut remains active. To limit this influence, we introduce Fresh-State Attention Firewall (FSAF), a training-free intervention that isolates fresh computation from the prior CoT. Across five VLMs, FSAF raises visual update rate from 35.28% to 53.61% and reduces prior-answer rate from 39.22% to 3.67%. Reliable VLM self-reflection therefore requires more than looking again: fresh visual recomputation must be protected from stale textual reuse.

- type: 
- query: 

---

## 30. `2608.03796v1`

**Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss**

Small language models are often the only option for deployment under tight latency, cost, and on-premises constraints, but they are rarely trained from scratch: a compressed model is usually recovered through knowledge distillation (KD). This recovery step largely decides the final quality, yet it is expensive. We present a practitioner's study of how to make distillation training efficient, organised around two systems contributions. First, we show that offline KD (caching the teacher's top-$K$ logits once and training the student against the cache) matches online distillation at near-identical training loss while removing the teacher from memory, running about 29\% faster per iteration, and reaching up to 41\% higher throughput on a single H200 GPU. Second, we introduce a \emph{fused, chunked KL loss} that never materialises the full vocabulary-sized logit tensor, making peak memory linear in the sequence length. This removes the memory spike that otherwise caps context length and lets us train at four times the context (32{,}768 tokens) on a single GPU. A separate output-head-only toy benchmark isolates the loss kernel and confirms its memory and iteration-rate scaling from 4K to 256K tokens. Together these make large-scale healing and hundreds of ablations affordable. We also report supporting ablations on loss design and sequence packing. We release our chunked-loss implementation: https://github.com/CompactifAI/Full-Chunked-KL-Loss.

- type: 
- query: 

---

## 31. `2608.01495v1`

**Probing the 3D Object-Level Understanding of Pre-Trained Detection Transformers**

Detection transformer models, including DETR and its extensions, learn to output a set of object-level embeddings that can be simultaneously decoded into 2D bounding boxes and class distributions. In this paper, we investigate what pre-trained 2D detection transformers understand about the 3D properties of objects. Specifically, we investigate the extent to which properties including the depth of objects from the camera and the 3D location of objects relative to the camera can be recovered from object-level embeddings using linear and non-linear probes. Across a range of detection transformer models, our results show a surprisingly strong and previously unknown ability of 2D DETR models to represent useful information about the 3D properties of objects, despite the complete lack of 3D supervision during model pre-training.

- type: 
- query: 

---

## 32. `2608.00726v1`

**Foveated Probes Recover Localized Binding Information in Vision Foundation Models**

Frozen vision foundation models are commonly evaluated through a single global image embedding, but this interface can conflate missing information with information lost at readout time. We study this distinction by keeping a pretrained vision encoder frozen and varying only the readout applied to its final patch tokens. We compare standard global readouts against a lightweight foveated readout, which attention-pools patch tokens using a learned or question-conditioned query, and against an oracle readout with access to the annotated target region. We evaluate these interfaces on three localized binding problems: a controlled synthetic color--shape binding task under clutter, a color-free crowded shape-detection variant, and a GQA-derived natural-image task where paired questions ask for the colors of different same-category objects in the same image. Global readouts perform near perfectly when the synthetic target appears alone, but collapse under clutter and counterfactual target edits, whereas the foveated readout recovers most of the oracle-accessible signal. On the GQA-derived task, question-independent global image vectors improve only modestly over question-only priors, while question-conditioned foveation substantially improves paired localized color accuracy. A counterfactual nuisance-to-signal ratio explains the synthetic failures: global pooling dilutes localized label-changing evidence while exposing the probe to nuisance variation from irrelevant objects. These results indicate that apparent spatial blindness in frozen vision models can arise from the global embedding interface rather than from an absence of spatial information in the frozen patch tokens.

- type: 
- query: 

---

## 33. `2608.03887v1`

**Omega-S: A Functional Resilience Index for LLM Fine-Tuning**

Fine-tuning a large language model on new data degrades what it previously learned. We present Omega-S, a drop-in penalty computed from the weight matrix alone: it needs no previous-task data, no Fisher matrix and no stored copy of the old weights. It is three lines in an existing training loop and adds under 4% to the cost of a step. Retention. On Llama-3-8B with LoRA, fine-tuned from code to prose and measured by HumanEval over ten seeds, Omega-S retains more of the original capability than no regularisation on 9 of 10 seeds (0.173 -> 0.238 absolute pass@1; sign test one-sided p=0.011, Wilcoxon p=0.006), as a retention ratio, 62.9% -> 84.1%. It also beats tuned weight decay on 10 of 10 seeds (p=0.002) and tuned EWC on 8 of 10 (p=0.014), every arm re-measured in the same session. Mechanism, measured rather than asserted. Omega-S is topological by construction, its objective built from Tr(A^3), but we measured which of its four factors actually moves and three do not: their elasticity with respect to the weights is at or below 1e-4, against 9e-3 for the degree-variance term. As implemented, the composite reduces to a penalty on the variance of node degrees, which means row magnitude in square modules and directional alignment in non-square ones. We report this because a method whose name promises one thing and whose gradient does another should say so. We also enumerate the open design choices, including a contrast-preserving construction that does what it was designed to do and makes retention worse on all ten seeds. Repeating an identical configuration, same seed and same hardware, gives a standard deviation of 0.104 in retention ratio. We have not found this quantified for low-rank fine-tuning of language models, and it bounds every seed-paired comparison in this literature, ours included. Code, per-seed results and the full record of negative results are available.

- type: 
- query: 

---

## 34. `2608.03160v1`

**Caved or Convinced: Temporal Sampling Gates Claim Deference in Video Large Language Models**

When asked which of two events came first, video large language models can fail in two opposite ways: cave to a false claim, or reject a true one. Prior video sycophancy work measures only the first and mitigates it by teaching the model to trust the user less, a fix known in text and image models to worsen the second. In video, both failures come from two causes the literature treats as one: availability, whether the sparse sampled frames contain the two events, and weighting, whether that evidence is trusted over the user. We separate them with two interventions that keep the claim fixed: a frame-preserving reorder that flips the claim's truth, and a sampling-offset shift that captures or misses both events at a fixed frame budget. When the events are missed, the two twins present identical frames, so each of the nine models we evaluate accepts a true and a false claim at the same rate, making Youden's $J=0$ by construction. Availability is necessary but not sufficient. Five of the nine read the order, yet four of those five still cave to the false claim, so their deference hits a weighting ceiling. Since trust cannot be calibrated over evidence that was never sampled, we propose a reversal test that cancels the model's order prior by scoring the sampled frames forward and reversed, then answers, resamples, or abstains without reading the claim. The test raises the order accuracy to 0.92-1.00 on the models that read the order and abstains rather than guesses on those that cannot.

- type: 
- query: 

---

## 35. `2608.03142v1`

**Minimax-Optimal Semiparametric Contextual Dynamic Pricing with Multimodal Revenue**

We study contextual dynamic pricing with arbitrary covariate sequences and bounded, possibly nonbinary purchase quantities. Demand follows a semiparametric surplus-index model with an unknown linear valuation parameter and an unknown Hölder-smooth response. We impose neither concavity nor strong unimodality on revenue and allow nonunique optimal prices. We develop a pilot-corrected layered decision-partitioning policy that combines directional pilot estimation, local polynomial learning, predictable data assignment, and global action elimination. Pilot correction removes the first-order effect of valuation-parameter error, while permanent labels enable concentration under adaptive sampling. The policy attains the minimax smoothness-dependent horizon rate up to logarithmic factors; a matching lower bound already holds for a constant-context binary-demand subclass.

- type: 
- query: 

---

## 36. `2608.00950v1`

**Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction**

We propose Swimm3R, a unified framework that combines medium-aware structure-from-motion (SfM) with Underwater Beta Splatting to address scattering- and attenuation-induced failures in underwater 3D reconstruction. Swimm3R distills in-air geometric priors into a feed-forward backbone and uses a physics head to regress underwater image-formation parameters, camera poses, and restored point clouds. Additionally, we introduce Underwater Beta Splatting, which extends Gaussian splatting with Beta primitives and scattering-aware geometric gradients for stable underwater geometry representation. We further establish the Barbados underwater video dataset to demonstrate the effectiveness of our method in challenging underwater environments. On this dataset, Swimm3R robustly recovers underwater scene structure under challenging scattering conditions, yielding coherent seafloor geometry. Using these predicted point clouds, the proposed Underwater Beta Splatting improves average PSNR by $1.47$ dB over WaterSplatting while increasing downstream localization performance by $2.0$ and $2.4$ percentage points in RRA@15 and RTA@15, respectively.

- type: 
- query: 

---

## 37. `2608.01457v1`

**Clear-Weighted Bit Allocation for Satellite Downlinks**

Earth-observation satellites capture more imagery than intermittent ground contacts can transmit. Onboard systems threshold a cloud detector, discard frames or tiles, and compress the survivors with a fixed codec. On expert-labeled imagery, these rules remove more than one-fifth of clear pixels, primarily through detector false positives. We train a neural codec with a clear-probability-weighted reconstruction loss, reallocating coded bytes from clouds to clear ground without requiring or transmitting a cloud map onboard. Each capture is encoded into a resumable base layer and a dependent refinement layer, while clear content is estimated from features produced by the encoder. At each contact, we causally rank arrived layers using estimated clear content, unfinished bytes, deadline slack, and aggregate deadline pressure. The scheduler serves base and computational deadlines, bounds stored residual bytes, and resumes interrupted packets. We evaluate the onboard-to-downlink pipeline using real entropy-coded bytes, orbit-derived interruptible contact capacities, and measured service time and energy on resource-constrained embedded accelerators. Clear-weighted codecs require up to 47.8\% fewer bytes than learned-compression baselines at matched clear-region quality. The optimized encoder consumes less time and energy than one pass of the cloud detector used by the frame-discard rules. Relative to fixed two-stage service on the same streams, our scheduler more than doubles deadline-full clear-content delivery for the interrupted combined cohort, reaches 83.6\% of a certified clairvoyant upper bound, and exceeds replayed reference orders in deadline-usable delivery.

- type: 
- query: 

---

## 38. `2608.03579v1`

**Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training for Efficient Ultra-LoRA Serving**

Modern multi-tenant Low-Rank Adapters (LoRAs) serving systems concurrently host tens to hundreds of LoRA adapters. Though powerful, this introduces a critical system dilemma between serving efficiency and task performance: higher-rank adapters generally achieve better downstream task performance, but their GPU VRAM footprint and Host-to-Device PCIe swapping overhead severely constrain scalability. Conversely, ultra-low-rank adapters ($r \le 2$) minimize both VRAM footprint and PCIe transfer overhead, but suffer from downstream task performance degradation. To solve this problem, we propose Subspace-Aligned LoRA Training (SALT), a serving efficiency-aware hierarchical fine-tuning framework. Our solution operates in three phases. First, a provider jointly trains high-capacity domain centroids on public data within the domain using a novel alignment regularizer that coheres in-domain task subspaces into a unified basis. Next, users fine-tune ultra-low-rank task residual adapters on private data atop those frozen centroids. Finally, during inference, the provider pins the centroid in GPU VRAM and dynamically swaps in each user's task residual on demand. Across LLMs of varying scales, SALT recovers high-rank accuracy using $r \le 2$ residuals, achieving up to 18.5% absolute accuracy gains over state-of-the-art compression baselines and reducing per-adapter memory by up to 16x. When integrated into vLLM, SALT improves serving throughput by up to 51% under PCIe bandwidth pressure and 28% under GPU VRAM constraints for Llama-3.2-3B.

- type: 
- query: 

---

## 39. `2608.03247v1`

**CIGTSurv: Clinical Information Guided Tri-modal Survival Prediction with Local Prototype Association and Global Feature Alignment**

Multimodal learning has significantly advanced survival prediction by integrating pathology images with genomic data. However, clinical information, despite its critical role in reflecting a patient' s overall health, remains underutilized due to its discrete, sparse, and low-dimensional nature. Furthermore, the inherent heterogeneity across these modalities pose significant challenges in modeling cross-modal interactions. In this paper, we propose CIGTSurv, a Clinical Information Guided Tri-modal framework for Survival prediction. Specifically, we first design a holistic text template and use pretrained foundation models to transform clinical tabular data into high-dimensional tokenized embeddings. Using clinical information as an anchor, we then introduce a dual-level interaction mechanism: 1) a local prototype association (LPA) module based on cross-attention to explicitly learn token-level correspondences between different modalities, and 2) a global feature alignment (GFA) loss based on Maximum Mean Discrepancy (MMD) to implicitly enhance cross-modal distribution consistency. Extensive experiments on five TCGA cancer cohorts demonstrate that CIGTSurv achieves state-of-the-art (SOTA) survival prediction performance. Our source code is publicly available at https://github.com/Daijing-ai/CIGT-Surv.git.

- type: 
- query: 

---

## 40. `2608.03562v1`

**Robust General Utility for Reinforcement Learning**

Reinforcement learning (RL) with general utility extends classic RL by optimizing an arbitrary utility functional of the policy-induced occupancy measure, thereby enabling a broader range of applications. However, previous work on general utility RL typically assumes the evaluation utility is fixed and correctly specified. In practice, the utility used at deployment can deviate from the training one, creating a robustness gap that prior work does not address. Motivated by this, we propose robust general-utility RL, a minimax learning framework that trains policies against utility misspecification within a prescribed uncertainty set. Our framework strictly generalizes standard general-utility RL while also providing a unified view of many existing RL frameworks, including reward-robust RL and constrained RL, through appropriate choices of the utility uncertainty set. We further develop provably convergent stochastic algorithms for two regimes. For concave utilities, we develop a projected stochastic gradient descent-ascent method and establish stationarity guarantees. For the more challenging nonconcave regime, we propose a stochastic prox-extragradient algorithm that mitigates ill-posed behavior induced by nonconcavity, with convergence guarantees to approximate first-order stationarity. Experiments on LLM safety alignment and exploration maximization tasks further corroborate the convergence behavior consistent with our theory.

- type: 
- query: 

---

## 41. `2608.03284v1`

**Test-Time Scaling for Safe Text-Guided Image Generation via Intermediate Clean Estimates**

Ensuring safety and policy compliance in text-to-image diffusion models remains a critical challenge, as benign or adversarial prompts can often elicit prohibited content, e.g. nudity and protected intellectual property. While training-based unlearning methods are effective, they are computationally expensive and prone to catastrophic interference with general capabilities. Conversely, existing test-time defenses are primarily prompt-centric, relying on modifying textual descriptions only, and overlook the visual signals for detection. In this paper, we propose to leverage the intermediate clean image estimated during the generation process and employ a sparse margin objective to detect prohibited concepts. When a violation is detected, we immediately intervene by optimizing a structured low-rank residual in the text-conditioning space via truncated backpropagation. This design allows weight-preserving detection, keeps non-violating inference latency nearly unchanged as the maximum budget increases, and offers flexibility in safety performance via test-time scaling. Extensive experiments on Stable Diffusion v1.4 and v3.5 across nudity removal, IP protection, and style erasure demonstrate superior performance across suppression, fidelity and preservation compared to prior weight-preserving baselines, providing a scalable and flexible solution for safe generative deployment.

- type: 
- query: 

---

## 42. `2608.01142v2`

**EulerLoRA: Rank-Driven Jump Dynamics for Calibrated Parameter-Efficient Fine-Tuning**

Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning, but standard LoRA produces a single deterministic model and does not directly support predictive uncertainty estimation. We introduce EulerLoRA, a stochastic extension of LoRA that generates multiple predictive trajectories by sampling structured variations along the rank-one components of shared low-rank adapters, while preserving the deterministic LoRA transformation in expectation. We evaluate EulerLoRA with vision transformers on CIFAR-10, CIFAR-100, and HAM10000, together with out-of-distribution detection on SVHN. Across these benchmarks, EulerLoRA achieves comparable or improved performance relative to strong LoRA-Ensemble baselines. Using two rank-20 adapters, EulerLoRA requires approximately 3 million trainable adapter parameters, compared with about 10 million for a rank-8, 16-adapter LoRA-Ensemble, corresponding to roughly 69% fewer trainable parameters. These results show that useful predictive diversity can be obtained from a small number of shared adapters.

- type: 
- query: 

---

## 43. `2608.01348v1`

**Prompt-Driven Simulation with Feature Perturbation for Cross-Domain Few-Shot Object Detection**

Data augmentation, which simulates diverse visual variations to expand the source distribution and induce synthetic domain shifts, is a simple yet effective strategy for mitigating severe domain shifts and limited labeled target data in cross-domain few-shot object detection (CD-FSOD). Existing approaches rely on conventional data augmentation, such as Color-Jitter, Mosaic, and background-centric adaptation (e.g., Domain-RAG), which are limited in modeling complex domain shifts and often lead to suboptimal performance. In this paper, we propose PSP-FSOD, a principled framework that integrates prompt-driven domain simulation with feature perturbation regularization to improve generalization in CD-FSOD. To enable controllable domain synthesis, we design a prompt-driven strategy that leverages the visual grounding capability of large VLMs to jointly model foreground and background variations, generating semantically consistent yet domain-diverse training samples. Moreover, we adopt a grounding-aware generation scheme that guides object placement and alleviates semantic-spatial misalignment, thereby improving foreground adaptation. To ensure training stability and robustness, we further introduce a noise-induced feature perturbation mechanism that injects Gaussian noise into multi-scale intermediate features with distribution correction, encouraging consistent predictions under perturbations and reducing reliance on domain-specific cues. Extensive experiments demonstrate that PSP-FSOD produces high-quality domain-diverse supervision and learns domain-invariant representations, consistently improving performance across CD-FSOD benchmarks.

- type: 
- query: 

---

## 44. `2608.03228v1`

**SAKI: Score-Aware Low-Rank Key Indexing for Long-Context KV Retrieval**

Existing low rank KV cache methods preserve either model weights or key variance, neither of which directly reflects the attention scores used during inference. We derive the expected attention score distortion caused by rank r key compression and show that it yields a covariance weighted low rank objective. Under a margin condition, controlling this distortion also improves top k recall. The optimal rank r solution has a closed form asymmetric factorization obtained from the SVD of the covariance weighted query key operator. This motivates SAKI, a training free KV cache index that directly preserves attention scores rather than key reconstruction quality. Across LLaMA 3.1 8B, Qwen 2.5 7B, Mistral 7B v0.1, and Llama 3.2 3B, SAKI outperforms key PCA at every tested rank. At rank 32, it removes 13 to 30 percent of PCA's remaining top 64 recall error, including improvements from 0.748 to 0.799 on LLaMA 3.1 8B and from 0.786 to 0.850 on Qwen 2.5 7B. It improves 68 to 89 percent of attention heads per model, with the largest gains in deeper layers. Predicted score MSE reductions closely match empirical measurements, with a Pearson correlation of 0.997, while ablation studies confirm that the gains arise from optimizing the attention score objective rather than covariance weighting alone. Analysis of the scoring operator further explains why weight only, invariant subspace, and key reconstruction methods can be suboptimal.

- type: 
- query: 

---

## 45. `2608.01734v1`

**LLM-Guided Retrieval for Prediction of Molecular Perturbation Responses**

Predicting transcriptomic responses to small-molecule perturbations across cell lines is central to drug discovery, but exhaustive profiling of drug-cell combinations is infeasible. We frame molecular perturbation prediction as retrieve-and-aggregate: approximate an unmeasured drug's response in a cell line by aggregating measured responses of a small set of biologically related compounds. We propose LLM-Guided Retrieval (LGR), where a large language model (LLM) ranks candidate neighbor drugs (restricted to those profiled in the target cell line); after which a fixed mean aggregator combines their observed expression deltas to form the prediction. We evaluate on the Tahoe-100M single-cell perturbation atlas under unseen-drug, unseen-cell-line, and open-world regimes. LGR consistently improves over drug mean, ChemCPA, and chemistry-based kNN baselines, with the strongest gains for unseen cell-line generalization, where it achieves higher correlation and lower error than mean baselines. Across settings, LGR improves directional (sign) accuracy of gene regulation, indicating better recovery of biologically meaningful perturbation effects even when magnitude-based metrics are similar. These results suggest that retrieval quality, rather than predictor complexity, is a key driver of zero-shot molecular perturbation prediction, and that LLMs can provide a useful biological prior when used as constrained retrieval modules.

- type: 
- query: 

---

## 46. `2608.01035v1`

**WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA**

Vision-Language-Action (VLA) models have emerged as a prominent paradigm for end-to-end autonomous driving; however, their efficient deployment is severely constrained by high computational latency and exposure bias arising from sequential autoregressive decoding. Conversely, while specialized diffusion policies enable low-latency, parallel execution, training them from scratch typically yields narrow, single-task architectures that lack holistic visual-linguistic reasoning. Successfully transforming pre-trained autoregressive generalists into parallel diffusion models could combine multi-task cognitive intelligence with execution efficiency, yet this transition presents a formidable architectural challenge due to mismatched attention patterns (causal versus bidirectional) and divergent optimization objectives. To bridge this divide, we introduce WAM-Diff2, a multi-task discrete diffusion VLA framework powered by a three-stage hierarchical distillation strategy. By structuring the architectural shift through progressive block-wise adaptation, block-wise distillation, and model-wise cross-scale distillation, WAM-Diff2 preserves the underlying semantic foundations of the base model while accelerating inference. Extensive evaluations across driving understanding, perception, and planning benchmarks demonstrate that WAM-Diff2 effectively mitigates exposure bias and achieves performance parity with autoregressive baselines. Crucially, the autoregressive-to-diffusion transition yields a 2.8x decoding speedup, which scales to an ultimate 15.1x acceleration when combined with system-level optimizations including FlashInfer and CUDA Graphs.

- type: 
- query: 

---

## 47. `2608.03023v1`

**Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing**

Remote sensing semantic segmentation is hindered by costly pixel-level annotations, motivating training-free open-vocabulary methods. Recently, the recent release of DINOv3 brings DINO.txt, which equips the standalone DINO backbone with image-text contrastive learning and thus opens up the possibility of open-vocabulary segmentation. We propose DinoSplat-OV, a training-free framework that adapts DINOv3 to remote sensing without fine-tuning or additional pretraining. Targeting the dense distribution, multi-scale nature, and large size of remote sensing imagery, we design two core modules. Its Text-aware Laplacian Propagation module de-noises patch-level predictions by combining textual semantic affinities with local visual similarity, improving regional consistency while preserving boundaries. Its Gaussian Splatting Upsampling module reconstructs pixel-level features through RGB-guided anisotropic aggregation and test-time optimization. A global-anchor sliding-window strategy further supports large-scale imagery. Experiments on UDD5, DOTA, and LoveDA demonstrate competitive or superior performance over existing training-free methods, effectively filling the gap of DINO-series models in training-free open-vocabulary segmentation and providing a viable new path for further advances in this direction.

- type: 
- query: 

---

## 48. `2608.03388v1`

**Don't Let Me Ask for It: LLMs Show Deficiencies in Active Multi-Turn Information Acquisition for Abductive Inference**

Abductive reasoning requires forming hypotheses that explain observed evidence and revising them as new evidence becomes available. While large language models (LLMs) are often evaluated on whether they solve abductive reasoning tasks correctly, less is known about how they acquire evidence, update their hypotheses, and decide when to stop. We introduce Alien Abduction game, an interactive probe for studying these behaviours under different interaction modes. The modes vary in whether evidence is provided upfront or across turns, and whether queries are selected by the model or examples are provided by the oracle. Across models, providing evidence upfront leads to higher success rates than distributing it across turns. In multi-turn settings, some models commit before using the available evidence, while others exhaust the turn budget without converging. Models also achieve higher success rates when examples are provided by the oracle than when they select their own queries, although their final hypotheses are more consistent with the evidence they selected. These findings suggest that models may form hypotheses that fit self-selected evidence without sufficiently distinguishing them from alternatives, and may struggle to validate and refine their hypotheses or determine when to stop.

- type: 
- query: 

---

## 49. `2608.02603v1`

**WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity**

Controllable video generation models are increasingly being developed as world models. Accordingly, evaluating them in this role extends beyond the apparent appearance of generated videos to the inherent reactivity of the worlds they depict: the ability to infer from the scene state how the world should react and to generate plausible consequences not explicitly described in the input. Yet existing benchmarks mainly assess visual quality or explicit instruction fulfillment by checking whether requested actions and interaction outcomes are realized, leaving inherent reactivity underexamined. We introduce WorldExam, a hierarchical diagnostic benchmark spanning four levels: Visual Quality, Control Adherence, Spatial Consistency, and World Reactivity. It comprises 1,474 cases across eight dedicated tasks and supports unified evaluation of camera-, action-, and language-driven model paradigms. The World Reactivity level evaluates scene-conditioned reactions and goal-directed behaviors beyond what is explicitly specified in the input. Evaluation of 20 representative models reveals a clear capability split. Camera-driven models excel at camera control, but their interfaces do not support dynamic interaction; action-driven models control subjects more precisely but often leave the world unresponsive; and language-driven models perform better on interaction but follow complex controls less faithfully. No model combines broad task coverage with consistently strong performance, showing that high visual quality and explicit instruction fulfillment do not guarantee inherent reactivity.

- type: 
- query: 

---

## 50. `2608.03239v1`

**Relational Priors as Convergence Pressure in LLM-Based Multi-Agent Systems**

Large language model-based multi-agent systems (LLM-MAS) are designed through roles, debate protocols, and aggregation rules. These choices create implicit social expectations: agents may be expected to trust, challenge, defer to, or collaborate with peers. We study the effects of making inter-agent relation semantics explicit. We use a minimal signed-network formulation of relational priors and inject natural-language renderings into agent system prompts while holding the task protocol fixed. Across a commons-governance simulation and multi-agent debate, relational priors primarily act as convergence pressure: increasing relational positivity tends to make agents coordinate or agree more readily. This pressure can help when utility rewards behavioral alignment, as in sustainable resource governance and subjective consensus. It does not, however, reliably improve accuracy. In objective QA debates, higher positivity can increase agreement even when correctness-conditioned agreement does not improve and may decline in some settings. Effects vary by model backbone, relation type, and topology; explicit neutrality is not equivalent to omitting relational framing. We argue that relational priors should not be a default add-on for LLM-MAS. Their safer use is diagnostic and task-specific: compare against a no-prior baseline, monitor correctness-conditioned metrics when truth matters, and omit the relational layer when validation does not justify it.

- type: 
- query: 

---

## 51. `2608.02206v1`

**CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution**

Sparse-view 3D Gaussian Splatting Super-resolution is highly challenging since the sparse and low-resolution (LR) inputs lack sufficient geometric and high-frequency information for accurate reconstruction. To achieve high-quality reconstruction, existing sparse-view super-resolution methods adhere to two-stage pipeline that performs LR Gaussian reconstruction and then high-resolution (HR) Gaussian refinement, which directly results in stage-wise Gaussian transfer and reconstruction error accumulation. To this end, we propose CLEAR, a Conflict-aware Learning via Evidence-guided Adaptive Routing, as the first unified single-stage framework for Sparse-view 3D Gaussian Splatting Super-resolution. Specifically, CLEAR performs joint the optimization of authentic LR observations and external HR priors within a unified Gaussian representation. To mitigate the gradient conflicts introduced by sparse supervision during training, we propose a Gaussian-wise conflict-aware optimization strategy that regards the LR gradient as a reliable anchor and applies evidence-conditioned soft correction only to severe HR conflicts. Moreover, to recover high-frequency details, we introduce an evidence-guided Patch-to-Gaussian routing mechanism which estimates patch reliability and detail demand, lifts them into Gaussian space, and selectively routes high-frequency gradients and densification. Finally, we employ shared Gaussian dropout and a detached mid-training anchoring to enhance the robustness of training framework. Extensive experiments on both synthetic and real-world $4\times$ super-resolution benchmarks demonstrate that CLEAR consistently achieves state-of-the-art rendering quality and superior geometric fidelity.

- type: 
- query: 

---

## 52. `2608.03270v1`

**GUI-Lens: Coarse-to-Fine Cropping for GUI Grounding with General-Purpose VLMs**

GUI grounding maps natural-language instructions to click locations and is essential for reliable GUI agents. The task remains difficult on high-resolution, densely populated interfaces because a vision-language model (VLM) may recognize a requested control without locating it precisely enough for interaction. Most existing methods provide various forms of localization assistance, but still rely on a direct click prediction, allowing visual ambiguity or an inaccurate initial estimate to propagate to the final result. In this paper, we introduce GUI-Lens, a coarse-to-fine grounding framework that allows a general-purpose VLM to determine the target through active visual observations. Specifically, GUI-Lens extracts OCR text and detected UI components from the screenshot and presents their positions as coordinate references. Using the instruction, the current view, and these references, the VLM selects the region and scale of the next view, which is cropped and enlarged to provide finer visual details. This process continues over successively focused views until the target is determined. Proposed crops and clicks are checked against the instruction throughout the process, and the final local position is mapped back to the original screen coordinates. Experiments on four GUI grounding benchmarks and three general-purpose VLM backends show that GUI-Lens improves overall grounding accuracy by up to 24.9 percentage points and achieves state-of-the-art performance with GPT-5.5.

- type: 
- query: 

---

## 53. `2608.03529v1`

**Consensus Measures for Unstructured Biomedical Text Annotations**

Biomedical literature is increasingly mined for knowledge beyond the questions it was written to answer. Because the target concepts are not known in advance, annotators prefer open-ended labels, whose agreement is hard to quantify. We study soft inter-rater reliability for annotators providing unstructured texts for biomedical annotation tasks. Synthetic experiments show that soft reliability can be quantified using a variety of semantic equivalence measures, and that the choice of measure affects failure modes of the estimation. Embeddings are scalable, but limited when differentiating similar but distinct concepts. Large language models are promising, but limited by scalability for estimating agreement by chance. Finally, we suggest measures based on natural language inference as a sensible compromise.

- type: 
- query: 

---

## 54. `2608.02471v1`

**Action-grounded tissue affordance enables anticipatory auto-framing that lowers surgeon cognitive workload during laparoscopic surgery**

Computational attention models could help surgeons manage the visual demands of laparoscopy, but they require dense spatial labels that are difficult to obtain because surgical intent is highly specialized and tacit. Here, we introduce DiffeoAfford, an action-grounded tissue affordance framework that retrospectively derives visual attention supervision from completed surgical procedures. By combining diffeomorphism-constrained tissue tracking with instrument trajectory analysis, DiffeoAfford generates affordance hotspot labels without manual per-frame annotation. A real-time prediction model trained on these labels anticipates relevant surgical regions and enables AffordView, an assistive auto-framing system for laparoscopic visualization. The proposed framework aligns with expert annotations and intraoperative surgeon gaze, and reduces surgeon cognitive workload during real-world evaluations using subjective, physiological, and behavioral measures.

- type: 
- query: 

---

## 55. `2608.00626v1`

**Where Does Generative Difficulty Reside? An Empirical Study of Target Representations**

The target representation defines the distribution an image generator must learn, yet it is often treated as an interchangeable interface. This assumption is particularly questionable for continuous masked generators, which combine contextual inference from visible tokens with conditional modeling of each missing token. We study raw pixels, SD-VAE latents and DINOv2 as well as MAE representation-autoencoder features within a unified masked autoregressive rectified-flow model. Under a shared ImageNet training budget, these spaces exhibit distinct optimization and inference regimes. DINOv2 converges fastest in both iterations and computation but benefits strongly from a wider local denoiser and direct context fusion. Pixels optimize substantially more slowly and require a different prediction, masking, and guidance configuration. MAE reconstructs images more faithfully and exhibits clear semantic clustering, yet produces generations substantially worse than DINOv2. The representations also respond differently to classifier-free guidance and occupy distinct precision-recall trade-offs. Together, our results show that compression, reconstruction fidelity, token dimensionality, and visible semantic clustering do not individually predict generative behavior. Instead, target representations redistribute difficulty across contextual modeling, per-token denoising, and inference-time distributional control.

- type: 
- query: 

---

## 56. `2608.01651v1`

**Bole: Efficient Tree Speculation for Hybrid-Attention Language Models**

Hybrid-attention large language models combine full attention with recurrent linear attention to reduce long-context inference costs, yet their autoregressive decoding remains memory-bound. Tree speculative decoding offers an attractive acceleration path, but existing tree-speculation systems are designed around the key--value caches of full-attention models. On hybrid models, they traverse recurrent layers branch by branch and materialize a full state for every proposal node, causing verification latency and transient memory to scale poorly with tree and batch sizes. We present Bole, a kernel--runtime co-design that enables efficient tree speculation for hybrid-attention LLMs. Bole transforms the linear-attention recurrence into a tree-structured closed form and realizes it with a resource-efficient GPU kernel, verifying all proposal nodes in parallel and accelerating linear-attention tree verification by 3.4--7.7$\times$. It losslessly encodes speculative state updates as token-level factors and reconstructs only the state selected after sampling, reducing transient state memory by 82--99$\times$ and freeing GPU capacity for KV caches. Its integration into SGLang, a widely deployed production LLM serving engine, couples efficient state management with a batch-wide verification budget calibrated to the complete hybrid forward. Across four models, two GPU platforms, and diverse datasets, Bole delivers up to $4.72\times$ the offline decode throughput of autoregressive decoding and up to $2.03\times$ that of the strongest tree-speculative baseline. Under online agent workloads, it reduces TTFT and TPOT by up to $67.6%$ and $49.9%$, respectively, over the strongest tree-speculative baseline.

- type: 
- query: 

---

## 57. `2608.03974v1`

**JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion**

Real-time video editing requires low-latency causal generation with bounded computational resources while preserving source fidelity and long-term temporal consistency. We present JoyAI-Video-Edit, a 16B-parameter autoregressive diffusion framework for real-time, open-ended video editing without access to future frames or a predefined video duration. Our method combines chunk-wise autoregressive adaptation, Source-Anchored Distribution Matching Distillation (SA-DMD), and Long-Horizon Autoregressive Distillation to reduce train--inference mismatch, preserve source fidelity during two-step generation, and mitigate accumulated temporal drift. Extensive automatic and human evaluations show that JoyAI-Video-Edit substantially outperforms existing streaming editors and remains competitive with strong offline systems on both short and long videos. The complete system achieves end-to-end 720p video editing at approximately 30 FPS on a single Nvidia B200 GPU. Code is available at https://github.com/jd-opensource/JoyAI-Video-Edit.

- type: 
- query: 

---

## 58. `2608.02144v1`

**Quaternion Tensor Modeling for Joint Color-Polarization Demosaicking**

Division-of-focal-plane (DoFP) color polarization cameras enable snapshot acquisition of color polarization mosaic images, but the inherently sparse sampling pattern makes color polarization demosaicking severely ill-posed. Existing methods often fail to jointly exploit the correlations among polarization channels and the physical constraints inherent in polarization imaging, resulting in noticeable demosaicking artifacts. To address this issue, a quaternion-tensor-based color polarization demosaicking (CPDM) method incorporating Stokes-domain total variation (TV) regularization is proposed. Correlation analysis shows that the correlations among polarization channels are stronger than those among color channels. Accordingly, the color polarization images acquired at $0^\circ$, $45^\circ$, $90^\circ$, and $135^\circ$ are encoded into the four components of a third-order quaternion tensor, with the color channels organized along its third mode. A low-rank prior is then imposed on the quaternion tensor to exploit the global structural redundancy in the color polarization data. Moreover, spatial gradients are mapped to the Stokes domain through an orthogonal transformation to separate intensity, polarization and residual variations, with adaptive quaternion weights enabling component-specific regularization and preserving the energy consistency of the reconstructed Stokes vectors. An efficient optimization algorithm is derived for the resulting model. Extensive experiments demonstrate the superior demosaicking performance of the proposed method.

- type: 
- query: 

---

## 59. `2608.03082v1`

**DiverseDiT++: Quantifying, Analyzing, and Promoting Representation Diversity in Diffusion Transformers**

Recent advances in Diffusion Transformers (DiTs) have enabled remarkable progress in visual synthesis, benefiting from their superior scalability. To facilitate DiTs' capability of capturing meaningful internal representations, recent works such as REPA incorporate external pretrained encoders for representation alignment. However, the underlying mechanisms governing representation learning within DiTs remain poorly understood in the community. To this end, this paper first presents a systematic analysis of the representation dynamics of DiTs via quantifying the diversity of block-wise representations. Specifically, we introduce a novel metric, termed the Weighted Diversity Score (WDS), to measure the representational discrepancies across different blocks. Through extensive investigations on the evolution and influence of internal representations under various settings, we reveal that representation diversity across blocks is a critical factor for effective representation learning in DiTs. More importantly, WDS exhibits a strong correlation with synthesis quality across diverse settings, model scales, and training stages (Pearson's $r=-0.869$ with $\log(\text{FID})$), suggesting its potential as an indicator to reflect model performance and a principled guide for model optimization. Based on this key finding, we propose DiverseDiT++, a novel framework that explicitly promotes diverse representation learning. Concretely, our method incorporates long residual connections to diversify input representations across blocks and a representation diversity loss to encourage blocks to learn distinct features. Extensive experiments on ImageNet $256\times256$ and $512\times512$ demonstrate that our DiverseDiT++ yields consistent performance gains and convergence acceleration when applied to different backbones with various sizes,...

- type: 
- query: 

---

## 60. `2608.01174v1`

**Does Machine "know" interpersonal pragmatics? Evidence from MARBERT's learning of emoji pragmatics in Arabic digital discourse**

This study examines Transformer-based models' ability to learn emoji pragmatics in Arabic digital discourse (ADD), providing evidence from MARBERT's behavior with interpersonal pragmatic functions (IPFs). A corpus of 8,504 unique emoji-posts collected from Facebook via Python was used in the study. These posts were manually annotated, developed, and labeled for five IPFs: Politeness, Respect, Solidarity, Empathy, and Encouragement. A mixed-method approach was employed comprising statistical methods and interpretative analyses involving speech act theory, politeness theory, and rapport management theory. MARBERT was fine-tuned to model these context-dependent pragmatic functions. Findings demonstrate MARBERT's ability to learn these IPFs, achieving strong performance on unseen data, with an accuracy of 93%, a micro F1-score of 0.61, and a macro F1-score of 0.56, demonstrating its effectiveness in capturing interpersonal functions beyond conventional sentiment analysis. Function-level evaluation showed that Politeness and Respect were identified more accurately than Solidarity, reflecting differences in the explicitness and contextual dependence of IPFs. The study concludes that Transformer-based models learn patterns of face management and relational communication but remain challenged by highly implicit social meanings. It contributes a novel computational approach to modeling emoji pragmatics and advances the integration of interpersonal pragmatics with NLP for digital communication research.

- type: 
- query: 

---

## 61. `2608.01005v1`

**Hierarchical Solomonoff Induction: An Unbounded Machine Learning Model**

Solomonoff Induction, or SolInd, provides an ideal unbounded model of a priori sequence prediction but cannot naturally describe extrapolation from a given training dataset, as performed by Large Language Models. We apply de Finetti's theorem on exchangeable distributions to SolInd to produce what we call Hierarchical Solomonoff Induction, or HSI, which maintains a hyperprior over all Solomonoff priors that can be conditioned on previously observed sequences. We extend Wood et al.'s proof that universal mixtures of semimeasures are equivalent to SolInd to show that universal mixtures of these mixtures are also equivalent, proving that HSI=SolInd. We also prove that HSI's excess error on any distribution, compared to its true generator, is bounded by that generator's complexity in the hyperprior. This result is directly comparable to SolInd's prediction error being bounded by the Kolmogorov complexity of the sequence being predicted, and forces HSI's average excess error to converge to 0 as a dataset grows, leading to optimal prediction in the limit. We claim that HSI is an ideal unbounded model of sequence prediction given a dataset in the same way that SolInd is ideal over individual sequences.

- type: 
- query: 

---

## 62. `2608.03990v1`

**Assessment of Conditional Diffusion Model for Synthetic Histopathology Image Generation**

Synthetic histopathology image generation has emerged as an approach that may address data scarcity in computational pathology, yet current evaluation methodologies may not fully assess synthetic data quality for medical applications. This work investigates and addresses limitations in existing evaluation metrics, investigating an approach for assessing synthetic histopathology image quality through domain-specific metrics and downstream task validation. We show that conventional synthetic data evaluation metrics such as Frechet Inception Distance (FID) and Inception Score (IS) may have limitations when applied to histopathology images due to their reliance on ImageNet-pretrained feature extractors. To address these limitations, we propose for consideration modified FID and IS approaches utilizing foundation models pretrained on digital pathology datasets, supplemented by precision-recall based metrics as part of an additional quality assessment. Using conditional denoising diffusion models trained on four benchmark datasets, with a two-step training approach, we generated synthetic datasets with systematically varied quality characteristics. We also measured the correlation between the synthetic data quality metrics with downstream nuclei segmentation performance using common metrics including the aggregated Jaccard index (AJI+) and the Dice coefficient. The study results suggest that pathology-specific metrics may provide improved discriminative power. Specifically, the modified Inception Score indicates higher correlation with downstream task performance (r=0.6096 with AJI+, p=0.0122), compared to the original IS (r=0.0708, p=0.7944). Our observations indicate that increasing the variety of generated training data has a higher positive correlation with segmentation model performance than improving the visual fidelity of individual generated images.

- type: 
- query: 

---

## 63. `2608.01633v1`

**GraphIR: Architecture-Level Search States for LLM-Guided Neural Architecture Evolution**

Large language models (LLMs) enable neural architecture search (NAS) directly over executable neural network programs. However, code-level flexibility does not provide the architecture state needed for effective mutation: LLMs must infer tensor dependencies, editable components, and compatibility constraints from implementation details. To address this representation mismatch, we propose GraphIR, an architecture-aware intermediate representation that supplements executable programs with a mutation-aligned candidate state. GraphIR organizes each candidate through three complementary views: a computation skeleton describing tensor flow, a mutation surface exposing editable modules and operations, and a validity envelope capturing interface contracts, propagated shapes, and downstream dependencies. To evaluate our method, we construct NAS-Dependency, a 120-question benchmark covering six complementary dependency-reasoning dimensions. The diagnostic shows that GraphIR is particularly effective at identifying exact producer occurrences, tracing dependency propagation, and diagnosing interface and failure risks. Across six downstream benchmarks including CLRS, GraphIR achieves the best overall search performance while maintaining comparable model size and favorable end-to-end NAS efficiency when integrated into OpenEvolve. These results show that a mutation-oriented architecture state provides an effective interface between executable neural programs and LLM-guided architecture evolution.

- type: 
- query: 

---

## 64. `2608.01321v1`

**BiCAA: Bidirectional Credit Assignment for Search-Augmented Agent**

Multi-step search is a fundamental capability for search agents, enabling them to iteratively acquire, refine, and integrate external evidence for complex reasoning QA. However, vanilla GRPO allocates rewards exclusively based on the model's final outputs, yielding outcome-only supervision with no supervisory signals for intermediate reasoning steps. Such sparse supervision easily causes training instability and redundant search behaviors on multi-step search tasks. To mitigate this limitation, we adopt process reward to deliver stepwise supervision signals. For this process reward, we propose two complementary criteria to judge each search step: whether the step yields new evidence to facilitate problem solving, and whether it forms an efficient, pivotal intermediate decision within the overall reasoning trajectory. Building on this insight, we propose BiCAA: a bidirectional credit assignment framework that delivers dense, distinguishing process rewards for search-augmented agents. BiCAA builds bidirectional process rewards by fusing two complementary signals: forward solvability gain and hindsight success criticality. The former quantifies step-wise improvements in answer plausibility, while the latter evaluates each step's necessity for final success via hindsight outcome-based criticality scoring. We modulate and aggregate the two signals and then fuse them with the outcome reward. Experiments on search-augmented QA benchmarks show that BiCAA stabilizes policy optimization, reduces redundant search behavior, and achieves competitive performance.

- type: 
- query: 

---

## 65. `2608.02358v1`

**ScrambleToolBench: Agents Search Exhaustively Even When Their Own Map Points to the Next Step**

To operate robustly in open-world environments, autonomous agents should be able to infer the behavior of unfamiliar systems through interaction alone, even in the absence of documentation. However, existing tool-use benchmarks expose semantic tool schemas in static environments, allowing agents to rely on prior knowledge rather than autonomous discovery. To address this limitation, we introduce ScrambleToolBench, an interactive terminal benchmark designed to isolate behavioral reasoning. By removing semantic cues and enforcing a continuous task curriculum, the benchmark requires agents to uncover hidden tool behaviors entirely through trial-and-error interaction. The benchmark further introduces dynamic challenges, including mapping drift, stochastic action failures, and temporal execution windows, to evaluate whether agents can revise and adapt their hypotheses as the environment changes. Our evaluation of state-of-the-art language models reveals that successful initial discovery does not translate into robust adaptation. When faced with structural changes such as mapping drift, agents fail to use deductive strategies such as cycle tracing, and instead exhibit belief inertia or fall back to exhaustive search. Increasing test-time reasoning only amplifies this expensive brute-force search rather than enabling deductive recovery. While equipping agents with persistent memory reduces compounding errors, they remain unable to efficiently infer structural changes, highlighting a gap in current agent reasoning.

- type: 
- query: 

---

## 66. `2608.00582v1`

**Writing-System-Level Tokenizer Adaptation for Byte-Level BPE**

Pretrained byte-level BPE tokenizers can segment underrepresented languages inefficiently. Replacing a tokenizer changes the meaning of nearly every token ID, while vocabulary expansion enlarges the model's embedding and output matrices. We study post-hoc adaptation that keeps the model-vocabulary size fixed and preserves most existing token-to-ID assignments as a construction-time compatibility property. Directly transferring tokens from a language-specific tokenizer does not guarantee derivability through the target BPE merge graph: an inserted entry can conflict with the target's greedy merge ranks. We formalize this failure as the merge ordering problem and introduce BPE-guided insertion, which builds each transferred token through a target-reachable decomposition. Our pipeline uses script-aware row selection to limit collateral fragmentation, reconstructs target-script byte-level prerequisites, and applies guided insertion to maintain merge-graph reachability. On Ukrainian adaptations of Nemotron and GPT-OSS, it reduces token counts by 33.5% and 36.6%, keeps changes on English and the evaluated four-language European aggregate within 0.05%, and retains 78.5%/77.3% of original model-vocabulary rows at the same IDs. Constraint-matched global and frequency-based removal achieve similar Ukrainian compression but increase English/European token counts by 0.7-2.2%; fresh same-size retraining compresses Ukrainian slightly more but retains effectively no same-ID rows and increases English token counts by 7.6-8.6%. The reallocation increases token counts on the evaluated three-language Cyrillic micro-aggregate by 6.7%/10.1%. Structural audits find all 28,134/45,398 inserted BPE nodes reachable under ordinary rank-ordered merging and no retained same-ID model-vocabulary entry newly broken. We release all tokenizers and code.

- type: 
- query: 

---

## 67. `2608.03218v1`

**Self-Supervised Representation-Guided Generative Dataset Distillation**

Dataset distillation compresses a large training set into a compact synthetic set while retaining its downstream utility. Most existing methods target randomly initialized networks, whereas modern vision systems often adapt frozen pretrained encoders with lightweight modules. Distilled samples should therefore preserve the discriminative geometry of the pretrained representation space, which existing generative objectives do not explicitly consider. We propose self-supervised representation-guided generative dataset distillation (SRG), a framework that translates the SSL geometry into diffusion guidance. Specifically, SRG constructs class-wise prototypes from real-image SSL representations and performs guidance through three SSL-space objectives for prototype alignment, inter-class discrimination, and intra-class assignment. During diffusion sampling, it adopts a stage-wise guidance strategy: early denoising is anchored to the latent of the real image whose SSL representation is nearest to the assigned prototype, whereas later denoising is guided by the SSL-space objectives. This division preserves the visual realism provided by the generative prior while progressively steering samples toward representative and class-discriminative regions of the SSL representation space. SRG consistently outperforms the evaluated generative baselines across multiple datasets and IPC settings. A cross-encoder evaluation further indicates transfer across pretrained representation spaces. These results demonstrate the effectiveness of representation-guided generation for dataset distillation with pretrained SSL models.

- type: 
- query: 

---

## 68. `2608.01204v1`

**ShiJianBench: From Dialogue to Decision for Long-Horizon Evaluation of Investment Advisors**

Conversational investment advisors influence not only what users know, but also how they make subsequent decisions as market conditions evolve. Existing evaluations primarily assess response quality or observed outcomes, leaving the long-horizon pathway from advisor language to investor behavior difficult to audit. We introduce ShiJianBench, an offline framework for evaluating conversational investment advisors through matched investor trajectories under fixed historical market feedback. At its core is a multi-agent investor simulator with explicit evolving state variables, motive-driven deliberation, long-term memory, and dialogue-grounded updates. The simulator is calibrated against aggregate behavioral patterns from 7,199 real users, and advisor policies are evaluated using separate investor-side, service-side, and content-side metrics under a hard compliance gate. Experiments on Chinese fund-market traces from 2021 to 2026 identify a stable leading group of LLM advisors that combines substantially stronger personalized content with competitive investor-side trajectory outcomes. These results reveal a systematic distinction between producing a high-quality response and delivering an effective long-horizon intervention, motivating trajectory-aware evaluation of conversational advisors.

- type: 
- query: 

---

## 69. `2608.01963v1`

**OSSDD - a New Open Dataset for Sentinel-1 Ship Detection**

Ship detection in Synthetic Aperture Radar (SAR) images plays an important role for maritime situational awareness, especially with respect to different illegal activities at sea such as illegal fishing, smuggling or border violations. Modern ship detection methods using neural networks usually require large training datasets, which are considerably scarcer in the SAR domain than in the electro-optical domain. While several free datasets exist for this task, their availability and usability vary. In this paper, OpenSARShip-Ship Detection Dataset (OSSDD), a new dataset based on the well-known OpenSARShip 1.0 dataset is proposed for training neural networks for SAR ship detection. OSSDD is freely available and contains 15,197 Sentinel-1 amplitude patches in VV and VH polarization, binary ship masks, axis-aligned bounding box and rotated bounding box annotations for a total of 55,759 ships. The construction of the dataset, the contents and structure of the downloadable data and experiments with three common detector models (Faster R-CNN, FCOS, DETR) are shown and discussed. The results serve as benchmarks for future experiments. The dataset is available on Hugging Face at https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset.

- type: 
- query: 

---

## 70. `2608.03017v1`

**Paired Recipient-based Evaluation of Survival Prediction for Deceased Donor Kidney Transplants**

There has been significant interest in using machine learning algorithms to predict kidney transplant outcomes, such as the number of years until a graft inevitably fails. These prediction algorithms could possibly be used for pre-transplant donor-recipient matching to identify more compatible donors and recipients and thus improve post-transplant outcomes. In this study, we explore the use of survival prediction models trained on deceased donor kidney transplant data from the Scientific Registry of Transplant Recipients (SRTR). We propose a novel paired recipient-based evaluation framework that compares graft outcomes between two recipients who received kidneys from the same deceased donor, allowing us to evaluate the counterfactual benefit of changing the recipient for a certain donor. We find that five different survival prediction models, ranging in complexity from linear to deep learning-based models, all result in ~60% paired recipient-based accuracy. We further translate this accuracy into an interpretable quantity of post-transplant years gained. We also highlight major limitations of the commonly used concordance index (C-index) metric for evaluating survival prediction accuracy in this setting and demonstrate that our proposed paired recipient-based accuracy metric is more clinically relevant and better reflects real-world allocation settings.

- type: 
- query: 

---

## 71. `2608.03599v1`

**Disentangling Language Modeling and Boundaries**

Byte-level language models are usually argued for on the grounds of robustness, multilingual fairness, and character-level skills. We point to a different, structural advantage: because they read and write bytes, any two of them share an output space, so knowledge transfer between them is exact and independent of how either was originally tokenized. We hypothesize that the two distributions a byte-level model produces, one over the next byte, one over where its patch boundaries fall, can be disentangled and changed almost independently. A model could absorb a teacher's capability while keeping its own boundaries, or change how it places those boundaries while keeping its capabilities. We lay out the two experiments that would settle the hypothesis, alongside preliminary measurements of the properties they rest on. We argue that the community should move toward a byte-level interface as a shared standard: if the hypothesis holds, then once byte-level models are the norm, transferring capabilities and reshaping boundaries between them become cheap and routine, free of the per-model tokenizer that blocks them today.

- type: 
- query: 

---

## 72. `2608.03817v1`

**UHP Detection: LVLMs have their Unique Hallucination Pattern in the Consistency Space**

Large vision--language models (LVLMs) demonstrate strong multimodal reasoning capabilities but remain prone to hallucination, where model predictions are not grounded in visual evidence. Existing black-box hallucination detection methods estimate uncertainty through a single consistency metric, implicitly assuming that model uncertainty can be adequately characterized by a single measure. However, hallucinations exhibit diverse manifestations of uncertainty across different behavioral probes, making a single measure insufficient to characterize their underlying behavior. We propose \emph{Unique Hallucination Pattern (UHP) Detection}, a fully black-box framework that models hallucination as a structured uncertainty pattern defined by two axes: perturbation modality (image vs.\ text) and logical polarity (a statement vs.\ its negation). Their intersection produces four complementary consistency groups that capture distinct manifestations of model uncertainty, from which both within-group and between-group features are extracted to train a lightweight classifier. Through comprehensive experiments on AMBER and PhD across three LVLMs, UHP Detection consistently outperforms prior black-box and white-box baselines, with improvements of up to $+18.72\%$ AUC-ROC and $+20.07\%$ AUC-PR over the strongest black-box methods. Extensive ablation studies demonstrate that each consistency group contributes complementary information and that their combination forms a structured hallucination pattern. Furthermore, cross-dataset evaluation shows that this learned pattern generalizes across benchmarks, indicating that hallucination behavior reflects a model-specific consistency pattern. \textbf{Code is publicly available at} https://github.com/amirezzati/uhpdet.

- type: 
- query: 

---

## 73. `2608.03041v1`

**PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling**

Deep reinforcement learning (DRL) approaches for flexible job shop scheduling (FJSP) heavily rely on attention-centric architectures to achieve state-of-the-art performance. However, these models suffer from excessive parameter counts and prohibitive inference latency as problem scales expand. While liquid neural networks (LNNs) offer a parameter-efficient alternative for modeling adaptive state evolution, their inherently sequential dynamics bottleneck computational efficiency. To resolve this trade-off, we propose PLAN (Parallel Liquid-inspired Approximation Network), a lightweight representation learning framework that reformulates continuous liquid-state dynamics into a discretized and parallelizable formulation. PLAN structurally decouples state evolution from context aggregation, where liquid-inspired updates handle the primary evolving state representation, and a lightweight context aggregation module provides complementary global context. Furthermore, PLAN acts as a versatile, plug-and-play backbone that generalizes to complex FJSP variants, pairing with a compact stochastic module for stochastic FJSP and replacing heavy heterogeneous graph transformers in multi-faceted dynamic FJSP. Extensive evaluations across deterministic, stochastic, and multi-faceted dynamic FJSP benchmarks show that PLAN reduces the average makespan by 1.2%, 1.4%, and 2.3%, respectively, compared with the corresponding state-of-the-art baselines, with the improvement reaching 10.2% in one benchmark setting. PLAN also reduces average inference latency by 13.2%, 31.7%, and 26.9%, respectively, with a maximum reduction of 69.2% on the largest instances, while using only 22$-$47% of the baseline parameters.

- type: 
- query: 

---

## 74. `2608.03197v1`

**On the Implicit Flatness Bias of Sharpness-Aware Minimization: A Linear Stability Analysis with Quantitative Hyperparameter Bounds**

Sharpness-Aware Minimization (SAM) improves generalization by seeking parameters whose loss is robust to local adversarial perturbations, but the quantitative mechanism underlying its implicit bias toward flat minima remains unclear. In particular, the perturbation radius $ρ$ is typically treated as an isolated tuning parameter, despite defining the neighborhood in which SAM measures sharpness. We analyze mini-batch SAM near an interpolating minimum through linear stability. Under local linearization and gradient-noise alignment assumptions, we prove that every linearly stable minimum satisfies $λ_{\max}\leq\sqrt[3]{bΓ/(2ρη^2)}$, where $λ_{\max}$ is the largest Hessian eigenvalue, $b$ is the batch size, $η$ is the learning rate, and $Γ$ bounds the gradient norm. The bound quantitatively characterizes SAM's implicit flatness bias: holding the other quantities fixed, a smaller batch size, a larger learning rate, or a larger radius restricts linearly stable SAM to flatter minima. It also exposes a necessary trade-off: $ρ$ should be large enough to promote flatness, yet remain local enough to preserve the approximation and stable training. We validate this prediction in a controlled study of 900 models on CIFAR-100 with ResNet-18 and VGG-19, where increasing $ρ$ is consistently associated with a smaller largest Hessian eigenvalue across batch-size and learning-rate settings. Finally, we instantiate the analysis in Taylor-Locality Controlled SAM (TLC-SAM), which adjusts $ρ$ using the observed Taylor-approximation error and further reduces the top Hessian eigenvalue relative to fixed-radius SAM. Our results provide quantitative hyperparameter bounds and a stability--locality perspective for analyzing and designing SAM variants.

- type: 
- query: 

---

## 75. `2608.01284v1`

**Training nGPT**

The normalized Transformer (nGPT) realizes hyperspherical representation learning by constraining model parameter vectors and activation vectors to the unit hypersphere. In this paper, we describe a practical training recipe for nGPT and evaluate it on modern hybrid Mamba-2--Transformer Mixture-of-Experts (MoE) models. The recipe introduces Logit Gradient Preconditioning, Logarithmic Learning Rate Decay, GatedAdamW, angular update control, and optional exploration mechanisms. Compared with an unnormalized model of the same hybrid MoE architecture trained with AdamW, the 14B-total-parameter nGPT model reaches the same validation loss using approximately half as many training tokens. The recipe scales across the models considered, which contain up to 14B total parameters.

- type: 
- query: 

---

## 76. `2608.02034v1`

**Upper-Expectile Multi-Step Q-Learning for Off-Policy Reinforcement Learning**

Multi-step returns accelerate reward propagation in off-policy reinforcement learning, but couple the evaluation of each decision to the suboptimal logged actions that follow it, inducing a pessimistic bias that grows with the horizon. We propose Expectile $n$-step Q-learning (ENQ), which replaces the symmetric $n$-step temporal-difference (TD) loss with an asymmetric expectile loss on the action-value error, with expectile level $τ$ as the only method-specific hyperparameter added beyond $n$-step TD. We prove that the ENQ operator is a $γ^{n}$-contraction. Under deterministic dynamics, at $τ=1$, its bias vanishes at the optimal action-value function $Q^*$ on covered in-support pairs, and the corresponding fixed point satisfies the separation-$n$ instance and its multiples of the lower-bound inequality used by Long-Horizon Q-learning (LQL). Under stochastic dynamics, the operator bias admits two-sided bounds with horizon-independent noise constants. Using a single expectile level $τ=0.8$ and a fixed backup horizon across 27 manipulation and navigation task instances, ENQ is competitive with LQL on aggregate, achieves higher measured training-step throughput in our profiling study, and benefits more from a ten-critic ensemble in a controlled scaling experiment.

- type: 
- query: 

---

## 77. `2608.02428v1`

**DF$^3$: World Modeling via Decoder-Free Feature Forecasting in Autonomous Navigation**

Forecasting future states from video sequences is a critical challenge for autonomous robotic systems and a fundamental objective of world modeling. Prior generative methods operating at the pixel level inevitably overemphasize task-irrelevant details, leading to prohibitive computational overhead. While latent-based approaches attempt to mitigate this by predicting features directly, the persistent reliance on heavy decoders for state-to-task mapping remains a computational bottleneck. In this work, we propose Decoder-Free Feature Forecasting (DF$^3$), a novel framework that models world evolution entirely within the latent space and directly derives task outputs, completely eliminating the need for a decoder. Specifically, DF$^3$ injects learnable spatial queries into the terminal blocks of a frozen vision foundation model to extract future state representations directly. By employing a lightweight, unified Motion-Aware Context Fusion (MACF) mechanism that seamlessly integrates coarse flow warping with fine-grained latent cross-correlation, these queries interact with historical token representations to explicitly align and forecast the feature of the next frame. Subsequently, a specialized set of task queries probes these forecasted features for the downstream task. Extensive experiments on public benchmarks and zero-shot deployment in a robotic simulator demonstrate that DF$^3$ achieves performance comparable to state-of-the-art methods while offering superior efficiency and flexibility for integrated perception and control.

- type: 
- query: 

---

## 78. `2608.03250v1`

**ShielDroid: A Hybrid Approach Integrating Machine and Deep Learning for Android Malware Detection**

The rapid advancement of modern technology has led to a significant increase in the use of smart devices, such as smartphones and tablets, resulting in the widespread adoption of mobile applications. Although applications are required to undergo malware screening before being published on official app stores, many malicious applications successfully evade detection by concealing sophisticated malware variants. These malicious behaviors are often activated only during runtime, making them difficult to identify through conventional static analysis. As a result, malware may remain undetected until after installation, potentially causing irreversible damage to users and their devices. This study presents a real-time Android malware detection framework that analyzes application behavior to accurately identify and classify complex malware. The proposed approach employs a hybrid dynamic analysis technique to distinguish malicious applications from benign ones. After preprocessing and filtering the collected dataset, the applications are classified using multiple machine learning algorithms. A comprehensive performance evaluation is conducted to compare the effectiveness of different classification techniques in terms of detection accuracy and execution time. Experimental results demonstrate that a hybrid model combining Random Forest and a Multilayer Perceptron achieves the best overall performance, attaining an accuracy of 97.5% with an execution time of 22.945 seconds. The proposed framework can enhance mobile device security by enabling timely detection of malicious applications and reducing the risk of cyberattacks.

- type: 
- query: 

---

## 79. `2608.02688v1`

**Learning Molecular Representations from Cellular Phenotypes with Structure Preservation**

Phenotypic drug discovery enables the discovery of functional relationships between molecular structures and cellular responses. However, existing multimodal representation learning methods often optimize cross-modal alignment without considering the intrinsic organization of chemical space, resulting in distorted molecular representations and loss of structural information. We propose \textbf{PhenMol}, a structure-preserving framework for phenotype-aware molecular representation learning. PhenMol disentangles molecular and cellular representations into shared and private components, enabling phenotype-guided alignment while preserving chemical structures through a dedicated molecular branch. This design integrates cellular phenotype information without disrupting molecular neighborhood organization. Experiments on approximately $3.04 \times 10^{4}$ molecule--cell morphology pairs demonstrate that PhenMol improves molecular property prediction across 270 bioactivity tasks, molecule--phenotype retrieval, and clinical trial outcome prediction. Moreover, ECFP4-based structural analysis shows that PhenMol better preserves molecular neighborhoods and reduces embedding distortion compared with existing multimodal alignment methods. These results highlight the importance of structure-aware constraints in multimodal molecular representation learning and provide an effective approach for integrating cellular phenotypes with chemical knowledge for drug discovery.

- type: 
- query: 

---

## 80. `2608.03812v1`

**OmniPack: Unified Token Compression for Efficient Omni-modal Large Language Models**

Omni-modal large language models (Omni-LLMs) have achieved remarkable performance on audio-visual understanding tasks, but processing long and highly redundant visual and audio token sequences incurs substantial computational overhead, demanding aggressive token compression for efficient deployment. Existing methods often degrade at low token budgets: pre-LLM compression may discard structurally important and globally distributed evidence, whereas inner-LLM compression often underexploits query-conditioned audio-visual collaboration. To address these limitations, we propose OmniPack, a training-free framework that coordinates structural compression before the LLM with task-relevant semantic refinement within the LLM. Before the LLM, OmniPack removes structural redundancy through modality-specific importance, global coverage, and similarity-aware merging. After sufficient multimodal interaction, it further consolidates diverse, task-relevant representations through textual guidance and audio-visual collaboration. Extensive experiments on five benchmarks with three Omni-LLM backbones demonstrate that OmniPack consistently achieves the best performance-efficiency trade-off across diverse retention ratios, outperforming all existing methods. Notably, on Qwen2.5-Omni-7B, OmniPack preserves 98.0% of the original performance while reducing FLOPs to 16.7%, and still retains 92.9% of the original performance with only 6.8% of the original FLOPs.

- type: 
- query: 

---

## 81. `2608.02135v1`

**Cardiovascular Digital Twins from Physics Based to Data Driven Approaches**

Cardiovascular digital twins aim to create patient-specific computational models that evolve with clinical data to support diagnosis, prognosis, and therapy optimisation. Mechanistic models provide physiological interpretability but remain computationally demanding, whereas data-driven approaches improve scalability yet risk limited robustness. Emerging physics-informed, graph-based, and hybrid methods integrate physical constraints with relational learning across vascular networks. We review modelling paradigms, data assimilation frameworks, validation challenges, and translational pathways toward clinically deployable cardiovascular digital twins.

- type: 
- query: 

---

## 82. `2608.03664v1`

**Morphology-Aware Implicit Super-Resolution Network for Pathological Images**

Accurate diagnosis in Digital Pathology (DP) relies on high-resolution whole-slide images, yet clinical deployment is often limited by hardware costs. Super-Resolution (SR) offers a promising alternative by computationally enhancing low-resolution acquisitions. However, existing SR methods frequently struggle to preserve fine-grained cellular morphology, leading to texture oversmoothing and blurred structural boundaries under complex tissue variability. To address this issue, we propose Morph-ISR, a morphology-aware implicit super-resolution framework for DP that restores diagnostically relevant details with sub-pixel precision. Morph-ISR reformulates SR as a continuous coordinate-based reconstruction problem and integrates an Implicit Position-aware Kernel Generator (IPKG) to adaptively model spatially varying tissue morphology. To further enhance structural fidelity, a Morphological Fidelity Prior (MFP) is introduced, leveraging semantic guidance from a pre-trained cell segmentation network to enforce boundary-preserving and region-aware reconstruction, thereby improving the representation of critical cellular boundaries and nuclear textures. Experiments on TCGA and SurGen datasets show that Morph-ISR achieves the best LPIPS and ST-LPIPS among the evaluated methods, reducing them by up to 38.37% and 39.55%, respectively, over the second-best methods while maintaining strong PSNR and SSIM. These results demonstrate superior preservation of diagnostically relevant cellular boundaries and nuclear textures, while compact parameterization and high throughput support efficient edge deployment. Code and trained models will be released upon publication.

- type: 
- query: 

---

## 83. `2608.01186v1`

**QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction**

While feed-forward 3D Gaussian Splatting (3DGS) enables efficient 3D reconstruction, achieving high-fidelity rendering remains challenging. Existing pixel-aligned approaches suffer from spatial inflexibility and massive structural redundancy, whereas query-based methods lack 3D priors and entangle geometry with appearance, yielding blurry, pose-dependent results. To overcome these deficiencies, we propose \textbf{QuerySplat}, a feed-forward 3DGS framework driven by geometric priors and explicit appearance decoupling. Specifically, we design a dual-branch query-based decoder: the geometry branch leverages a pretrained Vision Geometric Model for spatial understanding, which intrinsically endows QuerySplat with pose-free modeling capabilities, while the appearance branch recovers high-frequency details through a dedicated pathway separated from geometric attribute regression. Extensive experiments demonstrate that QuerySplat mitigates the blurry rendering issues of early query-based models and consistently outperforms pixel-aligned approaches in rendering fidelity. On the challenging DL3DV benchmark, it achieves state-of-the-art novel view synthesis performance, with average PSNR gains of 2.30 dB and 1.04 dB over the best pose-free and pose-required baselines, respectively. Project Page: https://inspatio.github.io/querysplat.

- type: 
- query: 

---

## 84. `2608.03428v1`

**OliveGemma: A 3 Billion Visual Language Model for Recognising the Mediterranean & European Diet**

Image based dietary assessment offers a scalable alternative to self reported food diaries, yet fine-grained food recognition remains challenging due to high intra-class variability and visually similar dishes. This study presents OliveGemma, a vision language model for recognising and reasoning about Mediterranean and European cuisine. Built on the open-weight PaliGemma-2-3B architecture, OliveGemma is fine-tuned with LoRA on a unified corpus of 17,340 images from three European research project datasets (MedGR, ODIN, and VIPPSTAR), reconciled into a vocabulary of 216 composed dish categories and paired with 102,642 instruction style question-answer items covering dish recognition, likely and visible ingredients, class boundary discrimination, visual evidence and overall visual food understanding. Under a 3-fold cross-validation scheme, OliveGemma achieves a top-1 accuracy of 92.96% +/- 0.91%, exceeding the strongest CNN baseline (DenseNet-121) by 7.31% and outperforming zero-shot frontier models with exact instructions and bounded classes including Gemini Flash 3 and 3.5, GPT-5.4 Mini, and Claude Haiku 4.6 by 8%, 46%, and 64% respectively. Furthermore, OliveGemma demonstrates competitive performance on Top-3 and Top-5 accuracy, being second best across CNNs and frontier models, surpassed only by DenseNet-121. In addition, OliveGemma achieves 90.79% +/- 1.3% Exact-Set on the likely ingredients of the food categories. These results demonstrate that PEFT adaptation of a small VLM can surpass substantially larger proprietary models on specialised food recognition. The model is publicly available at https://huggingface.co/JamesZar/OliveGemma-3B and the experiments and results can be found at https://github.com/tsiokris/OliveGemma.

- type: 
- query: 

---

## 85. `2608.02551v1`

**Who Should Be Generated? Justifying Demographic Targets in Open-Ended Generation**

Fairness evaluation concerns not only what a model produces, but also what its outputs ought to be compared against. When a model generates "a CEO in the United States," the prompt leaves demographic realization to the model. Existing group fairness definitions assume that sensitive attributes are given on the input side. Generative audits instead examine output-side demographic composition, yet the targets they compare it against are typically supplied rather than justified. The upstream question is what the target distribution should be. We formalize this missing-target problem for demographic-value-unspecified generation and decompose target construction into four commitments: the evaluative object, prior admissibility, allocation, and operationalization. In this framework, we admit the geographic prior under a geographic-membership interpretation for the declared public-world use. The occupational prior, under an incumbency interpretation, requires an independently defended objective such as workforce-composition fidelity. Instantiating this construction in AP-Bench, we find substantial distribution divergence from geography-derived targets, ranging from 0.508 to 0.606 on a 0-to-1 scale. Replacing each geography-derived target with an equal-category comparator, while holding generations and measurement fixed, produces model-specific mean absolute cell-level $\mathrm{JSD}_2$ changes ranging from 0.279 to 0.355. Target construction is therefore not a preliminary to fairness evaluation but a component of it. What we supply is not a universal target, but a framework that makes explicit the justification required before a distribution can serve as a fairness standard.

- type: 
- query: 

---

## 86. `2608.03101v1`

**Double Down on Defense: Strengthening Deep Perceptual Hashes against Evasion Attacks without Retraining**

Near-duplicate image matching is crucial for trust and safety, provenance verification, copyright enforcement, and large-scale visual search. Modern platforms increasingly rely on deep perceptual hashes, which map visually similar images to nearby representations despite common image transformations. However, adversarial perturbations can cause near-duplicates to evade matching. We present DualShield, a plug-in defense that improves the robustness of existing deep perceptual hashes without retraining or modifying their underlying models. DualShield combines matching-time randomized smoothing, which aggregates decisions over perturbed reference-query pairs, with publication-time hardening, which adds an optimized imperceptible perturbation to each reference image before publication. Together, these mechanisms provide certified and empirical robustness. DualShield achieves a certified $\ell_2$ radius of approximately 0.3, guaranteeing that query perturbations within this radius cannot evade matching. We further evaluate it against adaptive white-box, black-box, and image-transformation attacks. Across eight deep perceptual hashes and three datasets, DualShield substantially reduces attack success rates while preserving low collision rates. These results show that deep perceptual hashes can be strengthened without costly retraining by improving the matching procedure and hardening reference images before publication.

- type: 
- query: 

---

## 87. `2608.01370v1`

**Understanding Synergistic Interactions among Pathology Foundation Models via Adaptive Fusion**

Pathology foundation models (PFMs) provide strong tile-level representations via self-supervised pre-training on large-scale pathology images. Yet, PFMs are developed under diverse and often opaque data, architecture, and objective choices, inducing latent representational biases that limit robustness and obscure what each model specialises in. We present AdaFusion, a lightweight adaptive fusion framework that integrates complementary signals from multiple frozen PFMs through (1) low-dimensional feature compression and (2) a sample-conditioned gating module that reweights model-wise (and optionally channel-wise) contributions. Beyond improving predictive accuracy, AdaFusion provides contribution-driven interpretation that offers evidence consistent with model-specific preferences and synergistic interactions across tissue phenotypes. We evaluate AdaFusion on three public benchmarks spanning treatment response prediction, prostate cancer grading, and spatial gene expression inference. AdaFusion consistently outperforms individual PFMs and other fusion baselines, while providing interpretable tissue visualisation which aligns model preferences with morphological patterns. Code is available at: https://github.com/xyx-98/PathoOracle.

- type: 
- query: 

---

## 88. `2608.01743v1`

**Toward Plasticity-Preserving KL Regularization for Capability Retention in LLM Reinforcement Learning**

Reinforcement learning (RL) has become a central paradigm for large language model (LLM) post-training, but optimization toward new objectives can degrade capabilities already present in the base model. KL regularization is widely used to mitigate such forgetting by constraining policy drift toward a reference model. However, standard full-policy KL regularization constrains the entire response distribution and may unnecessarily restrict exploration and target-task learning. This raises a natural question: can a more precise constraint preserve existing capabilities while minimizing interference with learning new tasks? To this end, we propose \underline{Co}rrectness-Conditioned \underline{KL} Regularization (CoKL), a conditional regularization framework that narrows the preservation constraint from the full output distribution to correctness-conditioned response distributions. We instantiate CoKL with forward KL divergence and derive a practical finite-group training objective for RL-based LLM post-training. At the population level, CoKL decouples the total probability assigned to correct responses from their correctness-conditioned distribution, thereby regularizing the relative probability allocation among reference-supported correct responses without directly anchoring incorrect outputs or total correctness mass. We further show that full-policy forward and reverse KL regularization induce a strict optimal correctness gap when the reference policy is imperfect, whereas CoKL avoids this limitation. Experiments in controlled multi-solution environments and continual post-training settings across multiple model scales demonstrate that CoKL achieves a more favorable balance between target-task improvement and prior-capability retention than existing regularization methods. Our code is available at https://github.com/Lumina04/CoKL.

- type: 
- query: 

---

## 89. `2608.02588v1`

**The Condition-Number Barrier in Sparse Least Squares**

In [AS21], Axiotis and Sviridenko conjectured that the linear dependence on the restricted condition number in sparse convex optimization cannot be improved by a polynomial-time algorithm. We establish their conjectured lower bound for least-squares objectives, conditional on the randomized exact-volume Small-Set Expansion Hypothesis in the weighted regular-graph formulation of Raghavendra, Steurer, and Tulsiani [RST12]. Concretely, for every fixed $γ\in(0,1]$, there is no randomized polynomial-time algorithm that, with probability at least $2/3$, returns a vector $x$ such that, writing $s=\lVert x\rVert_0$, \[ \lVert Ax-b\rVert_2^2 \leq \min_{\lVert z\rVert_0\leq k}\lVert Az-b\rVert_2^2+\varepsilon \quad\text{and}\quad s=O\!\left(k\,κ_{s+k}^{\,1-γ}\right), \] where $κ_r$ is the restricted condition number at sparsity level $r$. The result holds even on rational instances with $A$ of full column rank. The proof was first obtained using a fully automated Gemini-based agentic system developed internally at Google. The authors have verified the proof and edited it for clarity of presentation.

- type: 
- query: 

---

## 90. `2608.03631v1`

**SEER: A Self-Grounded Evidence Interface for Controlled Spatial Relation Classification**

Spatial relation questions require a model to identify the queried subject and object before comparing their layout. Yet a VLM can recognize both entities and still answer from the wrong instance or an ambiguous global view. We ask whether making query-specific evidence explicit can mitigate this failure and propose SEER (Self-grounded Evidence for Entity-Relation Reasoning), a training-free inference-time evidence interface for frozen VLMs. SEER hides candidate relations during pair localization, constructs a query-specific view with explicit subject/object roles, and retains the full image and sparse box geometry as complementary evidence. For relation-choice protocols with exact inverse support, an optional refinement swaps the entity roles and changes the forward decision only when exactly one visual state obeys the corresponding inverse relation. On an image-disjoint GQA-Train900 test frozen before model scoring, SEER pools to +3.94 [2.17,5.72] over Full; the gain remains positive under label-independent grounding-order counterbalancing and on the 535 rows whose entity names are unique. The unchanged protocol yields +4.35 to +11.79 on all 2,434 filtered EmbSpatial pair-relation questions across three models. Matched controls separate local refocus from role-explicit conditioning. These results establish query-specific evidence construction as the principal intervention, with reciprocal consistency as a smaller protocol-specific refinement.

- type: 
- query: 

---
