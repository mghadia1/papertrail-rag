# v3 HELD-OUT labeling sample — 40 fresh papers (seed 20260905)

Disjoint from the dev sample. Author held-out queries here; do NOT dry-run.

## 1. `2608.02806v1`

**Fast Object Removal Attacks on Safety-Critical Video-based Perception Systems**

By leveraging data from video-based perception systems, intelligent transportation systems (ITS) support safety-critical applications that improve road safety. However, adversaries may manipulate video frames to compromise downstream perception modules, causing failures in safety-critical functions and increasing risks to vulnerable road users. This paper presents a novel attack model and an end-to-end framework for near-real-time targeted object removal attack on a video-based safety-critical system. The end-to-end attack pipeline consists of four stages: localizing targets in each frame, retrieving coherent patches from earlier frames, blending them using context-aware alpha compositing, and reconstructing attacked frames. Experiments at an intersection on the South Carolina Connected Vehicle Testbed (SC-CVT) show that reconstructed frames have high global similarity to the originals, with frame-level Peak Signal to Noise Ratio (PSNR) above 40 dB and Structural Similarity Index Measure (SSIM) above 0.996. Using the YOLO-based detector, the attack reduces object detections by up to 97.59% and achieves a frame-level attack success rate of 94.48%. Across the evaluated detectors and frame resolutions, the mean execution time ranges from 0.074 to 0.172 seconds per frame on GPU hardware, indicating near-real-time performance in testing. The forensic evaluation using several pretrained tamper-detection models shows limited ability to distinguish reconstructed from authentic frames. The findings suggest that video-based perception is vulnerable to stealthy object removal attacks that can degrade the performance of safety-critical applications by reducing object detectability. These findings can help develop mitigation strategies against adversarial object removal attacks that threaten safety-critical applications, such as vision-based pedestrian safety systems.

- type: 
- query: 

---

## 2. `2608.01565v1`

**DocNavRAG: Document-Structured Graph RAG with Stateful Evidence Construction for Complex Document Question Answering**

Answering complex questions over large document collections requires assembling complementary evidence across sections and documents. GraphRAG offers structured retrieval but typically uses fixed traversal, while agentic RAG operates over weakly structured interfaces. Our key insight is that agents should navigate document structure within and across documents rather than repeatedly search from scratch. We introduce DocNavRAG, which organizes document hierarchies and cross-region relations into a navigable graph, exposes graph operations for locating, navigating, expanding, and fetching, and maintains an evolving evidence state to guide retrieval until sufficient evidence is collected. Across four long- and multi-document QA benchmarks, DocNavRAG improves answer quality and context sufficiency over the strongest baseline by 7.8\% and 17.7\% on average.

- type: 
- query: 

---

## 3. `2608.00836v1`

**Nonlinear Laplacians Improve Signed-Directed Graph Learning**

While signed-directed graphs have been studied using linear Laplacians in the design of graph neural networks, relatively little research has focused on developing non-linear Laplacian operators for such networks. We introduce a non-linear Laplacian operator specific to signed and directed networks (NLSD). This non-linear operator extends the concepts of the signed Laplacian for signed graphs and the Laplacian for directed graphs. The NLSD calculates node-specific potentials based on features More precisely, if the potential discrepancy is not aligned with the edge direction, we ignore it (and vice versa) leveraging message-passing techniques only across edges where potential discrepancies align with the edge's direction. Utilizing this novel operator, we propose an efficient spectral GNN framework (NLSD-GNN). We conducted comprehensive evaluations focusing on node classification and link prediction, examining scenarios involving signed, directional, or both types of information. Our findings reveal that this spectral GNN framework not only integrates signed and directional data effectively but also achieves superior performance across diverse datasets.

- type: 
- query: 

---

## 4. `2608.03297v1`

**Distractor-Aware Truncation: Disentangling Context-Length Effects from Signal Loss in Long-Context LLM Benchmarks**

A standard claim in the literature on retrieval-augmented and memory-augmented language models is that shorter context is better when the relevant information is preserved. We test this claim by running every sample of two long-context benchmarks -- BABILong and GraphWalks (BFS) -- at four context-retention fractions (100%, 75%, 50%, 25%) under two truncation protocols. The first is the naive protocol implicitly used in much prior work: drop content from the middle of the prompt. The second is distractor-aware: identify the task-relevant content for each sample and drop only the rest. We evaluate three sizes of the Claude family (Haiku 4.5, Sonnet 4.6, Opus 4.7) and, to test cross-provider generality, GPT-5.5 from a different provider; we apply the same protocol to two further benchmarks (MRCR v2, Oolong). Under naive truncation, score collapses monotonically (paired Wilcoxon, Holm-corrected p_adj < 0.05 in all eight BABILong and GraphWalks cells). Under the distractor-aware protocol -- which preserves the signal by construction -- performance is preserved or improves: the two smaller Claude models show statistically significant gains on BABILong, while the larger models (Opus 4.7 and GPT-5.5) sit at their full-context ceiling. The naive collapse and its distractor-aware recovery replicate on GPT-5.5, ruling out a single-provider artifact. The mechanism is direct: under the naive protocol the answer-bearing content survives in fewer than 1% of samples at 25% retention; under the distractor-aware protocol it is preserved by construction. The naive protocol is therefore not a measurement of context-window effects; it is a measurement of how often middle-removal happens to spare the answer. We conclude that future studies of context-length effects must specify how they distinguish signal from distractor, or they are at best ambiguous between two opposite hypotheses.

- type: 
- query: 

---

## 5. `2608.02786v1`

**Evaluation Blindness: How Silent Measurement Failures Corrupt AI Systems from Training to Deployment**

AI systems can fail silently. The failure propagates through training loops, evaluation pipelines, and production monitoring stacks until downstream harm makes it visible. This paper introduces evaluation blindness: a measurement function M exhibits evaluation blindness with respect to failure class F when it produces readings indistinguishable from a healthy state while the system is actually failing, with no auxiliary signal flagging the gap. The problem surfaces at two lifecycle stages the literature has treated separately. At training time, reward models are gamed, importance-sampling corrections are silently miscalculated, and benchmark contamination inflates fine-tuning evaluations, all while loss curves look healthy and gradient updates proceed normally. At deployment time, monitoring fails to catch six classes of production failure, including an Operational category that is 100% silent by structural definition. We provide a formal detectability predicate unifying both stages. Four training-time case studies trace concrete breakdowns, including a real implementation bug in TRL PR #6594 where gradients are corrupted as loss decreases normally. A six-class taxonomy validated against 50 real-world incidents from court documents and regulatory filings finds that 53% of verifiable public failures were silent. A failure budget framework ties acceptable failure rates to use-case risk class. The implication is direct: measurement infrastructure is a correctness concern across the full AI lifecycle, not just at evaluation time. Data, code, and taxonomy schema are at https://github.com/priyanka25aug/llm-failure-taxonomy.

- type: 
- query: 

---

## 6. `2608.02446v1`

**Advancing Relevance Measurement with Vision-Language Models for Web-Scale Search**

Relevance evaluation plays a crucial role in personalized search systems, serving as a guardrail alongside user engagement metrics to ensure that search results align with user queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present a VLM-based automated relevance evaluation pipeline deployed within Pinterest Search for online A/B experiments. We rigorously validate the alignment between VLM-generated judgments and human annotations, demonstrating that VLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging VLM-based labeling further unlocks opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effects (MDEs) in online experiment measurements.

- type: 
- query: 

---

## 7. `2608.01709v1`

**SpatialQuery: Benchmarking Geometry-Grounded Multi-Instance Spatial Reasoning in Vision-Language Models**

Vision-language models (VLMs) achieve strong semantic understanding but remain unreliable in metric spatial reasoning, particularly when queries require comparing multiple instances of the same object category. We study this problem through the Closest-Instance Distance Query (CIDQ), where a model must identify the nearest visible candidate to a unique reference object and estimate their gravity-aligned floor-plane distance. We introduce SPATIALQUERY, a training- free framework for CIDQ reasoning from a single RGB image, together with SPATIALQUERY-1M, a benchmark containing over one million RGB-only question-answer pairs from 200 indoor scenes. SPATIALQUERY recovers instance-level metric geometry and transforms it into a canonical Bird's-Eye View through Scene Cubifying, which represents objects as uniformly sized, category-coded blocks to emphasize their relative floor- plane locations. We further propose Uncertainty-Aware Chain-of-Thought (UA-CoT) prompting, which incorporates geometry- derived per-instance uncertainty into the VLM reasoning process. Without task-specific fine-tuning or architectural modification, SPATIALQUERY with Qwen3-VL-8B achieves a Floor-MAE of 0.259 m, an Unc-Acc@0.3 m of 90.5%, and a proximity-decision accuracy of 84.18%, outperforming fine-tuned spatial specialists, general-purpose VLMs, and closed-source frontier models. Code, benchmark resources, and an interactive demo are available at https://namhai1810.github.io/SpatialQuery/.

- type: 
- query: 

---

## 8. `2608.02310v1`

**An Evidence-Grounded Retrieval-Augmented Transformer Framework for Health Misinformation Verification**

The rapid spread of false and misleading health information through digital platforms has become a major public health challenge, particularly during infectious disease outbreaks where delayed verification can influence public behaviour and hinder effective disease control. Although recent advances in automated health misinformation detection have shown encouraging results, most existing approaches rely heavily on global biomedical resources and often fail to capture the local context needed to verify claims in developing countries. This study presents a retrieval-augmented transformer framework designed to verify health-related claims using trusted evidence from the World Health Organization and the Nigeria Centre for Disease Control and Prevention. The framework combines semantic evidence retrieval with transformer-based classification to determine whether a claim is true, false, or misleading. To evaluate the proposed approach, a manually annotated dataset of 67 verified health claims covering coronavirus disease, Lassa fever, cholera, measles, and monkeypox was compiled from Nigerian fact-checking sources. Three transformer models and a retrieval-augmented configuration were evaluated. The Bidirectional Encoder Representations from Transformers model achieved the best performance, with an accuracy of 71% and a weighted F1-score of 0.66. Although retrieval augmentation did not improve classification performance because the current evidence repository was limited in size and coverage, the findings highlight the importance of comprehensive and authoritative knowledge sources for reliable health misinformation verification. The proposed framework provides a practical foundation for developing context-aware and evidence-driven health misinformation verification systems for Nigeria and other resource-constrained settings.

- type: 
- query: 

---

## 9. `2608.00925v1`

**Look Up and Look Back: Hidden Attention and Latent Orientation in a Frozen Foundation Model for Panoramic SLAM**

Monocular panoramic SLAM benefits from substantial visual overlap under large camera rotations, yet remains prone to errors caused by camera tilt, scale drift, and false loop closures. We show that a frozen panoramic geometry foundation model provides useful internal cues beyond its explicit geometric outputs: intermediate tokens encode gravity in the camera frame, while cross-view attention provides a compatibility cue for potential revisits. Building on these cues, we present HALO-SLAM. A gravity readout enables IMU-free spherical upright canonicalization. For loop closure, we introduce a cost-aware three-stage cascade combining DBoW2 event-level retrieval, attention-based compatibility filtering, and dense geometric validation through symmetric submap augmentation. Accepted revisits yield pixel-aligned 3D--3D correspondences in both local gauges, from which robust $\mathrm{Sim}(3)$ constraints are estimated and jointly optimized with sequential constraints in a global pose graph. Across 125 sequences from five real-world panoramic benchmarks, our method achieves \textbf{100\%} sequence success (\textbf{125/125}) under the stated criterion and the lowest ATE among the evaluated methods on all five benchmarks, reducing ATE by \textbf{30--88\%} relative to the best ERP-native baseline on each benchmark.

- type: 
- query: 

---

## 10. `2608.03545v1`

**Hi-TTRL: Regulating Consensus with Hints for Test-Time Reinforcement Learning**

Test-time reinforcement learning (TTRL) improves the reasoning capabilities of large language models without labeled data by updating the policy with pseudo-labels constructed through majority voting. While effective, the reward signal assigned from majority voting is highly sensitive to consensus strength, defined as the frequency of the most common answer within a rollout group. In TTRL, consensus strength plays a dual role: it reflects both the reliability of the pseudo-label and the distribution of advantages. Low consensus can amplify updates from unreliable pseudo-labels through disproportionately large advantages, whereas high consensus reduces reward contrast and ultimately yields vanishing gradients. In this paper, we introduce Hi-TTRL, a test-time reinforcement learning framework that utilizes hints during sampling to regulate rollout consensus strength. Hi-TTRL first estimates consensus strength from a partial rollout group. When the consensus strength falls outside a target interval, it invokes a Markov chain Monte Carlo (MCMC) hint sampler. The sampler targets the power-transformed prefix distribution and uses finite-step approximate sampling to generate rollout prefixes as hints. By tuning the power exponent, Hi-TTRL generates hints with a sharpened or flattened power target, steering rollout consensus strength toward the target interval. Experiments on multiple datasets and backbones show that Hi-TTRL consistently improves over standard TTRL, with ablations and consensus-steering analyses validating the effectiveness of adaptive hint-guided consensus regulation.

- type: 
- query: 

---

## 11. `2608.03382v1`

**LLM-Derived Priors for Thompson Sampling in Cold-Start Comment Recommendation**

Multi-armed bandit algorithms, especially Thompson sampling, are widely used in online recommendation. Despite their ability to adapt from online feedback, these methods often suffer from cold-start limitations when newly introduced arms have little or no interaction history. In our setting, the candidate arms are user-generated textual comments, whose semantic content can reveal a title's appeal before sufficient interaction feedback is available. We therefore use large language models (LLMs) to extract semantic signals from comment text and convert them into informative Bayesian priors that warm-start Thompson sampling under sparse early-stage feedback. To account for aggregate segment-level differences in response patterns, we maintain and update posteriors separately for each gender-age segment. In a real-world online A/B/C test, we compare a uniform prior with two LLM-based designs: a Gender Prior for demographic-affinity cues and a Content Prior for title-specific identity cues. The results show that LLM-based priors are most beneficial in sparse-feedback regimes -- with the largest gains emerging once a small amount of interaction evidence has accumulated -- and that prior design leads to distinct funnel-level effects. We further analyze prior-reward alignment and demographic heterogeneity, finding that click-oriented alignment is strongest for the Gender Prior and that treatment effects vary substantially across demographic segments. These findings suggest that LLM-derived priors can serve as a practical warm-start mechanism for text-rich bandit recommendation, while also revealing deployment trade-offs.

- type: 
- query: 

---

## 12. `2608.03067v1`

**Activation-Guided Neuron Intervention to Induce Alzheimer's-Related Computational Language Phenotypes in a Large Language Model**

Changes in spontaneous speech provide an early signal of cognitive dysfunction in Alzheimer's disease (AD) that large language models (LLMs) can detect. However, detection alone cannot establish whether the underlying model representations contribute functionally to behavior. We introduce an activation-guided intervention framework using Qwen3-8B. The framework identifies feed-forward neurons with higher activation rates for AD than control transcripts and modulates their output contributions during generation by scaling the corresponding down-projection weights. This yielded nine edited variants differing in intervention direction, magnitude, and scope. The original and edited models completed the same 12-turn neuropsychological battery, assessed through blinded human ratings and computational linguistic measures. Amplifying AD-associated neurons produced graded impairments in story recall, verbal fluency, working memory, procedural discourse, scene construction, and coreference resolution. Attenuation largely preserved performance and selectively improved several outcomes. Amplification also reduced lexical surprisal, idea density, syntactic complexity, and discourse quantity, broadly paralleling changes reported in human AD speech. These findings show that neurons identified solely from clinical language differences can influence behavior across multiple cognitive domains, providing proof of concept for an AD-related computational phenotype and a controlled framework for experimentally examining links between language and broader cognitive dysfunction.

- type: 
- query: 

---

## 13. `2608.01410v1`

**GenTrack: Physical Alignment for Robot-Native Motion Generation and Zero-Shot Humanoid Tracking**

General-purpose humanoid trackers can execute diverse references, but their zero-shot coverage depends on large embodied corpora that are costly to extend. Text-to-motion generators offer scalable supervision, yet models trained on human motion or retargeted data inherit a gap between kinematic plausibility and robot executability. Existing one-way pipelines fix either the generated corpus or the reward tracker. We introduce GenTrack, an online generator--tracker framework that alternates execution-grounded, group-relative generator alignment with tracker training on newly generated references; anchoring and rehearsal constrain drift. On Unitree G1, we evaluate GenTrack with ProtoMotions and SONIC backbones across three zero-shot tracking splits including public AMASS and LAFAN benchmarks, and a private out-of-distribution test set of 1,024 prompt-motion pairs in the wild. The online co-training strategy consistently produces generators that output more robot-executable motions with strong semantic alignment, and trackers with markedly broader zero-shot coverage and improved tracking accuracy, especially on out-of-distribution references. These results demonstrate that joint online post-training effectively narrows the executability gap between retargeted references and robot-native motion, advancing zero-shot humanoid control without additional data collection and beyond the limitations of a static reference pool.

- type: 
- query: 

---

## 14. `2608.00663v1`

**Geometry-guided Emotion Modulation for Controllable and Photorealistic Emotional Talking Face Generation**

Audio-driven emotional talking face generation aims to synthesize realistic videos with expressive facial dynamics. However, existing methods struggle to balance controllability and visual fidelity. Although implicit representations capture rich semantics, they lack structural guidance, often resulting in averaged emotional expressions. In contrast, explicit geometric methods offer better control over facial expressions but tend to sacrifice high-frequency texture details. To address it, we propose GemTalk, a diffusion-based framework that combines the semantic richness of implicit representations with the structural precision of explicit geometric priors. We introduce a Vision-guided Audio Emotion Projection (V-AEP) module to extract implicit emotional lip and expression features. At the same time, a Diffusion-based Geometric Priors Generator (D-GPG) generates identity-aware blendshape coefficients as explicit structural priors. Crucially, our Geometry-guided Emotion Modulation (GEM) module leverages these geometric priors to recalibrate the magnitude of implicit features, enabling precise, continuous control over emotional expressions, especially emotion intensity, without sacrificing visual quality. Extensive experiments show GemTalk achieves superior performance in photo-realism, and facial emotional dynamics.

- type: 
- query: 

---

## 15. `2608.01660v1`

**Ground, Cover, and Refine: Evidence-Centric Frame Selection for Long-Video Question Answering**

Long-video question answering requires identifying sparse yet critical evidence from videos containing thousands of frames under a constrained visual-token budget. Existing methods either select query-aware frames in a single pass or rely on timestamped text solely as retrieval guidance, leading to two key limitations. First, selected frames tend to cluster around local relevance peaks, and once the budget is exhausted, omitted evidence cannot be recovered. Second, textual and visual evidence remain weakly aligned. We propose GCR, a training-free framework that casts fixed-budget frame selection as a joint evidence curation problem. Ground converts timestamped text into temporal events, selects query-relevant real frame anchors, and renders each event text onto its temporally aligned frame. Cover supplements grounded events with direct visual anchors for complementary visual evidence and applies global maximal marginal relevance to preserve diverse context. Refine revisits omitted temporal regions and replaces the weakest revisable context frame with a real-frame medoid---but only when the medoid offers greater evidence value. GCR maintains a fixed number of chronologically ordered frames and requires no VLM training or architectural modification. Experiments on LongVideoBench and Video-MME, across three 7B backbones and frame budgets of 8, 32, and 64, demonstrate consistent improvements in long-video QA. With the 7B LLaVA-OV backbone and 32 frames, GCR achieves 64.25% and 62.15% on the two benchmarks, outperforming the strongest reproduced baselines by 2.54 and 1.93 percentage points, respectively.

- type: 
- query: 

---

## 16. `2608.01320v1`

**Dense Language Generation Made Simple: Deterministic, Randomized, and Multi-Order Algorithms**

Language generation in the limit is a theoretical framework for studying how a generator can learn to produce new valid strings from a stream of positive examples. In this model, an adversary chooses an unknown language from a countable family and enumerates its elements in an arbitrary order, while the generator must eventually output only elements of the language that have not yet appeared in the enumeration. Reliable generation is thus formalized through two eventual guarantees: validity and novelty relative to the observed data. To further quantify the breadth of the generator's outputs, Kleinberg and Wei (FOCS 2025, STOC 2026) introduced lower density as a measure of output coverage. Given an order representing the importance or relevance of possible outputs, lower density is the asymptotic lower bound, as $n$ grows, on the fraction of the first $n$ elements of the target language that the generator outputs before they appear in the data. Kleinberg and Wei showed that $1/2$ is the optimal lower-density guarantee for deterministic algorithms. We develop a simple and unified framework for obtaining optimal lower-density guarantees. We first give a deterministic algorithm that recovers the optimal guarantee of $1/2$ with a significantly simpler analysis than prior work. We then demonstrate the flexibility of our framework through two extensions. First, against an oblivious adversary, randomization raises the optimal guarantee to $1-1/e$. Second, for any finite collection of orders, the optimal deterministic and randomized guarantees can be achieved simultaneously with respect to every order, so accommodating multiple notions of importance or relevance entails no loss in the optimal guarantee.

- type: 
- query: 

---

## 17. `2608.01021v1`

**Can Humans Dream of Electric Sheep? Human-Written Samples for Fine-Grained Vision-and-Language Hallucination Benchmarking**

In an age of rapid model turnover, how do we make hallucination evaluation more perennial? We explore whether human-written hallucination samples could take the place of model-generated hallucinations, in order to make benchmarking detection independent of particular models. To this end, we construct a dataset of 1,600 human-written samples, spanning four languages (Chinese, English, French, Italian), and 18,400 samples from five vision-and-language models, all annotated for hallucinations using a fine-grained span-level labeling scheme. We find that human-written samples result in higher agreement and allow greater control of dataset contents, while remaining distributionally similar to samples derived from vision-and-language samples and providing a reasonable portrayal of detection capabilities - suggesting that human data is a viable substitute for model-based hallucination benchmarks.

- type: 
- query: 

---

## 18. `2608.00713v1`

**Observatorio Lazaro: A self-populating database of anglicism usage in the Spanish press**

This paper describes Observatorio Lázaro, a language resource that monitors unassimilated lexical borrowings (predominantly English lexical borrowings or anglicisms) in the Spanish digital press. Since April 2020 the system has automatically processed the daily output of a collection of news outlets, detected borrowings with a neural sequence-labeling model, and made the results available through a public web interface and API. The result is a continuously updated diachronic database which, at the time of writing, records more than two million borrowings across 1.88 million articles and 993 million running tokens of text (2020-2026). The paper documents the resource: we describe the end-to-end pipeline (acquisition, detection, post-processing, storage and access), the data model and the terms of availability; we evaluate the resource through the detector's held-out performance (span-level F1=0.86 for the borrowing class), inter-annotator agreement on the training corpus (Cohen's kappa=0.91) and a manual precision audit of 1,000 spans from the deployed data; and we situate it with respect to Spanish borrowing lexicography, annotated borrowing corpora and neology-monitoring observatories. The data shows that unassimilated anglicisms are used in the Spanish press at a frequency of approximately two anglicisms per thousand tokens, and that this rate remains stable. Our statistical analysis over six years reveals that the anglicism vocabulary in Spanish behaves as an open and growing class, with 58.7% of its types attested only once (53.6% after correcting for detection precision), and that its density is highest in the fashion, technology and lifestyle sections and lowest in political and institutional news. The resource is intended to complement static borrowing dictionaries and one-off annotated corpora by providing a continuously updated record of borrowing in the Spanish press.

- type: 
- query: 

---

## 19. `2608.02691v1`

**Output-Aware Rotation for INT2 KV-Cache Quantization**

The key-value (KV) cache has become a major memory and bandwidth bottleneck in long-context large language model inference, making ultra-low-bit quantization increasingly important. However, existing rotation-based INT2 methods optimize cache statistics or proxy errors before the complete attention readout, even though the model is ultimately affected by the error propagated through attention and the output projection $W_O$. To address this mismatch, we propose \textit{OptR}, an output-aware rotation method that minimizes post-$W_O$ attention-output error. OptR decomposes the post-$W_O$ attention-output error into key- and value-induced terms and learns per-head orthogonal corrections through the full INT2 quantization and attention path. OptR further applies an attention-equivalent key reparameterization to reduce large channel-wise offsets without changing the softmax distribution. Across three models and five reasoning and coding benchmarks, OptR consistently improves both QuaRot and OSCAR and strengthens long-context retrieval, while preserving the paged KV-cache format with negligible inference overhead.

- type: 
- query: 

---

## 20. `2608.01395v1`

**Language Equality has a Price: A Systematic Investigation of Multi-turn LLM Performance for EU-24+**

We evaluate large language models (LLMs) as language agents playing goal-directed dialogue games in self-play across 30 languages: the 24 official EU languages plus six others. Unlike static or preference-based evaluation, this paradigm is multi-turn, reference-free and programmatically scored, and because the game mechanics are language-agnostic it extends to a new language by localising a fixed set of prompt and word-list files. Evaluating nine open-weight and commercial LLMs, we find that no open-weight model covers the EU-24 well: in every official language both commercial systems outscore every open-weight model, and the two weakest average below 40 points across the EU-24. The commercial systems stay ahead even in languages with four orders of magnitude less public web text, showing that linguistic parity is achievable, but not from public crawls alone. A model's home region lifts it without closing the gap: Chinese is the strongest of all 30 languages for two Chinese-developed models, yet the best Chinese score of any model belongs to a US commercial system. Coverage is also not parity of service. Pooled over models and languages, the median non-English language costs 31% more to run than English, and scores 10% lower.

- type: 
- query: 

---

## 21. `2608.00656v1`

**Simulation-Based Plate-Reverb Parameter Estimation from a Single Impulse Response**

We present a simulation-trained, non-iterative estimator for Task A of the 1st DAFx Parameter Estimation Challenge. Each unnormalized plate-reverb impulse response is summarized by amplitude, spectral, and decay descriptors, and an ensemble of tree regressors estimates the six target parameters in one pass. Across two independent synthetic validation sets, the normalized models outperform the training-set mean and an earlier raw-regression baseline. On a shared set, the final ensemble also outperforms a single run of the official default PSO at substantially lower inference cost. Since the official labels are hidden, parameter accuracy is measured on simulator-matched data, and the released responses support only audio-side consistency checks. The estimator returns point estimates without uncertainty.

- type: 
- query: 

---

## 22. `2608.03135v1`

**Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds**

Text-to-image diffusion models can generate individual concepts well, but they often omit or merge concepts incorrectly with multiple concepts. We trace these failures to an early coordination bottleneck: before denoising begins, prompt-conditioned attention may allocate different concepts to strongly overlapping spatial support, which can keep their attention coupled as denoising proceeds. This observation motivates treating compositional generation as a boundary-condition problem rather than repeatedly controlling the evolving trajectory. To this end, we propose Rectify-then-Diffuse (RTD), a training-free framework that rectifies the initial allocation once before standard denoising. Firstly, we propose Soft-Overlap Disentanglement (SOD), which converts normalized overlap between pilot concept maps into a differentiable and layout-agnostic separation objective. Secondly, we introduce Isotropic Gradient Rectification (IGR), which normalizes the SOD gradient and applies a bounded latent displacement with a consistent scale across prompts and initializations. Extensive experiments show that RTD achieves state-of-the-art compositional fidelity and robust gains. On the AE-Bench object pair subset, RTD improves BLIP-VQA by 45.8% and ImageReward by 19.6% over CO3 while running 2.3$\times$ faster. Code will be released at https://github.com/Z-yiwei/rectify-then-diffuse

- type: 
- query: 

---

## 23. `2608.00540v1`

**DiffuseAgent-MI: Distributionally-Grounded,Tool-Integrated Self-Evolving Agents for Faithful Visual Reasoning**

Tool-integrated vision-language agents have made remarkable progress on compositional and multi-step visual reasoning. Yet their outputs frequently exhibit unfaithfulness: the stated reasoning path diverges from the computation that actually produced the answer, undermining reliability in safety-critical applications. We present DiffuseAgent-MI, a self-evolving agent whose perceptual grounding is governed by a KL-minimal energy model over feature units, providing a distributional view of visual mechanistic interpretability. The agent learns an energy landscape that softly constrains generated samples to lie near the native prior conditioned on the chosen interpretable unit, closing the gap between the explanation and the internal representation. A verifier then supplies trajectory-level faithfulness rewards, and a repair branch re-conditions the energy when the verifier flags an unfaithful step. On GeoQA, SciVis, VQA-v2 and an in-house multimodal reasoning set, DiffuseAgent-MI improves accuracy by up to 5.1 points over prior self-evolving agents while more than doubling mutual-information faithfulness and human-interpretability agreement. Our analysis shows the energy term and the verifier are complementary: the former guarantees distributional faithfulness, the latter trajectory-level faithfulness, and only their combination closes both gaps.

- type: 
- query: 

---

## 24. `2608.03918v1`

**When and Where to Look: Adaptive Visual Evidence Scheduling for Efficient Long Video Understanding**

Efficient long-video understanding requires vision--language models (VLMs) to reason over a small number of frames selected as sparse visual evidence. Existing relevance-based methods rely on static one-shot selection with fixed frame budgets and candidate pools, while agent-based schedulers achieve adaptivity through costly multi-round reasoning and interactive search. We propose EcoFrame, a training-free framework for low-overhead query-adaptive visual evidence scheduling. EcoFrame leverages the VLM's inference feedback to determine when to increase the frame budget and where to search for additional candidate evidence. Specifically, entropy-gated budget scheduling uses output uncertainty to stop early when the current evidence is sufficient or progressively expand the frame budget otherwise. Meanwhile, attention-guided candidate proposal converts frame-level attention into a temporal prior, enabling dense local search in informative regions while preserving global coverage when attention is diffuse. Experiments on Video-MME, LongVideoBench, and MLVU demonstrate that EcoFrame achieves a better accuracy--efficiency trade-off across multiple VLM backbones. On Qwen2.5-VL, EcoFrame achieves an average accuracy of 64.4, surpassing BOLT at 63.5, while providing a $1.85\times$ speedup over AKS and BOLT. Compared with the agent-based A.I.R., EcoFrame maintains comparable accuracy with up to a $13.5\times$ inference speedup. Code will be available at https://github.com/AK-DREAM/EcoFrame.

- type: 
- query: 

---

## 25. `2608.01358v1`

**HopRefusalBench: Diagnosing Refusal Failures in Search-Augmented Agents for Multi-Hop Reasoning**

Search-augmented large language model agents are increasingly capable of solving knowledge-intensive tasks, but their behavior when a multi-hop question is fundamentally unanswerable remains poorly understood. Existing abstention benchmarks largely expose defects at the surface of single-hop queries and therefore cannot reveal failures that emerge only after valid intermediate reasoning and retrieval. We introduce HopRefusalBench, the first controlled benchmark of refusal within multi-hop search, comprising 889 unanswerable questions constructed from KILT-grounded entity paths. It crosses three causes of unanswerability (answer unknown, false premise, and underspecified context) with root, middle, and terminal topologies, making premise verification, intermediate-bridge validation, and terminal stopping separately observable. We further propose a final-outcome taxonomy spanning target-aware refusal, pseudo-refusal, hallucinated completion, and search-budget exhaustion, together with source-aware trajectory metrics for post-trigger continuation and token waste. Across ten frontier proprietary and open-weight models in search-augmented mode, the best model achieves a target-aware correct halting rate (TCHR) of only 42.9%. Root and middle items are consistently harder than terminal items, and all models attain their highest TCHR on false premises and their lowest on underspecified questions. Yet when pooled across categories, 84.7--98.4% of each model's explicit refusal-like responses identify the correct rationale, localizing the main bottleneck to committing to an appropriate non-answer; failed trajectories instead diverge into hallucination or search-budget exhaustion. These results establish refusal in multi-hop search as a consequential evaluation problem and provide a foundation for diagnosing and improving the reliability of search-augmented agents.

- type: 
- query: 

---

## 26. `2608.02826v1`

**Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model**

Reinforcement learning is a subfield of machine learning that studies how an agent interacts with an environment in order to extract as large a reward as possible. A standard approach to study such interaction is through Markov Decision Processes (MDPs) and the task of choosing an optimal policy --- a function that tells the agent which action to take. In this work, we study two types of MDPs --- finite-horizon and infinite-horizon discounted --- and propose new quantum algorithms for computing approximate optimal policies. Our quantum algorithms are based on a new combination of standard value iteration and quantum subroutines like quantum mean estimation and quantum maximum finding, overall enhanced with techniques from sample-optimal classical algorithms. Our resulting query complexities improve upon previous works, thus approaching already established quantum lower bounds.

- type: 
- query: 

---

## 27. `2608.03154v1`

**ANCHOR-RE: An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction**

Biomedical relation extraction (BioRE) extracts structured knowledge from biomedical literature for applications such as knowledge base construction and hypothesis generation. Traditional symbolic systems such as SemRep provide high precision but limited recall, while large language models (LLMs) offer stronger contextual reasoning but remain prone to false-positive predictions. We developed ANCHOR-RE, a framework that integrates ontology-guided reasoning, external knowledge grounding, and data-driven verification rules into LLM inference. We evaluated it on three BioRE benchmarks (SemRepGS, DDI, and ChemProt) using both proprietary and open-weight LLMs. To assess generalizability beyond benchmark datasets while reducing potential evaluation bias from LLM pretraining contamination, we conducted a temporal evaluation using 100 biomedical articles published in 2026. With the proprietary backbone, ANCHOR-RE outperformed direct LLM prompting, improving micro-F1 from 0.654 to 0.676 on SemRepGS, from 0.769 to 0.872 on DDI, and from 0.939 to 0.941 on ChemProt. On DDI and ChemProt, it also outperformed previously reported inference-only methods and approached fine-tuned or instruction-tuned systems without parameter updates. Similar performance gains observed with open-weight LLMs indicate that the benefits were not limited to the proprietary backbone. On the post-cutoff set, manual assessment of 500 randomly sampled predictions yielded a precision of 69%, maintaining consistent precision on previously unseen biomedical literature. Neuro-symbolic reasoning can improve the reliability of LLM-based BioRE without fine-tuning. Results across multiple benchmarks, model families, and post-cutoff literature support ANCHOR-RE as a practical training-free approach to biomedical literature mining.

- type: 
- query: 

---

## 28. `2608.02791v1`

**Better, Stronger, Faster, and Broader: Structured All-Mask Prediction for MLLM-Based Segmentation**

MLLM-based segmentation faces a core segmentation trilemma: high segmentation performance, preserved dialogue ability, and fast inference. Embedding-prediction methods may disrupt language modeling through pixel-level objectives, whereas next-token generation is inefficient for dense masks. We propose All-Mask Prediction, decoupling autoregressive dialogue from non-autoregressive mask prediction. Its binary instantiation, STAMP (Simultaneous Textual All-Mask Prediction), emits an in-vocabulary <SEG> trigger, fuses image-aligned mask tokens with corresponding patch features, and uses hybrid attention to classify all tokens as foreground or background in one pass. It thereby combines strong referring and reasoning segmentation with preserved multimodal ability and efficient inference. However, binary masks cannot retain multiple semantic or instance identities without repeated target-specific predictions. We therefore propose Structured All-Mask Prediction and develop STAMPlus. It generates a target list with explicit IDs and optional boxes, binds these IDs to a shared multi-class mask space, and jointly predicts all targets in one non-autoregressive pass. A single unified checkpoint retains STAMP's referring and reasoning capabilities while extending to open-vocabulary semantic, instance-aware, and remote-sensing small-target segmentation, where high-resolution mask-token scaling preserves finer spatial evidence. Across these settings, STAMPlus achieves state-of-the-art segmentation performance, preserves general multimodal instruction following, and reduces 12-category latency from 13.50s for repeated STAMP inference to 5.16s. Further analyses show that accurate target cues improve segmentation and learned spatial grounding benefits look-twice reasoning. Overall, STAMPlus resolves the trilemma beyond single-target prediction.

- type: 
- query: 

---

## 29. `2608.01160v1`

**Differentiable Lifting for Topological Neural Networks**

Topological neural networks (TNNs) enable leveraging high-order structures on graphs (e.g., cycles and cliques) to boost the expressive power of message-passing neural networks. In turn, however, these structures are typically identified a priori through an unsupervised graph lifting operation. Notwithstanding, this choice is crucial and may have a drastic impact on a TNN's performance on downstream tasks. To circumvent this issue, we propose $\partial$lift (DiffLift), a general framework for learning graph liftings to hypergraphs and cellular- and simplicial complexes in an end-to-end fashion. In particular, our approach leverages learned vertex-level latent representations to identify and parameterize distributions over candidate higher-order cells for inclusion. This results in a scalable model which can be readily integrated into any TNN. Our experiments show that $\partial$lift outperforms existing lifting methods on multiple benchmarks for graph and node classification across different TNN architectures. Notably, our approach leads to gains of up to 45% over static liftings, including both connectivity- and feature-based ones.

- type: 
- query: 

---

## 30. `2608.03577v1`

**Looking under the Wrong Lamppost: On the Limitations of Automated Translation Quality Estimation**

Automation of Translation Quality Estimation (QE) has emerged as a widely discussed approach to managing translation quality at scale, and a growing number of tools and technologies have been released in pursuit of this goal. However, the proliferation of new QE systems has not always been accompanied by robust, transparent, and reproducible research and testing. This gap deserves critical scrutiny. This paper examines some fundamental limitations of the QE technology from both theoretical and empirical perspectives, arguing that current QE systems are structurally ill-equipped to serve as reliable standalone tools in real-world translation workflows. The reviewed evidence suggests that QE suffers from a range of interrelated and largely unresolved limitations. Most fundamentally, the evaluation of the quality of translation at the level of isolated segments is problematic because it tends to miss out on cohesion, coherence, and stylistic and rhetorical text features. In addition, empirical research documents several other limitations and flaws, including failure to generalize, systematic biases, overfitting and distribution collapse, performance gaps, error annotation challenges, and data scarcity. These are structural limitations arising from the complexity of human language and translation as a cognitive and communicative act - limitations that more data and better architectures have so far not overcome. Consequently, segment-level QE scores should not be used as a standalone basis for routing, release, or review bypass in production; we argue future work should focus on automating human evaluation grounded in MQM.

- type: 
- query: 

---

## 31. `2608.01338v1`

**Driver2Map: Imitating Human Driving for Online High-Definition Map Construction**

High-definition (HD) maps are essential for autonomous driving systems. In constructing such maps, onboard multi-view camera images, standard-definition maps and satellite images provide crucial information. However, due to the modality and perspective differences among these data sources, existing methods often struggle to effectively align and fuse them, making online HD map construction still challenging. To address these issues, we propose Driver2Map, an online HD map construction model inspired by human drivers. Unlike existing HD map construction models that utilize only two modalities, our Driver2Map can simultaneously exploit three modalities. Specifically, we propose a "two-stage alignment" strategy to reduce spatial misalignment across different modalities. Additionally, we introduce "Pose-Guided BEV Fusion", a BEV (bird's-eye-view) generation module that leverages camera pose information to adaptively weight multi-view features, thereby effectively suppressing cross-view feature overlap during BEV generation. Also, we design a "Pretrained Prior for Map Refinement" module to refine the initial prediction by learning map structure priors, thus improving the HD map prediction under dynamic occlusions. Extensive experiments demonstrate that Driver2Map outperforms existing methods on both IoU and AP metrics.

- type: 
- query: 

---

## 32. `2608.03038v1`

**Beyond Accuracy: A Multidimensional Evaluation of Statistical Reasoning in Large Language Models**

Statistical reasoning is multidimensional, yet evaluations of large language models (LLMs) typically emphasize response accuracy while overlooking how models construct and communicate statistical explanations. This study demonstrates the value of a multidimensional evaluation by combining response accuracy, response behavior, structural topic modeling, and lexical similarity analysis. The framework is applied to explanations generated by 15 current-generation LLMs responding to 90 questions drawn from four statistics examinations spanning high school, undergraduate, and graduate levels. Accuracy varied substantially across models, ranging from 55\% to 78\%. In contrast, structural topic modeling revealed a common conceptual organization of statistical reasoning across all models, while lexical similarity analysis identified modest but consistent vendor-specific differences in explanatory style. Models developed by the same vendor (e.g. Anthropic, OpenAI) produced explanations that were slightly more similar than models from different vendors. These findings demonstrate that statistical reasoning in contemporary LLMs cannot be characterized by accuracy alone and illustrate how complementary analyses of response behavior and model-generated explanations provide a more comprehensive evaluation of statistical reasoning in generative AI.

- type: 
- query: 

---

## 33. `2608.01775v1`

**Multi-Source Dynamic Graph Learning for Compound-Flood Forecasting in Managed Coastal Systems**

Compound flooding in managed coastal systems is influenced by hydrological conditions and water-management activity observed across multiple monitoring stations. Current forecasting models can capture temporal dependencies with low average errors, but global error metrics may conceal poor reproduction of prolonged high-water plateaus that are relevant to flood early warning. Because hydrometeorological and operational signals are distributed across heterogeneous gages, single-site records do not fully represent high-water dynamics. Nevertheless, unconstrained fusion of cross-site signals can degrade the stability of local temporal forecasts. This work proposes an anchored forecasting framework that incorporates cross-site information through state- and lead-dependent bounded residual corrections. A multi-source regime representation constructed from hydrometeorological and operational observations adaptively calibrates inter-site relationships and correction scales, enabling targeted cross-site adjustment while preserving the local temporal forecast as a stable anchor. Beyond conventional global error statistics, we evaluate event-scale high-water characteristics through the temporal alignment of forecasted and observed high-water processes. Experiments demonstrate that selectively integrating multi-station dynamic conditions improves the prediction reliability of sustained high-water plateaus while maintaining high accuracy during routine hydrological conditions, supporting flood early warning and water-management decision support.

- type: 
- query: 

---

## 34. `2608.03855v1`

**Bi-semantic Chemical Embedder for Joint Representation Learning of SMILES and Natural Language**

Transformer models have revolutionized natural language processing (NLP), and text-based molecular representations like SMILES have successfully extended these architectures to chemistry. However, domain-adaptive pre-training often causes models to overfit to chemical syntax, catastrophically forgetting their foundational semantic capabilities. To address this challenge, we introduce CheMatE, a chemistry-oriented embedding model that jointly captures molecular structure and domain-specific natural language within the same representation space. Built on a ModernBERT backbone, CheMatE learns bi-semantic representations through a two-stage training procedure: continued masked language modeling (MLM) followed by a Matryoshka contrastive learning stage via Multiple Negative Ranking Loss (MNRL). First, we train the model using MLM on a novel, large-scale corpus of SMILES-annotated, long-context scientific documents that were constructed and curated from FineWeb and ChemPile (comprising 10.4B and 11.5B tokens, respectively). Subsequently, the model undergoes contrastive learning using a synthetic dataset of SMILES-text pairs algorithmically derived from our original training corpus. This design exposes the model to SMILES-enriched scientific literature, enabling bi-semantic understanding. We evaluate CheMatE across a range of downstream tasks covering molecular property prediction and scientific language understanding. Our results demonstrate that coupling our custom-curated datasets with this sequential training strategy yields robust, highly transferable representations. By effectively unifying structural and contextual signals within a single text-based framework, CheMatE achieves competitive performance across both specialized chemistry models and general-purpose language model baselines.

- type: 
- query: 

---

## 35. `2608.03086v1`

**Automatic Patient-Specific Microwave Ablation Planning Accelerated by a Physics-Guided Deep Learning Model**

Microwave ablation (MWA) is a promising minimally invasive treatment for liver tumors, but its therapeutic outcome strongly depends on patient-specific planning of antenna insertion trajectory, power, and treatment duration. Accurate numerical simulation can provide physically reliable ablation predictions; however, its high computational cost limits its use in optimization-based planning, where repeated forward evaluations are required. To address this issue, we propose a digital twin-based automatic planning framework that combines a neural ablation prediction model with a genetic algorithm. The model was trained on multiphysics simulation data generated from patient-specific tumor and vessel structures, antenna configurations, and treatment conditions, and was used as a fast forward model during planning. The prediction model achieved a Dice score of 95.1%, enabling accurate deep learning-based optimization. In 13 unseen planning cases, the proposed method improved ablation efficiency by 54.3% and reduced organ damage by 55.0% compared with clinician-defined planning, while slightly shortening the insertion path length by 3.3%. Most generated plans were also judged clinically applicable by MWA specialists. Furthermore, the framework enabled approximately 420-fold faster planning than numerical-simulation-based planning, demonstrating its potential as a fast digital twin for quantitative and personalized MWA treatment planning. The code is available at: https://github.com/SeonAengCho/MWA-Planning.git

- type: 
- query: 

---

## 36. `2608.03617v1`

**A machine-readable catalogue of the Tsiolkovsky papers (fond 555, Archive of the Russian Academy of Sciences), and a way to measure how well its handwriting can be read**

The personal archive of Konstantin Tsiolkovsky (1857-1935) is held as fond 555 of the Archive of the Russian Academy of Sciences. The archive scanned the fond and published the images, but with no queryable catalogue, no full-text search and no dataset: the holdings can only be browsed one page at a time. This paper describes a machine-readable catalogue of all 2,019 files and 51,008 scans, a dating for 1,969 files taken from the archive's own descriptions, a page-level classification of every scan into handwriting and typescript, and a growing corpus of machine transcriptions (currently 322 files, 5,454 scans). It also reports a way to measure handwritten-text-recognition accuracy in an archive with no ground truth. Archives of the typewriter era often preserve one text twice, as manuscript and as a typed copy; transcribing both and comparing isolates the reading error, since source and pipeline are identical and only page difficulty differs. Across 294 such pairs from 27 files, two readings of a handwritten page agree on a median 37% of words. On two files that also have a published edition the estimate can be checked against ground truth: it is unbiased to within a percentage point and ranks pages as the truth does (rank correlation 0.92 where the edition is a faithful witness). This bounds use: two variants of one work here share 19% of words, below the rate at which two readings of a single page agree, so the redactions cannot be collated word by word at this quality. That negative result is reported as such, and the constraint is built into the tool.

- type: 
- query: 

---

## 37. `2608.02778v1`

**Neural Networks with Local Converging Inputs for Efficient Options Pricing Models**

We present a novel application of Neural Networks with Local Converging Inputs (NNLCI) to improve the efficiency of existing numerical methods for pricing multi-asset options. The most concise input format for NNLCI has been introduced, offering substantial convenience and efficiency. NNLCI uses a neural network to locally correct solutions from a coarse mesh and a refined mesh (relative to the coarse one), requiring only a minimal amount of high-fidelity training data. We demonstrate this approach on cash-or-nothing options under the Black-Scholes equation in one, two, and three spatial dimensions, and on single-asset down-and-out barrier call options under the Heston stochastic-volatility model (whose pricing PDE is two-dimensional in the spot price $S$ and the instantaneous variance $v$). In each case, NNLCI reduces the root-mean-square error (RMSE) of the refined-mesh numerical solution by a factor of approximately 4-12 on test sets, even when the neural network is trained on only a small subset of parameter combinations. These results demonstrate that NNLCI significantly reduces computational requirements for high-dimensional problems in real-time options trading and risk management, offering low training costs and strong generalization ability.

- type: 
- query: 

---

## 38. `2608.00667v1`

**Band-Count Dense Modal Estimation with Fixed-Frequency Differentiable Resonator Refinement**

Task B of the 1st DAFx Parameter Estimation Challenge requires estimating the frequencies, decay rates, gains, and number of modes in a dense plate-reverb impulse response. Weak and overlapping modes make sparse peak detection prone to severe undercounting. We train an ExtraTrees regressor on simulator-generated data to predict mode counts in four frequency bands. These counts define dense frequency grids, after which a differentiable all-pole resonator model refines decay and gain while keeping frequency fixed. On two separate synthetic validation sets, the system reduces a local challenge-style error by about 66% relative to the official default peak-picking baseline. The improvement is mainly associated with lower mode-count mismatch, while decay and gain remain the largest error sources. These findings support separating modal-density estimation from continuous parameter fitting.

- type: 
- query: 

---

## 39. `2608.00881v1`

**AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving**

Large language model agents increasingly act through stateful tools, yet model generation and environment execution remain serialized at every step. As decoding accelerates, tool execution becomes a growing bottleneck. Existing action- or observation-only speculation leaves much of this latency exposed: value is concentrated in a few slow calls, some outcomes emerge only through execution, and longer lookahead typically requires an increasingly unlikely chain of action predictions. We present AOSpec, a lossless framework that co-speculates actions and observations across the full agent-environment loop. Expected Value Decoding (EVD) directs observation speculation toward outcomes with the greatest expected latency benefit, optimizing expected time hidden rather than hit rate. For outcomes only execution can reveal, AOSpec launches latency-critical target actions in isolated forks that contain their effects, while Joint Action-State Verification (JASV) verifies both the action and its origin state against committed execution before reuse. JASV recasts long-horizon action dependency from full-chain prediction into target action-state verification, breaking the lookahead--accuracy tradeoff and unlocking long-range overlap without sacrificing serial semantics. Across Terminal-Bench serving settings spanning four harnesses, five actor models, and five serving speeds, AOSpec outperforms every practical baseline, reducing mean end-to-end latency by 11.8-32.5% and p99 latency by up to 42.8%. Its gains increase as decoding accelerates, and its observation model transfers from Terminal-Bench to SWE-bench Verified without retraining.

- type: 
- query: 

---

## 40. `2608.03447v1`

**Approximate Speculative Decoding**

Speculative decoding accelerates autoregressive generation by verifying a draft block with a target model in parallel. Under standard greedy verification, decoding stops at the first draft token that differs from the target argmax, discarding the remaining target-scored suffix. Although accepting such a mismatch changes the decoding trajectory, it can make a contiguous suffix reusable when its tokens remain target-greedy under the realized prefix. In this paper, we introduce \textbf{Approximate Speculative Decoding (ASD)}, a training-free verifier that replaces binary first-mismatch truncation with budgeted longest-prefix selection. ASD accepts selected mismatches subject to a local target-logit regret gate, a per-block exception cap, and a persistent request-level regret budget, then reuses the contiguous target-greedy suffix without additional approximate decisions or target-model forward passes. ASD requires neither a new draft model nor fine-tuning, and exactly reduces to standard greedy verification when the budget is zero. Experiments show that ASD improves fixed-workload throughput by $3.05\%$--$15.26\%$ over matched strict verification and averages a $7.78\%$ gain across seven Qwen3-14B + DSpark-14B tasks. On DeepSeek-V4-Flash (284B) with DSpark it also raises verifier-side acceptance by roughly $10\%$--$16\%$ on GSM8K and MATH-500 in an FP4-to-FP8 compatibility setting. The source code is publicly available at: https://github.com/Kissmetothemoon/ASD

- type: 
- query: 

---
