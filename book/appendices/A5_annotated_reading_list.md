# Appendix E: Annotated Reading List
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A4\_the\_rag\_card\_and\_index\_datasheet](A4_the_rag_card_and_index_datasheet.md) · [A6\_notation\_quick\_reference](A6_notation_quick_reference.md)
Use this curated, chapter-mapped reading list to connect each Retrieval-Augmented Generation (RAG) paper with the exact result a senior interview candidate should be able to reproduce.
## TL;DR
- The list contains $`161`$ entries in $`14`$ clusters and $`52`$ core markers, covering $`160`$ distinct titles; Cumulated Gain-Based Evaluation appears in two clusters. The source estimates roughly $`500`$ cited works overall, a broader count not independently re-enumerated here.
- Learn the result together with its model, task, baseline, assumptions and uncertainty. An annotation is not a universal theorem.
- Preserve the supplied paper titles, author strings and priority markers; chapter links guide interpretation and do not replace the primary paper.
- Negative results are conditional evidence about failure modes, not guarantees that a method always fails.
- Read the ten-paper starter route for orientation, then follow the clusters relevant to the actual role or system.
## The story
Imagine fourteen shelves containing $`161`$ reading cards for $`160`$ distinct titles. One evaluation paper appears twice because it serves two chapter clusters. The source’s roughly five hundred cited works describe the wider bibliography, not a newly verified unique-work count in this appendix.
Fifty-two cards carry a `[core]` seal. The seal does not mean that the other cards are unimportant. It means a senior visitor should explain the sealed card without asking the curator for a hint.
The curator refuses title collecting. If a visitor says “Lost in the Middle,” the curator asks what baseline it used and what the comparison showed. A title is only a shelf label. The annotation is the knowledge that must leave the room with the visitor.
Several shelves hold uncomfortable findings. Retrieval can lower accuracy. Semantic chunking gains may not transfer. Automatic citation judges can disagree with people. Self-critique without outside feedback may not improve an answer. The curator keeps these cards near the front because good interviews probe limits, not just success stories.
Some cards seem to belong in neighboring libraries. Distance concentration, tail latency, and instruction hierarchies are not retrieval papers. The curator keeps them because those constraints still govern the RAG reading room.
A visitor with limited time follows the ten-card route. It introduces the book’s design decisions, selected negative results, index arithmetic and a threat model; it does not certify complete preparation for every senior interview.
## Decoder table


| Technical term | Plain-English meaning | Why it matters in this appendix |
| --- | --- | --- |
| Annotated reading list | A bibliography where every entry includes the claim to learn | The interview target is the result, not title recognition |
| Curated | $`161`$ entries; $`160`$ distinct titles | The broader roughly five-hundred-work estimate is separate; one title appears twice |
| Chapter-mapped cluster | A group aligned with a span of book chapters | The 14 groups provide a study order |
| `[core]` | Priority marker for an entry a senior candidate should discuss unprompted | It identifies the highest-priority readings |
| Claim to know | The specific result, mechanism, comparison, or limit attached to an entry | It prepares the reader for a staff-level follow-up |
| Negative result | Evidence that a method fails, hurts, or does not transfer under stated conditions | The appendix deliberately emphasizes these results |
| NLP | Natural language processing | Field named in paper titles |
| LLM | Large language model | Model class named in titles and annotations |
| BERT | Bidirectional Encoder Representations from Transformers | Model family used in representation and reranking papers |
| QA | Question answering | Task family, not one fixed evaluation protocol |
| IR | Information retrieval | Field encompassing ranking, retrieval and evaluation |
| Parametric knowledge | Knowledge carried in model weights | The first cluster compares it with retrieved knowledge |
| Non-parametric knowledge | Knowledge supplied outside model weights | It motivates retrieval and external datastores |
| RAG-Sequence | A marginalization named in the original RAG paper | The source says it differs from RAG-Token |
| RAG-Token | A second marginalization named in the original RAG paper | The distinction is conceptual, not an implementation detail |
| Long-tail knowledge | Facts involving infrequent entities/co-occurrences | Empirical accuracy/co-occurrence relations depend on the studied model and data |
| Tool use | Letting a language model invoke an external capability | Toolformer frames retrieval as one instance of it |
| Agentic retrieval | Retrieval interleaved with an acting or reasoning loop | ReAct supplies the reason-act pattern |
| Naive, advanced and modular RAG | A taxonomy of design patterns | Not a certification of maturity or an inevitable quality ordering |
| Model editing | Changing factual behavior by changing weights | Cluster 2 compares editing with retrieval and continual learning |
| Causal tracing | Interventions measuring internal causal influence | Does not identify an isolated weight location owned by each fact |
| GPT | Generative pre-trained transformer | A model-family label, not one fixed architecture/configuration |
| MLP | Multilayer perceptron | Feed-forward components studied by editing/circuit analyses |
| ROME | The editing method named by the source | Its causal-tracing result is the claim to reproduce |
| MEMIT | The scaling answer to ROME named by the source | Batch editing can degrade unrelated knowledge |
| EWC | Elastic weight consolidation | A parameter-importance-weighted retention method, not a zero-forgetting guarantee |
| Parameter drift | Movement of model weights during learning | Continual learning must limit it to reduce forgetting |
| Differentiable Search Index | A model generates document identifiers | Identifier decoding, validity, updates and evidence access still cost work |
| Generative information retrieval (GenIR) | Retrieval that generates identifiers or results | The position paper frames its relation to RAG |
| Memorization | Training-sequence information expressed by a model | Three reported empirical trends are not unrestricted probability laws |
| Divergence attack | Extraction mechanism named in an annotation | The candidate should know how it works, not merely that it exists |
| Deduplication | Remove repeated training text | Can improve extraction risk and quality; effects are recipe dependent |
| Intrinsic hallucination | One side of the hallucination taxonomy | The survey distinguishes it from extrinsic hallucination |
| Extrinsic hallucination | The other side of the hallucination taxonomy | The reader must know where each type originates |
| Faithfulness | Agreement with supplied evidence | An answer can be faithful to a wrong document |
| Factuality | Truth of the answer | It must remain separate from faithfulness |
| Irrelevant context | Retrieved material unrelated to the question | One core result shows it can reduce accuracy |
| Co-occurrence | How often entities appear together | It helps explain factual substitution errors |
| Sampling-based consistency | Agreement across sampled answers | SelfCheckGPT uses it as a hallucination signal at a cost |
| Confidence-based gating | Using confidence to decide a system action | Partial calibration provides both its foundation and limit |
| Nonparametric datastore | Keyed external content and serving structures | Addressable removal is not legal immunity for acquisition or copying |
| Training compute proxy | $`C\approx6ND`$ | A scoped dense-model FLOP estimate, not a full-system invoice |
| $`C`$, $`N`$, $`D`$ | Training FLOPs, parameters, cumulative processed tokens | Processed tokens differ from distinct corpus tokens |
| Compute-optimal training | Allocate a specified training budget across model and data | A training optimum need not minimize lifetime training-plus-serving cost |
| In-context learning | Behavior induced by examples in the prompt | The source stresses that it is conditioning, not a weight update |
| Content-free calibration | Calibration using inputs without task content | It addresses ordering sensitivity in few-shot prompts |
| Key-value (KV) head | An attention-head quantity named in the Llama 3 annotation | It enters the book’s memory arithmetic |
| Low-Rank Adaptation (LoRA) | A parameter-efficient fine-tuning method | Its rank-$`r`$ parameter count changes the fine-tuning-versus-retrieval calculation |
| $`r(d_{\mathrm{in}}+d_{\mathrm{out}})`$ | Two low-rank trainable factors for one matrix | Input/output dimensions and rank; excludes biases and base-path compute |
| Residual stream | The shared information path through a transformer | The circuits paper treats it as an information bus |
| $`QK`$/$`OV`$ factorization | Two circuit roles named in the source | One role decides where to look and the other what to move |
| Induction head | A mechanism for some context-dependent copying | Not a complete explanation of every RAG behavior |
| Integrated gradients | The attribution method named in the source | It bounds what feature attribution can claim |
| Attention map | Attention weights for specified layers/heads | Alone, not a causal grounding certificate |
| Chain-of-thought | A reasoning scaffold written through intermediate steps | The prompting cluster builds from it |
| Self-consistency | Sample paths and aggregate answers | Work, latency and error dependence must be measured separately |
| Tree of Thoughts | Search over reasoning states | Added scaffolding eventually stops paying |
| Prompt sensitivity | Output variation caused by prompt changes | POSIX turns it into a measurable engineering variable |
| Expected calibration error | The calibration quantity named in the source | Larger models are not automatically better calibrated |
| Semantic uncertainty | Uncertainty over estimated meaning groups | Clustering can err and consistent answers can all be wrong |
| Semantic entropy | Deployable uncertainty measure based on semantic groups | It operationalizes the same idea |
| Verbalized confidence | A black-box model stating its confidence | The source asks how far it can be trusted |
| Self-correction | A model critiquing and revising its own answer | Without external feedback, it does not reliably improve reasoning |
| Model collapse | Degradation under recursive synthetic-data training | Not automatically the same as inference-time self-reflection |
| Static embedding | Context-independent word representation | Objective differs from relevance training; usefulness is task dependent |
| Cross-encoder | Joint pair encoding/scoring | A general score is not reproduced by one query-independent document vector |
| Siamese encoder | Independent encoding, commonly with shared weights | Bi-encoder towers need not always share weights |
| Alignment and uniformity | The contrastive-learning framing named for SimCSE | It explains the role of dropout augmentation |
| Massive Text Embedding Benchmark (MTEB) | An embedding benchmark named in the source | Leaderboard position depends on the task |
| Anisotropy | Directional concentration in a representation space | Its effect on raw cosine depends on the embedding and task |
| Retrieval granularity | The size and structure of one retrieval unit | Dense X Retrieval treats it as a first-class decision |
| Semantic chunking | Choose boundaries using meaning-related signals | Audit real-document transfer rather than assuming all gains or failures generalize |
| Granularity ladder | Retrievable recursive summaries and source units | May improve global queries; does not prove flat retrieval is incapable |
| WindowDiff | A text-segmentation evaluation metric | The source notes the boundary errors it handles |
| $`P_k`$ | A segmentation metric named in the source | It under-penalizes certain boundary errors |
| Layout-aware input | Treating page layout as model input | LayoutLM makes layout first-class rather than preprocessing |
| Skip list | A probabilistic ordered data structure | Its expected-search derivation leads into HNSW |
| $`L(n)/p+1/(1-p)`$ | A skip-list expected-search bound form | State promotion and cost conventions; not an HNSW search bound |
| Hierarchical Navigable Small World (HNSW) graph | A hierarchical graph index for approximate neighbors | The source requires its constants, heuristic, and memory formula |
| $`m_L=1/\ln M`$ | Common HNSW level-multiplier convention | Require M greater than one; implementation conventions may differ |
| $`M_0=2M`$ | Common allocated base-layer link capacity | Capacity is not necessarily actual degree |
| $`4d+8M`$ bytes | fp32 vectors plus four-byte base-layer link slots | Scoped subtotal; omits upper layers, IDs, allocator and runtime |
| Navigable Small World (NSW) graph | The flat graph predecessor to HNSW | Its limitations motivate the hierarchy |
| Product quantization | At most $`k^m`$ combinations from $`km`$ subcentroids | Equal subspaces contain $`kd`$ stored scalar centroid values |
| Optimized product quantization | Product quantization after a learned rotation | The extra rotation can be worth its cost |
| FAISS | A library for similarity search and clustering | Select a concrete index, metric, precision and memory scope |
| SSD | Solid-state drive | DiskANN combines storage tiers; count both disk and RAM requirements |
| GPU | Graphics processing unit | Distinguish peak throughput from achieved service work |
| RAM | Random-access memory | Specify process, index, payload and runtime scope |
| Tombstone | A deletion marker in an index | FreshDiskANN should be read with deletion material |
| SimHash | A similarity hash named in the source | Its rationale comes from locality-sensitive hashing collision probability |
| LSH | Locality-sensitive hashing | Near/far collision guarantees require stated parameters and assumptions |
| Distance concentration | Reduced distance contrast under specified conditions | Not proof that exact search always dies or ANN is universally necessary |
| Approximate nearest neighbor (ANN) benchmark | A benchmark for approximate vector search | The reporting convention is a recall-versus-QPS curve |
| QPS | Queries per second | Benchmark at stated latency, recall, load, filter and hardware |
| Matryoshka representation | Trained nested representations | Truncation is not automatically useful for arbitrary embeddings |
| Predicate-agnostic search | Filtered search without one index per predicate | ACORN addresses over-filtering |
| Fan-out | Issue component requests across sources or shards | Tail effects depend on dependence, load and completion policy |
| $`p_{99}`$ | The tail-latency percentile named in The Tail at Scale | The source links it to fan-out in a sharded index |
| BM25 | The lexical ranking method named in the source | The reader must know where $`k_1`$ and $`b`$ enter |
| IDF | Inverse document frequency | State smoothing and corpus-count conventions |
| Dense Passage Retrieval (DPR) | Dense retrieval trained with in-batch negatives | Its bi-encoder efficiency is a core argument |
| Approximate Nearest Neighbor Negative Contrastive Learning (ANCE) | Dense-retriever training with hard negatives | The index must refresh during training |
| Contriever | The unsupervised retriever named in the source | It tests how far retrieval goes without labeled pairs |
| BEIR | Heterogeneous zero-shot retrieval benchmark | Its observed method ordering does not establish a universal ranking |
| Sparse Lexical and Expansion Model (SPLADE) | Learned sparse retrieval with implicit expansion | FLOPS regularization makes sparsity trainable |
| Masked language modeling (MLM) head | The prediction head used for masked-token training | SPLADE places implicit expansion there |
| FLOPS regularization | A differentiable sparsity/posting-work proxy | Not a direct measurement of accelerator FLOPs or latency |
| ColBERT late interaction | Token-level query-document interaction delayed until scoring | Its per-token storage cost motivates ColBERTv2 |
| MaxSim | The late-interaction operator named in the source | It is part of the ColBERT claim to know |
| Cross-encoder reranker | A second-stage model that scores query-document pairs jointly | Passage Re-ranking with BERT gives the simplest form |
| monoT5 | Sequence-to-sequence relevance classifier/reranker | Uses true/false two-token normalization, not the raw true logit |
| LambdaRank | Metric-weighted pairwise pseudo-gradients | Not necessarily an integrable global metric-gradient field |
| nDCG | Normalized discounted cumulative gain | State gain, discount, cutoff, judged set and zero-IDCG policy |
| RRF | Reciprocal rank fusion | Often $`c_{\mathrm{RRF}}=60`$ empirically; not universal context depth |
| Relevance feedback | Updating a query from judged results | Rocchio is the ancestor of modern reformulation |
| Pseudo-relevance feedback | Treating top results as if they were relevant | Unsupervised use can drift |
| HyDE | Searching with a generated hypothetical answer | The source supplies the mechanism but does not expand the name |
| doc2query | Predict likely queries for source documents | Expansion generation is incurred at ingestion and refresh |
| Reinforcement learning | Training against a reward signal | One reformulation paper optimizes retrieval reward |
| FLARE | Active retrieval triggered by confidence | The source supplies the mechanism but does not expand the name |
| Self-RAG | Retrieval, generation, and critique controlled by reflection tokens | Its critic is distilled into the generator |
| Corrective Retrieval Augmented Generation (CRAG) | A pipeline that evaluates retrieval quality and can fall back to web search | The evaluator has cost and fragility |
| Adaptive-RAG | A complexity-routed retrieval design | Its router learns from silver labels |
| IRCoT | Interleave retrieval and reasoning | Can help later-hop evidence; not every compositional question requires it |
| Compositionality gap | The gap that multi-hop retrieval tries to close | Self-Ask gives it a crisp definition |
| GRPO | Group relative policy optimization | Specify group sampling, rewards and the actual training implementation |
| Fusion-in-Decoder (FiD) | Encoding passages separately and combining them in decoding | Its benefit creates decode cost |
| REPLUG | Retrieval augmentation with a frozen black-box LM | Retriever learning needs likelihood access, not merely arbitrary text generation |
| Retrieval-Augmented Dual Instruction Tuning (RA-DIT) | Tuning retriever and generator without joint backpropagation | It is one joint-training alternative |
| RAFT | Training with distractors deliberately placed in context | The source supplies the paper title but does not expand the method label |
| Gumbel-Softmax | Continuous relaxation of categorical selection | Finite-temperature gradients do not equal exact hard-selection gradients |
| InPars | Query generation from documents with filtering | The annotation says filtering supplies the quality |
| Positional bias | Performance changing with evidence position | Lost in the Middle and related work address it |
| U-curve | Better use of evidence near context ends than the middle | The crucial comparison is against a closed-book line |
| IN2 | The context-utilization training label named in the source | The source uses this label without expanding it |
| Attention with Linear Biases (ALiBi) | A position-bias approach named in the source | It belongs to the length-extrapolation question |
| AIS | Attributable to identified sources | Interpret the explicature in context before checking support |
| Fact Extraction and VERification (FEVER) | A dataset with supported, refuted, and not-enough-information labels | Its labels underlie an attribution taxonomy |
| GopherCite | A system that samples and reranks against quoted spans | It exposes the cost of honest attribution |
| AttributionBench | A benchmark for automatic attribution evaluation | Automatic judges remain far from human agreement |
| FActScore | Atomic-fact evaluation of factual precision | Whole-answer scoring hides long-form errors |
| VeriScore | Context-aware verifiable claims and supported-count F1 proxy | Not semantic recall of all missing gold facts or verification of unverifiable content |
| RAGAs | Automated RAG evaluation with three named measures | The source warns to know what each measure omits |
| NLI | Natural language inference | Entailment/contradiction signals remain model- and context-dependent |
| Pool depth | Amount of judged candidate material | Unjudged-as-irrelevant can bias comparisons; not every bias is identical |
| RA-RAG | RAG with source reliability estimated from cross-source agreement | The source supplies the method name without expanding it |
| Credibility-Aware Attention Modification (CrAM) | Down-weighting attention heads influenced by unreliable documents | It is an inference-time intervention without retraining |
| Fairness of exposure | Allocate visibility subject to a specified fairness model | Relevance/merit can still define utility and fairness targets |
| Maximal marginal relevance (MMR) | Reranking for relevance and diversity | It remains a baseline for reducing redundancy |
| Indirect prompt injection | Untrusted external data carry adversarial instructions | Retrieval is one channel, not every RAG security risk |
| PoisonedRAG | A corpus-poisoning attack study | Report document access/budget and tested success conditions |
| GCG | Greedy coordinate gradient | A specified adversarial-suffix optimization method |
| Instruction hierarchy | Train priority among instruction sources | Improves a boundary behavior; not a formal isolation guarantee |
| C2PA | Signed provenance/content credentials | Signature integrity does not certify factual truth or source support |
| RAGRoute | Learned source selection for federated RAG | It avoids unnecessary fan-out cost |
| IO-awareness | Input/output-aware algorithm design | Explicitly count movement between memory levels |
| FlashAttention | IO-aware exact attention | Less memory traffic does not eliminate quadratic attention arithmetic |
| CLIP | The visual-language model named in the source | The source supplies its contrastive setup but does not expand the name |
| GraphRAG | Graph-based query-focused summarization | The source stresses community hierarchy, map-reduce, and index cost |
| MRAG-Bench | A vision-centric benchmark for multimodal RAG | The annotation asks what such benchmarks measure and how many are text-answerable |
| LightRAG | Entity/relation-oriented retrieval and update design | Measure refresh scope, quality and costs instead of assuming a universal bound |
| Louvain and Leiden | Community-detection algorithms | Connectivity guarantees are not global-optimality or semantic guarantees |


## Core mechanics
Use each annotation as a question to investigate in the original paper: what was measured, against which baseline, under which assumptions, and at what cost? Titles and supplied author strings are preserved here; per-entry years and venues are not invented. Correction: author lists for entries 6 and 9 in cluster 8, entries 5, 9 and 14 in cluster 11, entry 10 in cluster 13, and entries 1, 2, 3, 7, 12, 15 and 17 in cluster 14 (entries 57, 60, 101, 105, 110, 136, 144, 145, 146, 150, 155, 158 and 160 in continuous numbering) were corrected against arXiv and venue records and may differ from the printed source. The chapter citations are interpretation cross-references, not claims that every listed paper was experimentally reproduced during this review.
The actual list has $`161`$ entries, $`160`$ distinct titles, $`14`$ clusters and $`52`$ core markers. The repeated title is Cumulated Gain-Based Evaluation of IR Techniques. The source’s roughly $`500`$ cited works is a separate estimate of the wider bibliography.
For mathematical reading, distinguish exact identities from estimates: $`C\approx6ND`$ is a training-compute proxy, LoRA counts trainable factors rather than all FLOPs, $`4d+8M`$ is only a scoped HNSW subtotal, and PQ codebook scalars equal $`(km)(d/m)=kd`$ for equal subspaces. Read the conditions in the annotations before applying any number.
The source gives two usage notes. The annotations emphasize negative results, including retrieval lowering accuracy, semantic chunking gains not transferring, weak citation quality, and unreliable self-correction without external feedback. The list also includes governing constraints from non-retrieval papers, including distance concentration, tail latency, and instruction hierarchies.
### 1. What RAG is, and when it fails
**What.** This cluster supports chapter 1 and chapter 3.
**Why.** It defines the pattern, compares parametric and non-parametric knowledge, and includes the central negative result.
**Failure without it.** A candidate can name RAG without explaining its marginalizations, its datastore rationale, or when retrieval hurts.
**Cost and complexity.** The cluster contains 7 entries, including 2 marked `[core]`.
1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** `[core]`. Authors: Lewis, Perez, Piktus et al. Claim to know: The paper named the pattern. Know the parametric / non-parametric split. RAG-Sequence and RAG-Token are different marginalizations, not implementation details.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
2. **When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories** `[core]`. Authors: Mallen, Asai, Zhong et al. Claim to know: In the reported models/tasks, retrieval can change an otherwise-correct closed-book answer into a wrong one; the roughly $`10\%`$ figure is conditional on that evaluation, not a universal harm rate. Correction: Mallen et al. (2023, arXiv 2212.10511) report that for about $`10\%`$ of PopQA questions, retrieval augmentation made GPT-3 (text-davinci-003) answer incorrectly a question it answered correctly without retrieval, and this $`10\%`$ is a share of all questions, not of otherwise-correct answers. Study the popularity-stratified baselines and Wikipedia-page-view proxy.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
3. **Large Language Models Struggle to Learn Long-Tail Knowledge**. Authors: Kandpal, Deng, Roberts et al. Claim to know: The study relates factual accuracy to entity co-occurrence in the studied training data. This is empirical evidence motivating external knowledge, not a universal quantitative law for every model.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
4. **Nonparametric Masked Language Modeling**. Authors: Min, Shi, Lewis et al. Claim to know: Retrieval is a modeling choice rather than a bolt-on. This is the cleanest statement of what non-parametric capacity buys.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
5. **Toolformer: Language Models Can Teach Themselves to Use Tools**. Authors: Schick, Dwivedi-Yu, Dessi et al. Claim to know: It establishes retrieval as one instance of tool use. It is useful for the framing question “why is search special?”[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
6. **ReAct: Synergizing Reasoning and Acting in Language Models**. Authors: Yao, Zhao, Yu et al. Claim to know: It supplies the interleaved reason-act loop that later agentic retrieval builds on.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
7. **Retrieval-Augmented Generation for Large Language Models: A Survey**. Authors: Gao, Xiong, Gao et al. Claim to know: Use the naive/advanced/modular taxonomy to describe design patterns, not to certify maturity, quality or an inevitable progression.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
### 2. The alternatives: editing, continual learning, long context
**What.** This cluster supports chapter 2.
**Why.** It compares retrieval with editing, continual learning, and generative retrieval.
**Failure without it.** A candidate can recommend changing weights without knowing localization limits, ripple failures, forgetting, or indexing wounds.
**Cost and complexity.** The cluster contains 9 entries, including 1 marked `[core]`.
1. **Locating and Editing Factual Associations in GPT** `[core]`. Authors: Meng, Bau, Andonian et al. Claim to know: ROME uses causal tracing and rank-one edits to study factual associations in specified models. A causal intervention at middle-layer MLPs does not imply that each fact occupies an isolated, uniquely owned weight location.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
2. **Mass-Editing Memory in a Transformer**. Authors: Meng, Sharma, Andonian et al. Claim to know: MEMIT is the scaling answer to ROME. Know where batch editing starts to degrade unrelated knowledge.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
3. **Does Localization Inform Editing? Surprising Differences in Causality-Based Localization**. Authors: Hase, Bansal, Kim et al. Claim to know: Compare causal-localization scores with edit efficacy under the paper’s methods and models; localization need not reliably predict the best edit site. This is not a theorem that localization never informs any editor.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
4. **Evaluating the Ripple Effects of Knowledge Editing in Language Models**. Authors: Cohen, Biran, Yoran et al. Claim to know: Edits can fail to propagate consistently to related or entailed facts. Inspect the ripple evaluation rather than assuming all edits always fail or always preserve consistency.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
5. **Overcoming Catastrophic Forgetting in Neural Networks**. Authors: Kirkpatrick, Pascanu, Rabinowitz et al. Claim to know: EWC frames forgetting as a constraint on parameter drift.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
6. **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning**. Authors: Luo, Yang, Meng et al. Claim to know: Know how much is actually lost and at what scale.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
7. **Transformer Memory as a Differentiable Search Index**. Authors: Tay, Tran, Dehghani et al. Claim to know: DSI generates identifiers with a learned model instead of the usual explicit similarity-search path. Identifier validity, evidence access and corpus updates still incur work; update difficulty is not proof that incremental methods cannot exist.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
8. **DSI++: Updating Transformer Memory with New Documents**. Authors: Mehta, Gupta, Tay et al. Claim to know: Study the proposed new-document update method, its costs and retained-document performance. Its existence rules out a blanket claim that generative retrieval admits no incremental updating.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
9. **Rethinking Search: Making Domain Experts out of Dilettantes**. Authors: Metzler, Tay, Bahri et al. Claim to know: This is the position paper behind generative retrieval. It is useful for the “are GenIR and RAG contradictory?” exchange.[\[2\]](../part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
### 3. What the generator brings, and what it breaks
**What.** This cluster supports chapter 4.
**Why.** It covers memorization, extraction, hallucination, faithfulness, irrelevant context, and confidence.
**Failure without it.** A candidate can treat generation as a neutral final stage and miss its retained data, calibration limits, and context failures.
**Cost and complexity.** The cluster contains 9 entries, including 3 marked `[core]`.
1. **Quantifying Memorization Across Neural Language Models** `[core]`. Authors: Carlini, Ippolito, Jagielski et al. Claim to know: The study reports increased extractable memorization with model scale, duplication and supplied context in its tested regime. These are empirical trends, not three unrestricted probability laws.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
2. **Scalable Extraction of Training Data from (Production) Language Models**. Authors: Nasr, Carlini, Hayase et al. Claim to know: Know the divergence attack mechanism, not merely that it exists.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
3. **Deduplicating Training Data Makes Language Models Better**. Authors: Lee, Ippolito, Nystrom et al. Claim to know: The reported experiments reduce extractable memorization and can improve language-model quality through deduplication. Transfer and magnitude must be checked for the actual corpus and recipe.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
4. **Survey of Hallucination in Natural Language Generation**. Authors: Ji, Lee, Frieske et al. Claim to know: Know the intrinsic-versus-extrinsic taxonomy used by the book and where each type originates.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
5. **On Faithfulness and Factuality in Abstractive Summarization** `[core]`. Authors: Maynez, Narayan, Bohnet et al. Claim to know: This distinction supports the evaluation half of the book. An answer can be faithful and false.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
6. **Large Language Models Can Be Easily Distracted by Irrelevant Context** `[core]`. Authors: Shi, Chen, Misra et al. Claim to know: Adding irrelevant passages can degrade accuracy in the tested setup. More retrieved text is not guaranteed to help; the direction depends on content, model and task.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
7. **Impact of Co-occurrence on Factual Knowledge of Large Language Models**. Authors: Kang and Choi. Claim to know: Models can substitute a frequently co-occurring entity for the correct one.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
8. **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection**. Authors: Manakul, Liusie, Gales. Claim to know: Sampling-based agreement is a fallible hallucination signal, not a truth certificate. Account for sample cost and shared systematic errors.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
9. **Language Models (Mostly) Know What They Know**. Authors: Kadavath, Conerly, Askell et al. Claim to know: Self-evaluation is partially calibrated. This is both the foundation and the limit of confidence-based gating.[\[3\]](../part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
### 4. Privacy, law, and the argument for a datastore
**What.** This cluster supports chapter 5.
**Why.** It ties corpus composition, legal risk, toxicity, and training-data extraction to deployment design.
**Failure without it.** A candidate can make a datastore recommendation without its legal architecture or empirical corpus rationale.
**Cost and complexity.** The cluster contains 3 entries, including 1 marked `[core]`.
1. **SILO Language Models: Isolating Legal Risk in a Nonparametric Datastore** `[core]`. Authors: Min, Gururangan, Wallace et al. Claim to know: SILO trains on selected lower-risk tiers and keeps revocable material keyed in a datastore. This makes later serving exclusion addressable; it does not itself establish the lawfulness of acquisition, indexing or every later use.[\[4\]](../part-02-generator/05_data_privacy_and_the_legal_surface.md)
2. **RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models**. Authors: Gehman, Gururangan, Sap et al. Claim to know: Know what is measurably present in a web-scraped corpus.[\[4\]](../part-02-generator/05_data_privacy_and_the_legal_surface.md)
3. **Extracting Training Data from Large Language Models**. Authors: Carlini, Tramer, Wallace et al. Claim to know: The extraction result made corpus composition a deployment question rather than a research question.[\[4\]](../part-02-generator/05_data_privacy_and_the_legal_surface.md)
### 5. Scaling, economics, and in-context learning
**What.** This cluster supports chapter 6 and chapter 7.
**Why.** It supplies scaling arithmetic, compute-optimal practice, in-context behavior, calibration, model configuration, and adaptation cost.
**Failure without it.** A candidate can quote scaling or fine-tuning claims without using their arithmetic or knowing what mechanism changes.
**Cost and complexity.** The cluster contains 7 entries, including 4 marked `[core]`.
1. **Scaling Laws for Neural Language Models** `[core]`. Authors: Kaplan, McCandlish, Henighan et al. Claim to know: Use $`C\approx6ND`$ as a scoped dense-model training-FLOP proxy: $`N`$ is parameter count and $`D`$ cumulative processed training tokens. Study the original scaling regime separately from later compute-optimal results; attention, utilization and system overhead are not represented by this proxy.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
2. **Training Compute-Optimal Large Language Models** `[core]`. Authors: Hoffmann, Borgeaud, Mensch et al. Claim to know: Chinchilla studies a compute-optimal model/data allocation under its training setup. The often-quoted roughly twenty tokens per parameter is not a universal constant; lifetime inference demand can change the economic optimum.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
3. **Language Models are Few-Shot Learners** `[core]`. Authors: Brown, Mann, Ryder et al. Claim to know: This is the mechanism RAG rides on. In-context learning is conditioning, not a weight update.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
4. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?**. Authors: Min, Lyu, Holtzman et al. Claim to know: In the paper’s tasks, demonstration distribution and label-space cues can matter strongly even when labels are corrupted. Do not generalize this into “label correctness never matters,” especially for task learning or safety-critical examples.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
5. **Calibrate Before Use: Improving Few-Shot Performance of Language Models**. Authors: Zhao, Wallace, Feng et al. Claim to know: Content-free calibration, and the ordering sensitivity that makes it necessary.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
6. **The Llama 3 Herd of Models**. Authors: Grattafiori, Dubey et al. Claim to know: This is the source of the book’s default configuration numbers, including layer count, KV head count, head dimension, and the token budget used in memory arithmetic.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
7. **LoRA: Low-Rank Adaptation of Large Language Models** `[core]`. Authors: Hu, Shen, Wallis et al. Claim to know: For one adapted matrix, two rank-$`r`$ factors add $`r(d_{\mathrm{in}}+d_{\mathrm{out}})`$ trainable parameters, excluding biases and other adaptations. Fewer trainable parameters reduce optimizer/state costs but do not remove the frozen base-model forward and backward-path computation.[\[5\]](../part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
### 6. Reading the machine
**What.** This cluster supports chapter 8.
**Why.** It covers transformer circuits, induction, parameter storage, feature attribution, and the limits of attention as evidence.
**Failure without it.** A candidate can use mechanistic language without knowing which evidence supports information flow, storage, or attribution.
**Cost and complexity.** The cluster contains 5 entries, including 1 marked `[core]`.
1. **A Mathematical Framework for Transformer Circuits** `[core]`. Authors: Elhage, Nanda, Olsson et al. Claim to know: The residual stream is an information bus. Know the $`QK`$/$`OV`$ factorization, which circuit decides where to look, and which decides what to move.[\[6\]](../part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
2. **In-context Learning and Induction Heads**. Authors: Olsson, Elhage, Nanda et al. Claim to know: Induction heads provide a copying mechanism and evidence about some in-context behavior in the studied transformers. They are not a complete or universal causal explanation of retrieval-augmented generation.[\[6\]](../part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
3. **Transformer Feed-Forward Layers Are Key-Value Memories**. Authors: Geva, Schuster, Berant et al. Claim to know: The key-value-memory interpretation of feed-forward layers connects to model-editing experiments. It does not identify an exclusive location for all facts or an independent fact-key delete.[\[6\]](../part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
4. **Axiomatic Attribution for Deep Networks**. Authors: Sundararajan, Taly, Yan. Claim to know: Integrated gradients are the reference point for what feature attribution can and cannot claim about a generated answer.[\[6\]](../part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
5. **Attention is not Explanation**. Authors: Jain and Wallace. Claim to know: The paper challenges treating attention weights alone as faithful explanations in its evaluated settings. Attention visualizations require causal/behavioral validation; this is not proof that every attention-based analysis is useless.[\[6\]](../part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
### 7. Prompting, sensitivity, and abstention
**What.** This cluster supports chapter 9 through chapter 11.
**Why.** It covers reasoning scaffolds, prompt search, calibration, uncertainty, self-correction, and recursive-data failure.
**Failure without it.** A candidate can recommend more reasoning or self-critique without pricing samples, measuring sensitivity, or checking external feedback.
**Cost and complexity.** The cluster contains 11 entries, including 3 marked `[core]`.
1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** `[core]`. Authors: Wei, Wang, Schuurmans et al. Claim to know: This is the scaffold everything else in the cluster modifies.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
2. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**. Authors: Wang, Wei, Schuurmans et al. Claim to know: Sample reasoning paths and aggregate answers. Gains depend on task, error dependence and voting rules; separate total generation work from latency with shared prefixes or parallel sampling.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
3. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. Authors: Yao, Yu, Zhao et al. Claim to know: Search over reasoning states, and where the added scaffolding stops paying.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
4. **POSIX: A Prompt Sensitivity Index for Large Language Models**. Authors: Chatterjee et al. Claim to know: A measurable definition of prompt sensitivity makes it an engineering variable rather than folklore.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
5. **Large Language Models Are Human-Level Prompt Engineers**. Authors: Zhou, Muresanu, Han et al. Claim to know: Automated prompt search, and the baseline any hand-tuned prompt should beat.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
6. **On Calibration of Modern Neural Networks** `[core]`. Authors: Guo, Pleiss, Sun, Weinberger. Claim to know: Expected calibration error, and the observation that larger models are not automatically better calibrated.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
7. **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation**. Authors: Kuhn, Gal, Farquhar. Claim to know: Group outputs by estimated meaning before measuring entropy. This can avoid treating paraphrases as disagreement, but semantic clustering can err and low entropy can reflect a shared false answer.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
8. **Detecting Hallucinations in Large Language Models Using Semantic Entropy**. Authors: Farquhar, Kossen, Kuhn et al. Claim to know: Study semantic entropy as an uncertainty signal and validate its failure prediction on the deployment task. It is not a general hallucination-proofing guarantee.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
9. **Just Ask for Calibration**. Authors: Tian, Mitchell, Zhou et al. Claim to know: Verbalized confidence from a black-box model, and how far it can be trusted.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
10. **Large Language Models Cannot Self-Correct Reasoning Yet** `[core]`. Authors: Huang, Chen, Mishra et al. Claim to know: Self-critique without external feedback need not reliably improve reasoning in the tested settings. Compare feedback access, stopping, model and task when reading Self-Refine or Reflexion; differing conditions need not produce a contradiction.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
11. **AI Models Collapse When Trained on Recursively Generated Data**. Authors: Shumailov, Shumaylov, Zhao et al. Claim to know: The paper studies recursive training-data contamination by model-generated samples. It does not prove that an inference-time self-reflection loop automatically causes the same training-distribution collapse.[\[7\]](../part-03-prompting/11_abstention_and_calibration.md)
### 8. Representation, chunking, and document structure
**What.** This cluster supports chapter 12 through chapter 14.
**Why.** It connects embedding objectives, evaluation, granularity, chunking, tables, and page layout.
**Failure without it.** A candidate can select embeddings or chunks by leaderboard and label without checking task dependence, anisotropy, transfer, or structure.
**Cost and complexity.** The cluster contains 13 entries, including 3 marked `[core]`.
1. **Efficient Estimation of Word Representations in Vector Space**. Authors: Mikolov, Chen, Corrado, Dean. Claim to know: Study word2vec’s distributional objectives and static representations. Their objective differs from query-document relevance training; that does not make every use of static embeddings invalid.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
2. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** `[core]`. Authors: Reimers and Gurevych. Claim to know: Independent sentence encoding enables reuse of document vectors, whereas a general joint cross-encoder score cannot be reproduced by one query-independent document vector. Compare the lost joint interaction and measured efficiency; some pair-level work can still be cached.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
3. **SimCSE: Simple Contrastive Learning of Sentence Embeddings**. Authors: Gao, Yao, Chen. Claim to know: Dropout as the minimal augmentation, and the alignment/uniformity framing.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
4. **MTEB: Massive Text Embedding Benchmark**. Authors: Muennighoff, Tazi, Magne, Reimers. Claim to know: This is the benchmark you will be asked about when someone proposes an embedding model. Leaderboard position is task-dependent.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
5. **How Contextual are Contextualized Word Representations?**. Authors: Ethayarajh. Claim to know: Study anisotropy and similarity baselines in the evaluated representations. The reliability of raw cosine depends on the embedding and task, not merely on the existence of anisotropy.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
6. **Dense X Retrieval: What Retrieval Granularity Should We Use?** `[core]`. Authors: Chen, Wang, Chen et al. Claim to know: Propositions as the retrieval unit. This is the clearest statement that granularity is a first-class design decision.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
7. **Is Semantic Chunking Worth the Computational Cost?** `[core]`. Authors: Qu, Tu, Bao. Claim to know: Audit whether benchmark gains transfer from stitched or otherwise constructed inputs to naturally occurring documents, and compare against matched fixed-size baselines. Do not infer that every semantic chunker is ineffective or that all reported gains share one cause.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
8. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**. Authors: Sarthi, Abdullah, Tuli et al. Claim to know: Recursive summaries offer additional granularity and can help global/thematic questions. Measure them against strong flat retrieval; a query label alone does not prove that flat chunks cannot answer it.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
9. **Mix-of-Granularity: Optimize the Chunking Granularity for RAG**. Authors: Zhong, Liu, Cui et al. Claim to know: Route over chunk sizes rather than choosing one.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
10. **A Critique and Improvement of an Evaluation Metric for Text Segmentation**. Authors: Pevzner and Hearst. Claim to know: WindowDiff, and why $`P_k`$ under-penalizes certain boundary errors.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
11. **TaPas: Weakly Supervised Table Parsing via Pre-training**. Authors: Herzig, Nowak, Muller et al. Claim to know: Row and column position embeddings, and the assumptions they bake in.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
12. **TableFormer: Robust Transformer Modeling for Table-Text Encoding**. Authors: Yang, Gupta, Upadhyay et al. Claim to know: Study the table-structure encoding and the permutations it is designed to handle. Order invariance is architecture- and task-specific; ordered rows and columns can carry real semantics.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
13. **LayoutLM: Pre-training of Text and Layout for Document Image Understanding**. Authors: Xu, Li, Cui et al. Claim to know: Layout as a first-class input rather than a preprocessing step.[\[8\]](../part-04-representations/13_chunking_and_granularity.md)
### 9. Vector indexes and compression
**What.** This cluster supports chapter 15 through chapter 17.
**Why.** It supplies the graph, compression, storage, filtering, benchmarking, dimensionality, and tail-latency foundations.
**Failure without it.** A candidate can quote one latency number without deriving the index, checking recall, or understanding fan-out and high-dimensional limits.
**Cost and complexity.** The cluster contains 15 entries, including 5 marked `[core]`.
1. **Skip Lists: A Probabilistic Alternative to Balanced Trees** `[core]`. Authors: Pugh. Claim to know: Pugh derives probabilistic skip-list search bounds, including a form $`L(n)/p+1/(1-p)`$ under the stated promotion convention. HNSW borrows a hierarchy analogy, not the ordered skip list’s exact comparison bound or every search guarantee.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
2. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs** `[core]`. Authors: Malkov and Yashunin. Claim to know: Know the common $`m_L=1/\ln M`$ convention for $`M\gt 1`$, base capacity $`M_0=2M`$ and neighbor-diversity heuristic. The $`4d+8M`$ bytes rule includes fp32 vector payload and allocated base-layer links with four-byte IDs only; add upper layers and all other metadata/runtime terms.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
3. **Approximate Nearest Neighbor Algorithm Based on Navigable Small World Graphs**. Authors: Malkov, Ponomarenko, Logvinov, Krylov. Claim to know: Flat NSW, and why the hierarchy was needed.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
4. **Product Quantization for Nearest Neighbor Search** `[core]`. Authors: Jegou, Douze, Schmid. Claim to know: With $`m`$ equal subspaces in dimension $`d`$ and $`k`$ centroids per subspace, at most $`k^m`$ reconstructions use $`km`$ subspace centroids containing $`kd`$ scalar values. Thus centroid bytes are independent of $`m`$ only at fixed $`k,d`$ and value precision; code length still grows with $`m`$.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
5. **Optimized Product Quantization for Approximate Nearest Neighbor Search**. Authors: Ge, He, Ke, Sun. Claim to know: Learning the rotation before splitting, and when it is worth the extra step.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
6. **Billion-Scale Similarity Search with GPUs** `[core]`. Authors: Johnson, Douze, Jegou. Claim to know: FAISS. This is the source against which the book sanity-checks its memory rules of thumb.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
7. **DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node**. Authors: Subramanya, Devvrit, Kadekodi et al. Claim to know: Vamana, and the SSD-resident regime where the graph no longer has to fit in RAM.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
8. **FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search**. Authors: Singh, Subramanya, Krishnaswamy, Simhadri. Claim to know: Deletion and insertion in a graph index without a full rebuild. Read it alongside the tombstone material.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
9. **Similarity Estimation Techniques from Rounding Algorithms**. Authors: Charikar. Claim to know: SimHash, and the LSH collision-probability argument.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
10. **Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality**. Authors: Indyk and Motwani. Claim to know: Study the near/far collision guarantees and the parameter, approximation and output assumptions in LSH query bounds. High dimension alone does not establish a universal performance ranking against every graph index.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
11. **When Is “Nearest Neighbor” Meaningful?** `[core]`. Authors: Beyer, Goldstein, Ramakrishnan, Shaft. Claim to know: Distance concentration follows under specified distributional conditions and can weaken distance contrast. It neither proves exact search always becomes infeasible nor makes ANN mathematically necessary for every workload.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
12. **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms**. Authors: Aumuller, Bernhardsson, Faithfull. Claim to know: The recall-versus-QPS curve as the reporting convention. Ask for this plot whenever a vendor quotes one latency number.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
13. **Matryoshka Representation Learning**. Authors: Kusupati, Bhatt, Rege et al. Claim to know: Nested representations are trained to make selected prefixes useful. Truncation quality is empirical, and arbitrary embeddings do not acquire this property for free.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
14. **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings**. Authors: Patel, Kraft, Guestrin, Zaharia. Claim to know: Filtered search without a separate index per predicate, and the over-filtering failure it addresses.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
15. **The Tail at Scale**. Authors: Dean and Barroso. Claim to know: Fan-out can amplify request-level tail latency through the slowest required responses. Dependence, load, retries and quorum policy matter; measure the actual $`p_{99}`$ rather than assuming a universal shard formula.[\[9\]](../part-05-indexing/16_compression_and_index_economics.md)
### 10. Ranking: lexical, dense, sparse, multi-vector
**What.** This cluster supports chapter 18 through chapter 23.
**Why.** It covers lexical ranking, dense and sparse training, late interaction, reranking, learning to rank, evaluation, datasets, and fusion.
**Failure without it.** A candidate can default to dense-only retrieval without its out-of-domain counterexample, storage cost, benchmark bias, or ranking metrics.
**Cost and complexity.** The cluster contains 17 entries, including 8 marked `[core]`.
1. **The Probabilistic Relevance Framework: BM25 and Beyond** `[core]`. Authors: Robertson and Zaragoza. Claim to know: This is the reference for BM25. Know where $`k_1`$ and $`b`$ enter and why the length discount sits inside the saturation denominator.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
2. **Okapi at TREC-3**. Authors: Robertson, Walker, Jones et al. Claim to know: Read the original evaluation and parameterization. Popular defaults are implementation and collection choices, not universally optimal constants.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
3. **Pivoted Document Length Normalization**. Authors: Singhal, Buckley, Mitra. Claim to know: Why length normalization was needed at all, shown empirically.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
4. **A Statistical Interpretation of Term Specificity**. Authors: Sparck Jones. Claim to know: IDF’s origin. This is short and still the clearest statement of the intuition.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
5. **Dense Passage Retrieval for Open-Domain Question Answering** `[core]`. Authors: Karpukhin, Oguz, Min et al. Claim to know: Know the in-batch-negative setup, positive-label assumptions and independently encoded query/document efficiency argument. Check the evaluation protocol rather than relying on a citation-count superlative.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
6. **Approximate Nearest Neighbor Negative Contrastive Learning** `[core]`. Authors: Xiong, Xiong, Li et al. Claim to know: ANCE. Know the gradient-norm argument for why hard negatives matter and why the index must refresh during training.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
7. **Unsupervised Dense Information Retrieval with Contrastive Learning**. Authors: Izacard, Caron, Hosseini et al. Claim to know: Contriever, and how far retrieval can go without labeled pairs.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
8. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models** `[core]`. Authors: Thakur, Reimers, Ruckle et al. Claim to know: BEIR compares heterogeneous zero-shot tasks and shows strong lexical baselines against many tested dense retrievers. It motivates domain-specific comparison, not a universal dense-versus-lexical ordering.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
9. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking** `[core]`. Authors: Formal, Piwowarski, Clinchant. Claim to know: Study the masked-language-model vocabulary head, activation/pooling and learned expansion. FLOPS regularization is a differentiable sparsity/expected-posting-work proxy, not a measured count of hardware FLOPs or end-to-end latency.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
10. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction** `[core]`. Authors: Khattab and Zaharia. Claim to know: Late interaction and MaxSim. Know the storage cost per token and why that is what ColBERTv2 attacks.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
11. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**. Authors: Santhanam, Khattab, Saad-Falcon et al. Claim to know: Residual compression brings multi-vector storage into range.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
12. **Passage Re-ranking with BERT**. Authors: Nogueira and Cho. Claim to know: Joint query-document scoring is a common second-stage design. Its candidate, length, interaction and batching costs matter; later retrieval systems need not all share this architecture.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
13. **Document Ranking with a Pre-trained Sequence-to-Sequence Model**. Authors: Nogueira, Jiang, Pradeep, Lin. Claim to know: monoT5 renormalizes the logits of the single-token targets true and false: $`r=\sigma(z_{\mathrm{true}}-z_{\mathrm{false}})`$. It does not rank by the raw true logit. Distinguish this pointwise classification training from pairwise/listwise ranking losses.[\[11\]](https://arxiv.org/abs/2003.06713)
14. **From RankNet to LambdaRank to LambdaMART: An Overview** `[core]`. Authors: Burges. Claim to know: LambdaRank supplies metric-weighted pseudo-gradients built from pairwise preferences. A RankNet pair loss exists, but rank-dependent weights treated as fixed do not automatically make the whole field the exact gradient of one smooth global ranking objective.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
15. **Cumulated Gain-Based Evaluation of IR Techniques** `[core]`. Authors: Jarvelin and Kekalainen. Claim to know: DCG combines relevance gains with a chosen rank discount; nDCG divides by ideal DCG for the same judged query and cutoff. Logarithmic discount is a conventional design choice, not a uniquely optimal mathematical consequence.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
16. **MS MARCO: A Human Generated MAchine Reading COmprehension Dataset**. Authors: Nguyen, Rosenberg, Song et al. Claim to know: MS MARCO is widely used for retrieval training/evaluation. Study its annotation, query and candidate biases; it does not represent every workload or establish how most published retrieval numbers were produced.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
17. **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods**. Authors: Cormack, Clarke, Buettcher. Claim to know: RRF combines ranked lists using a stated smoothing constant, often $`c_{\mathrm{RRF}}=60`$ in the cited experiments. This is an empirical choice, not a universal optimum or the final context depth.[\[10\]](../part-06-retrieval-ranking/22_reranking.md)
### 11. Query understanding and control flow
**What.** This cluster supports chapter 24 through chapter 26.
**Why.** It connects query reformulation, expansion, retrieval gates, correction, multi-hop control, and learned search.
**Failure without it.** A candidate can propose a controller without its drift mode, generation hallucinations, evaluation cost, supervision source, or stopping rule.
**Cost and complexity.** The cluster contains 16 entries, including 5 marked `[core]`.
1. **Relevance feedback in information retrieval**. Authors: Rocchio. Claim to know: Study Rocchio’s vector update as a classical relevance-feedback baseline. Modern neural reformulation has related goals but need not literally descend from this exact update.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
2. **Query Expansion Using Local and Global Document Analysis**. Authors: Xu and Croft. Claim to know: Pseudo-relevance feedback, and the drift failure mode that makes unsupervised use risky.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
3. **Precise Zero-Shot Dense Retrieval without Relevance Labels** `[core]`. Authors: Gao, Ma, Lin, Callan. Claim to know: HyDE generates a hypothetical document and embeds it to retrieve real evidence. The generated document is a query representation, not verified evidence, and can amplify an erroneous assumption.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
4. **Query2doc: Query Expansion with Large Language Models**. Authors: Wang, Yang, Wei. Claim to know: The same idea in expansion form, with a cost comparison worth knowing.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
5. **Document Expansion by Query Prediction** `[core]`. Authors: Nogueira, Yang, Lin, Cho. Claim to know: doc2query moves document-expansion generation work to ingestion and refresh time. Account for corpus churn, generated-query storage and update costs rather than treating expansion as permanently free.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
6. **Task-Oriented Query Reformulation with Reinforcement Learning**. Authors: Nogueira and Cho. Claim to know: Rewriting trained against retrieval reward rather than against a reference rewrite.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
7. **Active Retrieval Augmented Generation** `[core]`. Authors: Jiang, Xu, Gao et al. Claim to know: FLARE. Know the confidence trigger and the distinction between implicit and explicit query formation.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
8. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** `[core]`. Authors: Asai, Wu, Wang et al. Claim to know: Reflection tokens as learned control. Know that the critic is distilled into the generator rather than run separately.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
9. **Corrective Retrieval Augmented Generation**. Authors: Yan, Gu, Zhu, Ling. Claim to know: CRAG evaluates retrieved quality, decomposes, and falls back to web search. Know what its evaluator costs and where it is fragile.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
10. **Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity**. Authors: Jeong, Baek, Cho et al. Claim to know: Silver-label supervision for a complexity router, and the honest account of where those labels come from.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
11. **Unified Active Retrieval for Retrieval Augmented Generation**. Authors: Cheng et al. Claim to know: Study the four classifier decisions, frozen-state features, silver labels and their measured routing errors. It is one implementable design, not a proof that every workload needs four classifiers.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
12. **Interleaving Retrieval with Chain-of-Thought Reasoning** `[core]`. Authors: Trivedi, Balasubramanian, Khot, Sabharwal. Claim to know: IRCoT interleaves retrieval and reasoning to find evidence for later hops. It addresses cases where one-shot retrieval fails; it does not prove that every compositional question requires recursive retrieval.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
13. **Measuring and Narrowing the Compositionality Gap in Language Models**. Authors: Press, Zhang, Min et al. Claim to know: Self-Ask, and a crisp definition of the gap multi-hop retrieval exists to close.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
14. **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning**. Authors: Jin, Zeng, Yue et al. Claim to know: Study reinforcement learning over reasoning/search trajectories, including tool protocol, rewards, search budget and termination. Learned stopping is one behavior, not the entire contribution.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
15. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning**. Authors: Shao, Wang, Zhu et al. Claim to know: The GRPO algorithm itself, which Search-R1 optimizes with.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
16. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**. Authors: Yang, Qi, Zhang et al. Claim to know: Inspect the multi-hop labels and shortcut artifacts. Some gains can exploit shortcuts; that does not render every benchmark result illusory.[\[12\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
### 12. Training the retriever and the generator
**What.** This cluster supports chapter 27 through chapter 29.
**Why.** It covers pre-training with retrieval, passage fusion, black-box generators, instruction tuning, distractors, differentiability, synthetic data, and supervision limits.
**Failure without it.** A candidate can say “train jointly” without understanding index staleness, decode cost, narrow interfaces, filtering, or recursive-data failure.
**Cost and complexity.** The cluster contains 14 entries, including 6 marked `[core]`.
1. **REALM: Retrieval-Augmented Language Model Pre-Training** `[core]`. Authors: Guu, Lee, Tung et al. Claim to know: REALM places retrieval inside pre-training and asynchronously refreshes document representations. Staleness occurs when document-side parameters or corpus/index state change; a query-only update need not invalidate stored document vectors.[\[13\]](../part-08-training/28_training_the_retriever.md)
2. **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering** `[core]`. Authors: Izacard and Grave. Claim to know: Fusion-in-Decoder. Passages are encoded independently and fused in the decoder. Know what that costs at decode time.[\[13\]](../part-08-training/28_training_the_retriever.md)
3. **FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation**. Authors: Hofstatter, Chen, Raman, Zamani. Claim to know: The cost fix for FiD, and the quality it trades.[\[13\]](../part-08-training/28_training_the_retriever.md)
4. **REPLUG: Retrieval-Augmented Black-Box Language Models** `[core]`. Authors: Shi, Min, Yasunaga et al. Claim to know: REPLUG uses a frozen language model and retrieval-conditioned likelihood signals; its retriever learning requires likelihood access. “Black box” does not imply compatibility with every text-only hosted interface.[\[13\]](../part-08-training/28_training_the_retriever.md)
5. **RA-DIT: Retrieval-Augmented Dual Instruction Tuning**. Authors: Lin, Chen, Chen et al. Claim to know: Tune both halves without joint backpropagation.[\[13\]](../part-08-training/28_training_the_retriever.md)
6. **RAFT: Adapting Language Model to Domain Specific RAG** `[core]`. Authors: Zhang, Patil, Jain et al. Claim to know: Train with distractors deliberately in the context. This is the most directly actionable training result in this part.[\[13\]](../part-08-training/28_training_the_retriever.md)
7. **The Power of Noise: Redefining Retrieval for RAG Systems**. Authors: Cuconasu, Trappolini, Siciliano et al. Claim to know: The counterintuitive finding that some irrelevant context can help, and the conditions under which it does not.[\[13\]](../part-08-training/28_training_the_retriever.md)
8. **Making Retrieval-Augmented Language Models Robust to Irrelevant Context**. Authors: Yoran, Wolfson, Ram, Berant. Claim to know: The complementary robustness-training result.[\[13\]](../part-08-training/28_training_the_retriever.md)
9. **Categorical Reparameterization with Gumbel-Softmax**. Authors: Jang, Gu, Poole. Claim to know: Gumbel-Softmax provides a continuous relaxation with a temperature trade-off. At finite temperature it is not an exact pathwise gradient of hard categorical or top-k selection, and it is not the only way to learn discrete decisions.[\[13\]](../part-08-training/28_training_the_retriever.md)
10. **InPars: Unsupervised Dataset Generation for Information Retrieval** `[core]`. Authors: Bonifacio, Abonizio, Fadaee, Nogueira. Claim to know: Study synthetic query generation and the filtering/selection recipe together. Filtering matters, but generation, coverage and retriever training also contribute; no universal single-source quality explanation follows.[\[13\]](../part-08-training/28_training_the_retriever.md)
11. **Promptagator: Few-shot Dense Retrieval From 8 Examples** `[core]`. Authors: Dai, Zhao, Ma et al. Claim to know: Promptagator demonstrates task-specific retrieval with eight seed examples in the reported setup. This motivates adaptation, not a proof that a general-purpose retriever can never transfer.[\[13\]](../part-08-training/28_training_the_retriever.md)
12. **Synthetic QA Corpora Generation with Roundtrip Consistency**. Authors: Alberti, Andor, Pitler et al. Claim to know: Consistency filtering, stated cleanly.[\[13\]](../part-08-training/28_training_the_retriever.md)
13. **Weak-to-Strong Generalization**. Authors: Burns, Izmailov, Kirchner et al. Claim to know: Study when a student trained on weak supervision exceeds its weak teacher and where it fails. A teacher’s aggregate score is not a strict mathematical upper bound on the student.[\[13\]](../part-08-training/28_training_the_retriever.md)
14. **The Curse of Recursion: Training on Generated Data Makes Models Forget**. Authors: Shumailov, Shumaylov, Zhao et al. Claim to know: Why a synthetic-data pipeline needs a real-data anchor.[\[13\]](../part-08-training/28_training_the_retriever.md)
### 13. Context assembly, attribution, and evaluation
**What.** This cluster supports chapter 30 through chapter 34.
**Why.** It covers evidence position, context use, attribution, factual precision, retrieval-aware evaluation, inconsistency detection, and judgment-set reliability.
**Failure without it.** A candidate can cite a curve or automatic score without its closed-book baseline, attribution limits, blind spots, or human-agreement gap.
**Cost and complexity.** The cluster contains 17 entries, including 5 marked `[core]`.
1. **Lost in the Middle: How Language Models Use Long Contexts** `[core]`. Authors: Liu, Lin, Hewitt et al. Claim to know: Inspect the position-controlled results and matched closed-book baseline. Retrieval-augmented accuracy can fall below closed-book in some evaluated settings; a middle trough is not a universal property of all models and tasks.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
2. **Make Your LLM Fully Utilize the Context**. Authors: An, Ma, Lin et al. Claim to know: IN2 is a training intervention using information-intensive long-context examples. Distinguish training-data/recipe changes from an architectural modification or an inference-only prompt.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
3. **Found in the Middle: Calibrating Positional Attention Bias**. Authors: Hsieh, Chuang, Li et al. Claim to know: Study the proposed positional-attention calibration and its measured effect. Improvement supports that intervention; it does not uniquely prove that all positional failures are mechanical rather than semantic.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
4. **Differential Transformer**. Authors: Ye, Dong, Xia et al. Claim to know: Study the proposed differential-attention mechanism, cost and measured context behavior. Treat it as a tested architectural proposal, not a settled long-run solution.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
5. **Train Short, Test Long: Attention with Linear Biases**. Authors: Press, Smith, Lewis. Claim to know: ALiBi, and the positional-encoding side of the length-extrapolation question.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
6. **Measuring Attribution in Natural Language Generation Models** `[core]`. Authors: Rashkin, Nikolaev, Lamm et al. Claim to know: AIS separates interpretation from evidence attribution. Resolve the statement’s explicature using its context; a sentence that is not self-contained in isolation can still be interpretable and attributable after context resolution.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
7. **FEVER: a Large-scale Dataset for Fact Extraction and VERification**. Authors: Thorne, Vlachos, Christodoulopoulos, Mittal. Claim to know: The supported / refuted / not-enough-information labels that the attribution taxonomy maps onto.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
8. **Teaching Language Models to Support Answers with Verified Quotes**. Authors: Menick, Trebacz, Mikulik et al. Claim to know: GopherCite, and the cost of doing attribution honestly. Sample, then rerank against the quoted span.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
9. **WebGPT: Browser-assisted question-answering with human feedback**. Authors: Nakano, Hilton, Balaji et al. Claim to know: Retrieval as a browsing policy, with citation as a trained behavior rather than a post-process.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
10. **AttributionBench: How Hard is Automatic Attribution Evaluation?** `[core]`. Authors: Li, Yue, Liao, Sun. Claim to know: Know how far automatic attribution judges are from human agreement. Cite this whenever someone proposes an LLM judge for grounding.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
11. **FActScore: Fine-grained Atomic Evaluation of Factual Precision** `[core]`. Authors: Min, Krishna, Lyu et al. Claim to know: Atomic-fact scoring exposes mixed correctness hidden by a coarse answer-level label. Whole-answer adequacy and critical-error measures still provide complementary information.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
12. **VeriScore: Evaluating the Factuality of Verifiable Claims**. Authors: Song, Kim, Iyyer. Claim to know: VeriScore extracts context-aware verifiable claims and combines support precision with $`R_K=\min(s/K,1)`$, where $`s`$ counts supported claims and $`K`$ is a domain cohort median. This is a supported-count proxy, not recall of all missing gold facts or a correction that verifies unverifiable assertions.[\[15\]](https://arxiv.org/html/2406.19276v1)
13. **RAGAs: Automated Evaluation of Retrieval Augmented Generation** `[core]`. Authors: Es, James, Espinosa-Anke, Schockaert. Claim to know: Faithfulness, answer relevance, and context relevance. Know what each does and does not measure before quoting a RAGAS number.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
14. **SummaC: Re-Visiting NLI-based Models for Inconsistency Detection**. Authors: Laban, Schnabel, Bennett, Hearst. Claim to know: Study sentence-level entailment aggregation as a consistency signal, including its evaluated advantages and limits. Failures of a tested document-level NLI setup are not proof that every document-level method must fail.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
15. **Cumulated Gain-Based Evaluation of IR Techniques**. Authors: Jarvelin and Kekalainen. Claim to know: This appears again because the evaluation chapters use it directly. The derivation is in section A.4.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
16. **How Reliable Are the Results of Large-Scale Information Retrieval Experiments?**. Authors: Zobel. Claim to know: Shallow judgment pools and treating unjudged items as irrelevant can bias evaluation. State pooling, coverage and uncertainty; the direction and magnitude are not identical for every system comparison. Correction: Zobel (SIGIR 1998) found that pooled TREC judgments give reliable relative comparisons between systems, even though measured recall is overestimated because many relevant documents are never judged, and he proposed a cheaper pooling strategy.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
17. **Overview of the TREC 2019 Deep Learning Track**. Authors: Craswell, Mitra, Yilmaz et al. Claim to know: How a serious judgment set is actually built.[\[14\]](../part-10-evaluation/33_evaluating_generation.md)
### 14. Trust, robustness, scale, and advanced variants
**What.** This cluster supports chapter 35 through chapter 41.
**Why.** It covers source reliability, credibility interventions, exposure, diversity, attacks, provenance, routing, attention cost, multimodal retrieval, and graph retrieval.
**Failure without it.** A candidate can promise trustworthy or advanced RAG without the source signal, attack path, authenticity boundary, serving cost, or graph-index objection.
**Cost and complexity.** The cluster contains 18 entries, including 5 marked `[core]`.
1. **Retrieval-Augmented Generation with Estimation of Source Reliability** `[core]`. Authors: Hwang, Park, Park et al. Claim to know: RA-RAG estimates source reliability using cross-source agreement in its stated setup. Agreement can also reflect copying or coordinated errors; this is one scalable estimation approach, not the only possible one.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
2. **CrAM: Credibility-Aware Attention Modification**. Authors: Deng, Wang, Zhu et al. Claim to know: Down-weight attention on heads most influenced by unreliable documents. This is an inference-time intervention with no retraining.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
3. **Trustworthiness in Retrieval-Augmented Generation Systems: A Survey**. Authors: Zhou, Zhang, Shao et al. Claim to know: The six-aspects-across-three-stages taxonomy used to organize the credibility chapter.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
4. **Fairness of Exposure in Rankings**. Authors: Singh and Joachims. Claim to know: Study stochastic ranking policies and exposure constraints relative to the stated merit/relevance convention. Exposure is the allocated resource; relevance is still needed to define utility and possibly fairness targets.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
5. **The Use of MMR, Diversity-Based Reranking**. Authors: Carbonell and Goldstein. Claim to know: Maximal marginal relevance. This is the classical answer to redundancy in a retrieved set and remains a reasonable baseline.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
6. **Not What You’ve Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** `[core]`. Authors: Greshake, Abdelnabi, Mishra et al. Claim to know: Study indirect instruction injection through untrusted external content. Retrieval is a common entry path, not the only one; this does not define every RAG security threat.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
7. **PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation** `[core]`. Authors: Zou, Geng, Wang, Jia. Claim to know: Study the attacker’s corpus access, document budget, success metric and tested models. A small poison budget can be effective in that threat model, not necessarily in every defended deployment.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
8. **Universal and Transferable Adversarial Attacks on Aligned Language Models**. Authors: Zou, Wang, Kolter, Fredrikson. Claim to know: GCG studies optimized adversarial suffixes and transfer under specified access and target assumptions. Report success rates and defenses rather than assuming every guardrail fails identically.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
9. **Jailbroken: How Does LLM Safety Training Fail?**. Authors: Wei, Haghtalab, Steinhardt. Claim to know: Know competing objectives and mismatched generalization as the paper’s explanatory mechanisms, not an exhaustive taxonomy of every possible safety failure.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
10. **The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**. Authors: Wallace, Xiao, Leike et al. Claim to know: Training instruction priority can improve resistance to lower-trust instructions. It is not a formal isolation guarantee and does not replace tool permissions, data controls or attack evaluation.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
11. **C2PA Technical Specification**. Authors: Coalition for Content Provenance and Authenticity. Claim to know: C2PA content credentials bind signed provenance assertions under a verification/trust model. Cryptographic integrity does not certify the truth of every depicted claim; linguistic support/attribution and signed provenance answer different questions.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
12. **RAGRoute: Learning Which Sources to Query in Federated RAG**. Authors: Guerraoui, Kermarrec, Petrescu et al. Claim to know: Source selection as a learned decision, and the fan-out cost it avoids. Correction: the paper’s title is “Efficient Federated Search for Retrieval-Augmented Generation” (Guerraoui et al., EuroMLSys 2025, arXiv 2502.19280), and RAGRoute is the name of its routing method. Its arXiv v2 adds “using Lightweight Routing” to the title and lists Dhasade as first author.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
13. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. Authors: Dao, Fu, Ermon et al. Claim to know: FlashAttention computes exact attention with reduced high-bandwidth-memory traffic through IO-aware tiling. It does not prove all prefill is bandwidth-bound or remove the quadratic attention arithmetic.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
14. **Learning Transferable Visual Models From Natural Language Supervision** `[core]`. Authors: Radford, Kim, Hallacy et al. Claim to know: Study CLIP’s contrastive text-image training and the three example bias mechanisms analyzed in Chapter 39. They are not an exhaustive universal list for every multimodal retriever.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
15. **MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models**. Authors: Hu, Gu, Dou et al. Claim to know: What multimodal RAG benchmarks actually measure, and how many are text-answerable.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
16. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** `[core]`. Authors: Edge, Trinh, Cheng et al. Claim to know: GraphRAG. Know the community hierarchy, the map-reduce query path, and that index cost is the objection to answer first.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
17. **LightRAG: Simple and Fast Retrieval-Augmented Generation**. Authors: Guo, Xia, Yu et al. Claim to know: Compare LightRAG’s entity/relation-oriented retrieval and proposed incremental update procedure with the cited hierarchical-summary design. Refresh costs and quality depend on implementation and workload; lack of that hierarchy alone does not prove an update bound.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
18. **From Louvain to Leiden: Guaranteeing Well-Connected Communities**. Authors: Traag, Waltman, van Eck. Claim to know: Study Leiden’s connectivity guarantees under the stated algorithm/objective and Louvain’s possible disconnected communities. Neither method guarantees a globally optimal partition or semantically correct summaries.[\[16\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
### Where to start
**What.** Follow the source’s ten-paper order when preparation time is one week rather than one month.
**Why.** The route preserves the five decisions, two most important negative results, index arithmetic, and threat model.
**Failure without it.** A compressed plan can reorder the source or omit one of the senior-loop coverage areas.
**Cost and complexity.** The route contains ten papers and gives up the broader coverage of the full curated list.
If time allows ten papers, read them in this exact source order:
1. Lewis et al. on RAG.
2. Mallen et al. on when not to trust it.
3. Karpukhin et al. on DPR.
4. Thakur et al. on BEIR.
5. Malkov and Yashunin on HNSW.
6. Jegou et al. on product quantization.
7. Khattab and Zaharia on ColBERT.
8. Liu et al. on Lost in the Middle.
9. Min et al. on FActScore.
10. Greshake et al. on indirect prompt injection.
The starter set provides a compact introduction to selected design decisions, negative results, index arithmetic and indirect-injection risks. Its coverage is a study recommendation, not an empirically established description of what most senior interview loops ask.
## Diagrams
The source appendix contains zero figures and zero tables. No captioned source visual is recreated or invented here.
## Whiteboard pack
### What to draw
1. Draw a large rectangle labeled “161 selected readings.”
2. Above it, write “roughly 500 cited works” and draw a narrowing arrow into the rectangle.
3. Divide the rectangle into 14 vertical shelf sections.
4. Label the first shelf “What RAG is” and the last shelf “Trust and advanced variants.”
5. Draw one sample reading card inside a shelf.
6. Write “title,” “listed first authors,” and “claim to know” on the card.
7. Add a `[core]` seal to the sample card.
8. Write “52 core” beside the seal.
9. Draw a path through ten cards below the shelves.
10. Label that path “one-week starting route.”
11. Add a final note: “Know the result, not only the name.”
### Spoken script
Appendix E maps fourteen chapter clusters to 161 entries for 160 distinct titles. Fifty-two entries carry a priority marker. I should explain the mechanism, baseline, assumptions and limits rather than repeat a title or turn one study’s result into a universal law. I distinguish training FLOPs from wall time, PQ combinations from stored scalars, and a supported-count factuality proxy from true semantic recall. The ten-paper route is a starting point, not complete coverage.
## Interview traps
### Probe 1: Why is recognizing a famous paper title insufficient?
The source expects the specific result, mechanism, comparison, or limit attached to the title. A staff-level interviewer can turn recognition into a follow-up such as asking for the baseline and outcome in Lost in the Middle.
### Probe 2: Does `[core]` mean the other readings can be ignored?
No. The marker means a senior candidate should discuss that entry without prompting, while the full list remains the curated subset worth reading. The 52 marked entries set priority inside 161 readings rather than replacing the other clusters.
### Probe 3: Why does the list emphasize negative results?
Negative results reveal conditions under which a design fails, degrades accuracy or does not transfer. They are useful interview preparation, but this appendix does not establish their frequency in secondary sources or interviews, and their study conditions remain part of the result.
### Probe 4: Why include papers that are not about retrieval?
Distance concentration, tail latency, and instruction hierarchy still constrain a RAG system. Their mechanisms govern index choice, serving behavior, and the treatment of retrieved text even when the paper itself studies another problem.
### Probe 5: What does the ten-paper route preserve when time is short?
It keeps the source’s exact order from foundational RAG through indirect prompt injection. The source says the route covers five decisions, two important negative results, index arithmetic, and the threat model.
## Key numbers


| Number or expression | Source commitment | Why it matters |
| --- | --- | --- |
| Roughly five hundred works | Source estimate for the wider bibliography | Not independently re-enumerated as a distinct-work count here |
| $`161`$ entries; $`160`$ titles | Actual curated entry/title count | The DCG paper appears in two clusters |
| 14 clusters | Chapter-mapped reading groups | This is the exact group inventory |
| 52 `[core]` entries | Priority markers counted from the source list | These should be discussable without prompting |
| 10 starter papers | Ordered route for a reader with one week rather than one month | It is the source’s compressed study plan |
| 1 follow-up question | The source says staff-level interviewers can distinguish naming from knowing this quickly | The annotation, not title recognition, is the preparation target |
| 2 usage notes | Emphasize negative results and include non-retrieval constraints that govern RAG | These notes explain the curation policy |
| Roughly $`10\%`$ | Share of all PopQA questions where retrieval made GPT-3 (text-davinci-003) wrong on a question it answered correctly without retrieval, in Mallen et al. (2023) (corrected from reported retrieval harm on an otherwise-correct subset) | Conditional study result, not a universal all-query rate |
| Three empirical memorization trends | Scale, duplication and supplied-context associations | Not unrestricted mathematical laws |
| $`C\approx6ND`$ | Scoped dense-model training FLOP proxy | Parameters times cumulative processed tokens; no full-system overhead |
| $`r(d_{\mathrm{in}}+d_{\mathrm{out}})`$ | Trainable factors for one LoRA-adapted matrix | Excludes biases, other adapters and frozen-base computation |
| Sample count | A work driver for self-consistency | Prefix reuse, error dependence and parallelism change costs/latency |
| $`L(n)/p+1/(1-p)`$ | A skip-list expected-search bound form | Not inherited as an exact HNSW query bound |
| $`m_L=\frac{1}{\ln M}`$ | HNSW level multiplier | It is one constant the core annotation requires |
| $`M_0=2M`$ | HNSW base-layer relation | It is another required constant |
| $`4d+8M`$ bytes | fp32 payload and allocated base-layer links only | Add upper layers, IDs, allocation and runtime |
| $`k^m`$, $`km`$, $`kd`$ | Combinations, subcentroids, stored scalar values | Centroid bytes independent of m only at fixed k, d and precision |
| $`p_{99}`$ | Tail-latency percentile named in The Tail at Scale | Fan-out governs it in a sharded index |
| $`k_1`$ and $`b`$ | BM25 parameters named in the core annotation | Their placement in the formula matters |
| $`c_{\mathrm{RRF}}=60`$ | Empirical smoothing choice in cited RRF setup | Not a universal optimum or final context depth |
| 4 classifiers | Components of the unified active-retrieval decision tree | They operate over frozen hidden states |
| Eight examples per task | Promptagator seed setup | Task adaptation evidence, not a proof against transferable retrieval |
| Two-step AIS structure | Interpretation/explicature, then attribution | Use surrounding context; standalone grammar is not required |
| 3 RAGAs measures | Faithfulness, answer relevance, context relevance | Know what each measure omits |
| 6 aspects across 3 stages | Trustworthiness taxonomy | It organizes the credibility chapter |
| 2 failure mechanisms | Competing objectives and mismatched generalization | Jailbroken asks the reader to know both |
| Three example CLIP biases | Mechanisms analyzed in Chapter 39 | Not exhaustive universal multimodal failure laws |
| 5 decisions | Coverage of the ten-paper starter route | The source does not enumerate them again here |
| 2 negative results | Most important negative findings covered by the starter route | They are part of the senior-loop emphasis |
| Group 1: 7 entries, 2 core | What RAG is, and when it fails | Exact group inventory |
| Group 2: 9 entries, 1 core | Editing, continual learning, long context | Exact group inventory |
| Group 3: 9 entries, 3 core | Generator benefits and failures | Exact group inventory |
| Group 4: 3 entries, 1 core | Privacy, law, datastore | Exact group inventory |
| Group 5: 7 entries, 4 core | Scaling, economics, in-context learning | Exact group inventory |
| Group 6: 5 entries, 1 core | Reading the machine | Exact group inventory |
| Group 7: 11 entries, 3 core | Prompting, sensitivity, abstention | Exact group inventory |
| Group 8: 13 entries, 3 core | Representation, chunking, document structure | Exact group inventory |
| Group 9: 15 entries, 5 core | Vector indexes and compression | Exact group inventory |
| Group 10: 17 entries, 8 core | Ranking methods | Exact group inventory |
| Group 11: 16 entries, 5 core | Query understanding and control flow | Exact group inventory |
| Group 12: 14 entries, 6 core | Training retriever and generator | Exact group inventory |
| Group 13: 17 entries, 5 core | Context, attribution, evaluation | Exact group inventory |
| Group 14: 18 entries, 5 core | Trust, robustness, scale, advanced variants | Exact group inventory |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A4\_the\_rag\_card\_and\_index\_datasheet](A4_the_rag_card_and_index_datasheet.md) · [A6\_notation\_quick\_reference](A6_notation_quick_reference.md)
