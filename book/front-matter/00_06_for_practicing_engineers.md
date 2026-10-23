# For Practicing Engineers
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_05\_for\_interview\_candidates](00_05_for_interview_candidates.md) · [00\_07\_notation\_and\_symbols](00_07_notation_and_symbols.md)
This unit maps the technical content of Retrieval-Augmented Generation (RAG) to the decisions, failures, costs, and trade-offs that practicing engineers face.
## TL;DR
- Size stored data and request work together: both corpus size and request rate can affect memory, compute, bandwidth and I/O.
- Try reversible search-time settings before disruptive rebuilds when relevant, but first check correctness, eligibility and data/model defects.
- Route symptoms to hypotheses about retrieval, context use, support, freshness and safety; interactions can involve more than one stage.
- A single end-to-end score cannot uniquely identify a cause. Use matched tests, traces and controlled interventions.
- Retrieval and more context can help or harm under particular tasks; the cited ten-percent harm rate concerns an otherwise-correct subset, not all requests. Correction: Mallen et al. (2023, arXiv 2212.10511) found that for about $`10\%`$ of PopQA questions, retrieval augmentation made GPT-3 (text-davinci-003) answer incorrectly a question it answered correctly without retrieval. The $`10\%`$ is a share of all questions, not of an otherwise-correct subset.
- Start with Chapters 30, 34, 17 and 37, then use the decision map for the actual workload.
- Reuse sound derivations under their assumptions while refreshing hardware, prices, models and implementation details.
## The story
Think of stored corpus and request rate as two coupled factory gauges. Inventory, representation, replicas and runtime affect storage; request processing also consumes memory bandwidth, compute, I/O and concurrency capacity. Per-query work depends on corpus size and configuration, so these are not independent memory-only and compute-only axes.
The intervention ladder starts with inexpensive reversible settings when they address the real bottleneck. Some build changes require a rebuild, while others support incremental migration. An index-family change affects operations. Trace the actual serving stages and test competing causes rather than assuming every architecture has five stations or one station owns every regression.
The factory also receives outside material. Retrieved text can be unreliable or hostile, so the intake path can place untrusted material inside a privileged process. A longer workbench does not ensure that workers use evidence placed in its middle. A label attached after production may be topically related without proving where the answer came from. The book is the control-room manual. It routes each symptom to the right chapter, shows the measured cost of optimistic defaults, and ends every new-tool review with two checks: the constraint addressed and the trade-off accepted.
## Decoder table


| Term | Meaning in this unit | Why it matters |
| --- | --- | --- |
| RAG | Retrieval-Augmented Generation, evaluated by stage and as a whole | It anchors the system named throughout the engineering decision map. |
| Index | The search structure whose memory, build, freshness, and operational costs must be sized. | It anchors a term used in the engineering decision map. |
| Index memory arithmetic | The end-to-end calculation that turns corpus scale into index memory needs. | It anchors a term used in the engineering decision map. |
| Query rate | Requests per unit time multiply per-request service work | Interacts with corpus, configuration, compute, bandwidth, memory and concurrency |
| Recall | Whether retrieval returns the needed material. It trades against latency and memory. | It anchors a term used in the engineering decision map. |
| Latency budget | The time constraint within which retrieval and the other system stages must operate. | It anchors a term used in the engineering decision map. |
| Search-time parameters | Often reversible request-level settings | A default first experiment only when relevant to the measured defect |
| Build parameters | Structural/training settings for an index | Changes may require full rebuild, incremental maintenance or migration |
| Index family | The underlying index choice. Changing it also changes the operational story. | It anchors a term used in the engineering decision map. |
| HNSW | Hierarchical Navigable Small World graph index | It anchors a graph-based index family in the engineering decision map. |
| IVF | Inverted File index | It anchors a partition-based index family in the engineering decision map. |
| Disk-resident index | An index family option that keeps the index on disk. | It anchors a term used in the engineering decision map. |
| Vocabulary mismatch | The case where user queries do not look like the documents they should retrieve. | It anchors a term used in the engineering decision map. |
| Query expansion and rewriting | A Chapter 24 route for vocabulary mismatch. | It anchors a term used in the engineering decision map. |
| doc2query | A Section $`29.2`$ document-expansion route for vocabulary mismatch. | It anchors a term used in the engineering decision map. |
| Learned sparse model | A Section $`21.1`$ model that expands implicitly. | It anchors a term used in the engineering decision map. |
| Hybrid lexical and dense channel | A retrieval design that combines lexical and dense results. | It anchors a term used in the engineering decision map. |
| RRF | Reciprocal Rank Fusion for lexical and dense channels | It identifies the rank-only fusion method in Section $`21.6`$. |
| Context window | The generator input area where evidence position affects use. | It anchors a term used in the engineering decision map. |
| Top-$`k`$ selection | Selecting evidence items for context | Count is distinct from actual tokens, support and placement |
| Positional U-curve | Conditional quality pattern versus evidence position | Not a universal quality-versus-depth or context-length law |
| Closed-book answer | An answer produced without retrieval. It can outperform a retrieval-augmented answer in a stated regime. | It anchors a term used in the engineering decision map. |
| Tombstone | Logical marker used by the read/eligibility path | Not immediate erasure of every replica, backup or cache |
| Graph degradation | Possible search/maintenance effects of some changes | Implementation and deletion pattern matter; not inevitable |
| Re-indexing | Rebuilding the index as the corpus changes. | It anchors a term used in the engineering decision map. |
| Sharding | A vector-store operating concern covered in Chapter 17. | It anchors a term used in the engineering decision map. |
| Replication | A vector-store operating concern covered in Chapter 17. | It anchors a term used in the engineering decision map. |
| Metadata filtering | A restriction on retrieval results that can silently empty a result set. | It anchors a term used in the engineering decision map. |
| Filtering and recall | Eligibility constraints can reduce available neighbors | Enforce security/ACLs; distinguish true absence from candidate-search failure |
| Reranking | Reordering retrieved candidates after retrieval. | It anchors a term used in the engineering decision map. |
| Rerank depth | How many candidates enter reranking, with a trade-off surface engineers must size. | It anchors a term used in the engineering decision map. |
| Engineering decision map | The source’s two-column route from a live decision or problem to its primary chapters. | It anchors a term used in the engineering decision map. |
| Multi-hop retrieval | Retrieval that follows more than one step. | It anchors a term used in the engineering decision map. |
| Iterative retrieval | Retrieval that repeats as the system develops the answer. | It anchors a term used in the engineering decision map. |
| Grounding | Producing an answer from supporting retrieved material. | It anchors a term used in the engineering decision map. |
| Judgment set | The evaluation set named alongside retrieval metrics. | It anchors a term used in the engineering decision map. |
| Faithfulness | Whether an answer remains supported by its evidence. | It anchors a term used in the engineering decision map. |
| Macro, micro, mezzo | Diagnostic views for regression hypotheses | Controlled evidence may reveal multiple causes or interactions |
| Remove-the-evidence ablation | The Section $`34.4`$ diagnostic that tests behavior after evidence is removed. | It anchors a term used in the engineering decision map. |
| Adaptive gating | The decision about whether to retrieve for a query. The book treats it as the first design decision. | It anchors a term used in the engineering decision map. |
| High-popularity questions | The stated question regime where retrieval-augmented accuracy falls below closed-book accuracy. | It anchors a term used in the engineering decision map. |
| Learned controller | A router, retrieval gate, or confidence-triggered loop trained on a query distribution. | It anchors a term used in the engineering decision map. |
| Distribution shift | A change in request distribution | Can alter policy outcomes; degradation is not logically inevitable |
| Semantic chunking | Meaning-related boundary selection | Transfer depends on corpus and evaluation; stitched benchmarks can favor it |
| Post-hoc citation | Reference attached after generation | Can support a claim without proving historical model use |
| Indirect prompt injection | Hostile instructions carried in external content | Retrieved data must not acquire privileged instruction authority |
| Datastore poisoning | Hostile content written into the source that retrieval can expose to the system. | It anchors a term used in the engineering decision map. |
| Multi-tenancy | A Chapter 38 concern for serving multiple tenants. | It anchors a term used in the engineering decision map. |
| Residency | A Chapter 38 concern about where data or service state must remain. | It anchors a term used in the engineering decision map. |
| Federation | A Chapter 38 concern for routing across separated sources. | It anchors a term used in the engineering decision map. |
| PDF-shaped corpus | A Portable Document Format (PDF) corpus whose tables and layout need the Chapter 14 treatment | It routes layout-sensitive ingestion to the correct chapter. |
| LLM-based graph extraction | A separate token-bill component in the worked scenario | One million chunks; not a complete graph-versus-vector build bill |
| Graph RAG | Retrieval using graph-structured information | Can serve relations, provenance, existing structured data or global queries; measure added value |
| Static corpus | Unchanging or mostly unchanged data over a declared horizon | A valid workload case; include actual churn when present |
| RAG card | An Appendix D reporting template for what a system retrieves, how it was evaluated, and what it abstains on. | It anchors a term used in the engineering decision map. |
| Index datasheet | An Appendix D reporting template for what an index costs. | It anchors a term used in the engineering decision map. |
| Design checklist | An Appendix C artifact meant to be run during a design review. | It anchors a term used in the engineering decision map. |
| Cost per query | The stage-by-stage cost measure covered in Chapter 37. | It anchors a term used in the engineering decision map. |
| Reusable principles | Definitions, units and derivations under stated assumptions | Implementation coefficients, hardware and workload conclusions can change |
| High-dimensional geometry | Geometry affecting distance distributions and pruning | Does not prove exact search always fails |
| Exact search | Search without approximation under the chosen scorer | Can be suitable for small/batched/GPU workloads or strict recall needs |
| Bandwidth arithmetic | The index calculation named alongside memory arithmetic. | It anchors a term used in the engineering decision map. |
| Token economics | The stable cost reasoning attached to token use. | It anchors a term used in the engineering decision map. |
| Evolving implementation layer | Models, databases, frameworks and rerankers | Refresh by change risk; no universal quarterly cadence |
| Orchestration framework | One volatile tool category that no book can track continuously. | It anchors a term used in the engineering decision map. |
| Constraint | The limitation a new technique claims to address. | It anchors a term used in the engineering decision map. |
| Trade-off | What the new technique gives up in exchange. | It anchors a term used in the engineering decision map. |
| $`k`$ | Number of evidence items presented | Choose from measured support, quality, token and latency trade-offs |


## Core mechanics
### 1. Engineering focus
#### What
The book uses interview questions as its organization, but almost none of its technical content is interview-specific. It covers index-sizing arithmetic, the trade-off surface behind rerank depth, and stage attribution for end-to-end accuracy regressions.
#### Why
Engineers perform the same calculations on the job under the same constraints and usually with less time. This unit maps chapters to the decisions they actually face.
#### Failure without it
An engineer can mistake the book for interview preparation and miss its direct use in planning, diagnosis, design review, and operations.
#### Stated cost
The source states a time constraint rather than a numeric price. On-the-job decisions usually allow less time.
### 2. When to Reach for This Book
#### You are sizing an index and someone asks how many machines
- **What:** Use Section $`16.5`$ for declared storage terms, then Section $`41.1`$ to combine workload and capacity assumptions.
- **Why:** Vector count $`N_v`$, dimension, precision, graph/ID terms and replicas affect storage. Request rate $`Q`$ multiplies per-query service demand, which also depends on the corpus, candidates, filters, caches and hardware.
- **Failure without it:** A storage-only or throughput-only plan can miss bandwidth, concurrency, placement, redundancy or tail-latency constraints.
- **Cost boundary:** Payload arithmetic and throughput lower bounds are not a complete node count. Benchmark the target configuration and reserve failure/rebuild headroom.[\[1\]](../part-05-indexing/16_compression_and_index_economics.md)[\[2\]](../part-12-systems-design/41_end_to_end_design_drills.md)
#### Recall is short of target and you have a latency budget
- **What:** Prefer reversible search-time tuning when it can address the measured recall/latency shortfall, then consider build settings and index family.
- **Why:** Reversibility often lowers experiment cost, but a malformed corpus, bad embeddings or an incorrect eligibility path needs a different fix.
- **Failure without it:** Either jumping to a rebuild or blindly tuning search can waste work.
- **Cost boundary:** Many structural changes require rebuilding or migration; some implementations support incremental changes. Do not assume every build setting forces a complete rebuild, and never weaken required access controls to raise recall.[\[3\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[4\]](../part-05-indexing/17_operating_a_vector_store.md)
#### Your users’ queries do not look like your documents
- **What:** Vocabulary mismatch has several routes, including query expansion/rewriting, doc2query document expansion, learned sparse expansion, and lexical/dense hybrid fusion with RRF.
- **Why:** These address different mismatches and are not an exhaustive or universally ranked list.
- **Failure without it:** Applying one familiar transformation can miss the actual representation, eligibility or relevance problem.
- **Cost boundary:** Compare indexing, training, request-time work and regression risk under a matched baseline; this page assigns no universal monetary amount.[\[5\]](../part-06-retrieval-ranking/21_learned_sparse_and_multi_vector_retrieval.md)[\[6\]](../part-07-query-control/24_query_reformulation.md)[\[7\]](../part-08-training/29_bootstrapping_training_data.md)
#### The retrieved document is right and the answer is still wrong
- **What:** Use Chapter 30 to test evidence position and assembly; a longer window need not fix a positional failure. Use Chapter 31 when a correct answer has an unsupported citation.
- **Why:** Retrieving appropriate text does not guarantee its use or a valid citation.
- **Failure without it:** Changing search may not address an assembly, generation or support problem; several stages can interact.
- **Cost boundary:** Measure the relevant intervention rather than assuming fewer or more tokens is always better.[\[8\]](../part-09-context-assembly/30_lost_in_the_middle.md)[\[9\]](../part-09-context-assembly/31_attribution_and_citation.md)
#### The index has to stay fresh and the corpus will not sit still
- **What:** Use Chapter 17 for logical deletion, read-time eligibility, maintenance and implementation-dependent effects on graph search; Section $`41.2`$ develops a freshness plan.
- **Why:** Workloads may change, though some corpora are static or mostly static.
- **Failure without it:** Ineligible/deleted data can remain reachable, or maintenance/search work can grow under particular deletion patterns.
- **Cost boundary:** Benchmark patching, compaction and rebuilding. Tombstones do not imply universal graph degradation or complete erasure of backups, caches and replicas.[\[4\]](../part-05-indexing/17_operating_a_vector_store.md)[\[2\]](../part-12-systems-design/41_end_to_end_design_drills.md)
#### Quality dropped and nobody changed anything
- **What:** Use macro, micro and mezzo views plus evidence-removal and other controlled ablations to test a regression. Preserve matched cohorts, versions and uncertainty.
- **Why:** An end-to-end decline is an outcome, not a unique stage diagnosis.
- **Failure without it:** Teams may repair the most familiar stage while missing data, labels, workload or interactions.
- **Cost boundary:** Diagnostics narrow hypotheses; no fixed five-stage map guarantees one cause or one corrective action.[\[10\]](../part-10-evaluation/34_evaluating_the_system.md)
#### Someone can write into your datastore
- **What:** Use Chapter 36 for indirect injection through retrieved documents, web/tool results and other external content. Datastore write access is a strong poisoning vector, not a prerequisite for every injection.
- **Why:** Retrieved text must remain untrusted data; inclusion in context does not legitimately grant instruction authority.
- **Failure without it:** The system may mistake hostile data for trusted instructions or execute unsafe actions.
- **Cost boundary:** Enforce permissions and instruction/data boundaries, validate tool actions and test realistic attacks. Provenance and filtering reduce particular risks without proving universal safety.[\[11\]](../part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
### 3. Chapter Map by Engineering Decision
#### What
Use the following map to route an engineering decision to its primary chapters.


| Decision or problem | Primary chapters |
| --- | --- |
| Whether to retrieve at all | Chapters 1 and 25 |
| Choosing chunk size and overlap | Chapter 13 |
| Tables, layout, and PDF-shaped corpora | Chapter 14 |
| Choosing an embedding model | Chapters 12 and 20 |
| Index family, including HNSW, IVF, and disk-resident | Chapter 15 |
| Compression and index memory | Chapter 16 and Section $`41.1`$ |
| Freshness, deletion, and re-indexing | Chapter 17 and Section $`41.2`$ |
| Metadata filtering and over-filtering | Sections $`17.5`$ and $`17.6`$ |
| Adding a lexical channel and hybrid fusion | Chapter 18 and Section $`21.6`$ |
| Reranking and rerank depth | Chapter 22 |
| Query rewriting and conversational turns | Chapter 24 |
| Routing across multiple sources | Chapters 25 and 38 |
| Multi-hop and iterative retrieval | Chapter 26 |
| Fine-tuning the generator for grounding | Chapter 27 |
| Training or adapting the retriever | Chapter 28 |
| Manufacturing retrieval training data | Chapter 29 |
| Context ordering and top-k truncation | Chapter 30 |
| Citations and attribution | Chapter 31 |
| Retrieval metrics and judgment sets | Chapter 32 |
| Faithfulness and answer-quality metrics | Chapter 33 |
| Diagnosing regressions across stages and interactions | Chapter 34 |
| Conflicting or unreliable sources | Chapter 35 |
| Prompt injection and datastore poisoning | Chapter 36 |
| Latency budget and cost per query | Chapter 37 |
| Multi-tenancy, residency, and federation | Chapter 38 |
| Images, video, and non-text corpora | Chapter 39 |
| Graph relationships, provenance and global queries | Chapter 40 |


#### Why
The map begins with the engineering problem rather than the book sequence. It gives a direct path from a live decision to the relevant material.
#### Failure without it
An engineer must search the full chapter sequence instead of starting from the problem at hand.
#### Stated cost
The map itself states no cost. Several routes point to cost-bearing decisions, including index memory, rebuilds, rerank depth, latency, and cost per query.
### 4. If You Only Read Four Chapters
#### Chapter 30, Lost in the Middle
- **What:** Test whether added evidence, changed placement or a longer window helps the target workload.
- **Why:** Lost-in-the-middle results measure quality versus evidence position under particular models/tasks. They are not a universal law in $`k`$ or context length.
- **Failure without it:** Increasing context can waste work or introduce distractors even when candidate inclusion improves.
- **Cost boundary:** A smaller $`k`$ can be a quality intervention, but it can also remove necessary evidence. Measure support, answer quality, latency and cost jointly.[\[8\]](../part-09-context-assembly/30_lost_in_the_middle.md)
#### Chapter 34, Evaluating the System
- **What:** Apply the system-evaluation chapter as a diagnostic method, not a mechanical guarantee of exactly one cause among five stages.
- **Why:** A decline can reflect multiple stages, interactions, changing labels or workload mix.
- **Failure without it:** A team can confuse an aggregate symptom with a causal explanation.
- **Cost boundary:** Traces and controlled tests require effort. The book does not establish that a chapter pays for itself at the first regression.[\[10\]](../part-10-evaluation/34_evaluating_the_system.md)
#### Chapter 17, Operating a Vector Store
- **What:** Account for tombstones, sharding, replication, eligibility filtering, maintenance and implementation-dependent search behavior.
- **Why:** Many production corpora change; static or mostly static workloads are also possible.
- **Failure without it:** A static benchmark can omit relevant update, deletion, replica and read-path costs.
- **Cost boundary:** Match the actual churn and failure model rather than asserting all production data is dynamic or every deletion needs a full rebuild.[\[4\]](../part-05-indexing/17_operating_a_vector_store.md)
#### Chapter 37, Latency, Cost, and Systems
- **What:** Measure stage-level latency, total work and monetary cost separately, including queueing, I/O, caches and fixed capacity.
- **Why:** Intuition can misidentify a bottleneck, but this page does not quantify how often or by what universal factor.
- **Failure without it:** Optimizing one stage can leave another dominant or merely move the bottleneck.
- **Cost boundary:** Tokens, FLOPs and idealized throughput are proxies unless calibrated to the target serving system; none alone is an invoice or tail-latency benchmark.[\[12\]](../part-12-systems-design/37_latency_cost_and_systems.md)
### 5. Where This Book Disagrees with the Consensus
#### What
The book treats retrieval folklore as unusually optimistic. It attaches measurements to positions that cut against that folklore so engineers can check them.


| Position | Why it matters | Failure without it | Stated cost |
| --- | --- | --- | --- |
| Retrieval is not a free improvement | The cited result flips a correct answer to wrong on about $`10\%`$ of all PopQA questions (corrected from a roughly $`10\%`$ harm rate on the otherwise-correct closed-book subset).[\[13\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md) | Always-retrieve or always-bypass can violate task needs. | Measure action benefit, work and evidence-policy compliance. |
| More context is not automatically better | Positional effects are measured under particular models, tasks and context setups.[\[8\]](../part-09-context-assembly/30_lost_in_the_middle.md) | A larger window may retain distractors or poor placement. | Select depth and actual tokens by matched quality, support, cost and latency. |
| Semantic-chunking gains need transfer tests | Stitched unrelated texts can favor topic-boundary detection; real-document effects differ.[\[14\]](../part-04-representations/13_chunking_and_granularity.md) | A team assumes a benchmark gain transfers without matching corpus structure. | No universal failure or benefit follows; compare fair baselines. |
| Post-hoc support is not proof of model use | An added citation can entail a claim without being the source actually used.[\[9\]](../part-09-context-assembly/31_attribution_and_citation.md) | Citation presence, support and historical attribution are conflated. | Validate support separately; attribution design adds work and remains fallible. |
| Learned controllers need shift monitoring | A changed request distribution can alter calibration and action quality.[\[15\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md) | A team assumes an in-distribution policy remains optimal without checking. | Monitor outcomes and evidence requirements; not every shift degrades every controller. |
| A graph must justify its added work | Relations, provenance, existing structures or global queries can motivate graph retrieval.[\[16\]](../part-12-systems-design/40_graph_rag.md) | Graph extraction is applied without measured incremental value. | The worked extraction/embedding token bills are $`\$2{,}850`$ versus $`\$6`$ for one million chunks, a $`475`$-fold component ratio, not an all-in build ratio. |


#### Why
Each disagreement converts an optimistic default into a decision with a measurable failure mode or explicit trade-off.
#### Failure without it
Teams can always retrieve, always add context, trust benchmark gains, attach citations after generation, ignore query shift, or build a graph without testing whether the corpus justifies those choices.
#### Stated cost
Two numerical examples need explicit scope. The roughly $`10\%`$ retrieval-harm figure concerns the cited otherwise-correct closed-book subset. Correction: Mallen et al. (2023, arXiv 2212.10511) found that for about $`10\%`$ of PopQA questions, retrieval augmentation made GPT-3 (text-davinci-003) answer incorrectly a question it answered correctly without retrieval. The $`10\%`$ is a share of all questions, not of the otherwise-correct closed-book subset. The graph-cost scenario uses $`10^6`$ chunks, not necessarily documents, each with $`300`$ source tokens: $`3\times10^8`$ tokens at $`\$0.02`$ per million gives $`C_{\mathrm{emb}}=\$6`$. Extraction uses $`450`$ input and $`100`$ output tokens per chunk at $`\$3`$ and $`\$15`$ per million, giving $`C_{\mathrm{ext}}=\$1{,}350+\$1{,}500=\$2{,}850`$ and token-bill ratio $`2{,}850/6=475`$. This excludes index construction/operation, resolution, summaries, storage, retries and other work; it is not a ratio of complete graph and vector-system builds or a live provider quote.[\[13\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)[\[16\]](../part-12-systems-design/40_graph_rag.md)
### 6. Artifacts You Can Take to Work
#### What
Appendix C provides design checklists for retrieval, indexing, evaluation, credibility, and latency. They are written for use in a design review rather than passive reading.
Appendix D provides the RAG card and index datasheet. These reporting templates state what a system retrieves from, how it was evaluated, what it abstains on, and what its index costs.
#### Why
Retrieval systems pass between teams more often than they are rewritten. A new owner’s questions are predictable enough to answer in advance.
#### Failure without it
A handoff can omit the system’s sources, evaluation boundary, abstention behavior, or index cost.
#### Stated cost
No numeric cost is stated. The artifacts reduce repeated ownership questions and missing handoff information.
### 7. Staying Current
#### What
The reusable layer includes explicitly defined geometry, arithmetic, units, objectives and evaluation constructs. High dimension can make particular exact-tree pruning ineffective, but exact scans can remain competitive depending on corpus size, batching, hardware and recall requirements. A principle remains valid under its assumptions; a practical coefficient or best design can change.[\[3\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[1\]](../part-05-indexing/16_compression_and_index_economics.md)[\[17\]](../part-10-evaluation/32_evaluating_retrieval.md)[\[12\]](../part-12-systems-design/37_latency_cost_and_systems.md)
Models, databases, frameworks and rerankers evolve. “Every quarter” is a source illustration, not a measured universal cadence. Refresh benchmarks when the workload, implementation, model or requirements change.
#### Why
When a new technique appears, ask two questions. Which constraint does it address? What does it trade away?
#### Failure without it
For a proposed technique, ask which constraint it addresses, which matched baseline supports it and what cost or risk it adds. A missing explanation is a reason to request evidence, not logical proof that no measurement was performed.
#### Stated cost
This page gives no fixed monetary amount for keeping evaluations current. Re-evaluation frequency should follow change risk and deployment needs, not an asserted universal quarterly schedule.
## Diagrams
The manifest records zero numbered figures and zero numbered tables for this unit. Page xxiv contains one unnumbered, two-column engineering decision map, reproduced above as Markdown. It has no number or caption, so the manifest count remains 0/0.
## Whiteboard pack
### Numbered draw order
1. Draw corpus/representation and request/workload gauges, with shared arrows to memory, compute, bandwidth, I/O and concurrency.
2. Draw search-time tuning, structural changes and index-family changes as a default intervention ladder, with exceptions for correctness and eligibility defects.
3. Mark reversible settings, implementation-dependent rebuild/migration, and operational redesign.
4. Trace actual serving stages and mark competing or interacting regression hypotheses, not one guaranteed culprit.
5. Plot quality versus evidence position for a stated model/context condition, labeling start, middle and end.
6. Draw external retrieved content behind an untrusted-data boundary, never as newly privileged instructions.
7. End with: which constraint, which evidence, which trade-off, and what reversal condition?
### 90 to 100 word script
Size storage and request service demand jointly, then test the actual bottleneck. Reversible tuning is often a good first experiment, but not a substitute for correcting eligibility or data defects. Trace regressions through matched interventions and allow interacting causes. Positional failures are conditional, not a universal longer-window impossibility. Retrieved text remains untrusted; distinguish claim support from proof of model use and price each proposed intervention.
## Interview traps
### 1. Why not make retrieval the default for every query?
Retrieval can help or harm. The cited roughly $`10\%`$ harm rate concerns an otherwise-correct subset under specific models/tasks, not all queries. Correction: Mallen and colleagues (2023, arXiv 2212.10511) found that for about $`10\%`$ of PopQA questions, retrieval augmentation made GPT-3 (text-davinci-003) answer incorrectly a question it answered correctly without retrieval. The $`10\%`$ is a share of all questions, not of an otherwise-correct subset. Gating is a useful design question, but currentness, source support and safety requirements can mandate retrieval even when the model seems confident.[\[13\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)[\[15\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
### 2. If evidence is lost in the middle, why not use a longer context window?
A longer window need not remove observed middle-position loss in a particular setup. Test placement and evidence selection on the target model; fewer documents can help by removing distractions but can also remove necessary evidence.[\[8\]](../part-09-context-assembly/30_lost_in_the_middle.md)
### 3. What should you tune when recall misses its target under a latency budget?
Try reversible search-time settings first when they address the actual shortfall. Structural tuning may require rebuilding or migration depending on implementation. Correct data, model and eligibility defects directly, and never relax required access controls for recall.[\[3\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[4\]](../part-05-indexing/17_operating_a_vector_store.md)
### 4. Why is a post-hoc citation weak grounding evidence?
A post-hoc citation may provide valid support for a claim. It does not by itself prove that the cited source was used in the original generation. Evaluate entailment/support and historical use as separate claims; integrating attribution into generation can help traceability but is not proof of universal faithfulness.[\[9\]](../part-09-context-assembly/31_attribution_and_citation.md)
### 5. How do you diagnose a quality drop when nobody changed the system?
Use versioned matched tests, stage traces and targeted ablations to investigate an end-to-end decline. Macro, micro and mezzo views guide the search; several stages or interactions may be causal. Repair supported causes and re-evaluate rather than promising a unique stage diagnosis.[\[10\]](../part-10-evaluation/34_evaluating_the_system.md)
## Key numbers


| Number | Meaning |
| --- | --- |
| 7 reach cases | The source routes seven recurring engineering situations to specific chapters and sections. |
| 27 decision rows | The engineering decision map contains 27 problem-to-chapter mappings. |
| Four listed routes | Examples for vocabulary mismatch, not an exhaustive set; compare their distinct costs |
| Illustrative five-stage map | An end-to-end score cannot uniquely diagnose causes; interactions are possible |
| 4 chapters | The recommended short path is Chapters 30, 34, 17, and 37. |
| Six positions | Conditional cautions rather than universal impossibility claims |
| Roughly $`10\%`$ | Share of all PopQA questions on which retrieval flipped a correct answer to wrong in the cited result (corrected from cited otherwise-correct subset flipped by retrieval, not all user requests) |
| Retrieval-gating decision | A useful organizing question constrained by support/currentness and safety policy |
| Order-of-magnitude errors | Possible intuition failures, not a quantified claim about most engineers |
| $`10^6`$ chunks | Not necessarily documents; assumes three hundred source tokens per chunk |
| $`\$2{,}850/\$6=475`$ | Extraction versus embedding token bills only; excludes complete build/operating costs |
| 5 checklist areas | Appendix C covers retrieval, indexing, evaluation, credibility, and latency. |
| 2 reporting templates | Appendix D provides the RAG card and the index datasheet. |
| 2 stages, then 1 system | Retrieval and generation must be evaluated separately and then together. |
| 2 review questions | Ask which constraint a new technique addresses and what it trades away. |
| Refresh when changes matter | Quarterly is illustrative, not a universal change or evaluation law |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_05\_for\_interview\_candidates](00_05_for_interview_candidates.md) · [00\_07\_notation\_and\_symbols](00_07_notation_and_symbols.md)
