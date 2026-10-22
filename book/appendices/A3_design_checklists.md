# Appendix C: Design Checklists
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A2\_question\_bank](A2_question_bank.md) · [A4\_the\_rag\_card\_and\_index\_datasheet](A4_the_rag_card_and_index_datasheet.md)
Use these five operational checklists to turn a Retrieval-Augmented Generation (RAG) design review into named decisions, measured quantities, and explicit exit criteria.
## TL;DR
- Run these checklists. Do not merely read them. Each line needs a number or a named decision.
- Internalize the retrieval checklist for interviews. It covers five decisions in ten minutes, with two minutes per phase.
- Run the indexing checklist before provisioning and before migrations. Derive memory, recall, latency, churn, and rebuild requirements.
- Run the evaluation checklist before trusting a dashboard. It must locate failures across retrieval, generation, and the full system.
- Run the credibility checklist before outsiders can write to the datastore. Trace sources, test attribution, limit attackers, and enforce access during retrieval.
- Run the latency checklist with a stopwatch on a real query. Measure five stages, optimize the dominant constraint, and derive cost per query.
- Treat every exit criterion as a gate. If the criteria fail, the design is not finished.
## The story
Picture an aircraft crew preparing for departure. The crew does not admire five laminated cards and call the plane ready. They run every line. A box receives a measured value or a named choice. A box filled with “fast,” “large,” or “safe” remains open.
The retrieval card is the route card. It asks how large the trip is, when to leave the gate, which airports to use, how to navigate, and how to brief the passengers. In system terms, it fixes problem size, the retrieval decision, the data sources, the search method, and the evidence shown to the generator.
The capacity card counts source items, chunks, stored vectors, identifiers, links, text, replicas and rebuild artifacts separately. It distinguishes exact-neighbor recovery by ANN from human-judged task relevance; either can fail while the other looks good.
The evaluation card calibrates the cockpit instruments. It checks the test set before flight, separates retrieval readings from generation readings, and traces a bad result to its owning stage. A single end-to-end score is like one warning light for the whole aircraft. It cannot tell the crew where the fault lives.
The credibility and robustness card secures the cabin and cargo hold. It records where each document came from, what happens when sources conflict, whether citations support the answer, who can load content, and what one poisoned document can damage. It also enforces each passenger’s access rights during retrieval, before private material reaches the model.
The latency and cost card starts the stopwatch and checks fuel use. It measures embedding, approximate search, reranking, prompt processing, and answer decoding under real load. The crew reduces the dominant delay first and records what quality each cheaper choice gives up.
Departure requires five exit stamps. If any card lacks its stamp, the aircraft stays at the gate. The honest review says which requirement remains open. It does not hide the gap behind a confident adjective.
## Decoder table


| Technical term | Plain-English meaning | Why it matters |
| --- | --- | --- |
| Retrieval-Augmented Generation (RAG) | A system that retrieves external material and gives it to a generator | The five checklists cover its retrieval, storage, evaluation, trust, latency, and cost decisions |
| Operational checklist | A list that must produce measurements or named choices | Reading a line without answering it does not finish the design |
| Exit criteria | Conditions that must hold before a checklist passes | A failed exit criterion means the design is not finished |
| Retrieval design skeleton | Five decisions about size, retrieval timing, sources and transformations, search, and evidence presentation | It gives an interview answer a complete structure |
| Corpus size in chunks | The number of retrievable units in the collection | It drives memory independently of traffic |
| Chunk size | The amount of content in each retrievable unit | It helps determine corpus count, memory, and prompt tokens |
| Queries per second (QPS) | The median and peak request rates | They drive compute independently of corpus size |
| Latency budget | The allowed response time | It turns “fast” into a measurable requirement |
| Time-to-first-token | Delay before the answer begins | A product may care about this more than total latency |
| Total latency | Delay until the answer finishes | It captures the whole user wait |
| Freshness requirement | Permitted source-to-serving lag | Constrains feasible options; no one-to-one minutes/hours/days architecture rule |
| Accuracy bar | A named metric with a cutoff | An adjective cannot determine whether the design passes |
| Retrieval gate $`g(x)`$ | The explicit rule that decides whether a query retrieves | Always retrieving is a measured choice, not a neutral default |
| Query classifier | A gate that predicts whether a query needs retrieval | It is one named gating family |
| Generator confidence | A generator signal used to decide whether to retrieve | It is another named gating family |
| Self-reflective token | A model-produced signal that requests retrieval | It is another named gating family |
| Closed-book path | The path that answers without retrieved evidence | The design must say what this path returns and whether it may abstain |
| Abstention | Refusing to answer | It can protect quality, but it changes coverage |
| Controller drift | Degradation in a learned retrieval gate | Query distribution shift can make the gate fail |
| Query distribution shift | Change in the query population | Can degrade a learned gate; monitor rather than assume inevitable degradation |
| Store | A source collection that retrieval can search | Multiple stores require selection and disagreement rules |
| Source-selection rule | The rule that chooses which stores to search | It prevents an undefined multi-source path |
| Query transformation $`\Phi`$ | The operation applied to a query before search | The design must name the operation and when it fires |
| Identity transformation | Searching with the query unchanged | It is one allowed query transformation |
| Query compression | Shortening a query before search | It is one allowed query transformation |
| Query expansion | Adding search terms to a query | It is one allowed query transformation |
| Query rewriting | Rephrasing a query for retrieval | Conversational rewrites must resolve referents beyond the current turn |
| Query decomposition | Splitting a query into smaller search questions | It is one allowed query transformation |
| Conversational referent | A word whose meaning depends on earlier turns | A rewrite that misses it can search for the wrong subject |
| Chunking scheme | The rule that divides content into retrievable units | It needs evidence from the actual corpus |
| Semantic chunking | Chunking based on meaning boundaries | Its value must be measured on the target corpus, not borrowed from a benchmark |
| Scoring function | The formula that ranks retrieved items | It defines what the search engine considers a good match |
| Lexical channel | Search based on words or exact terms | Dense-only search needs justification for long-tail and exact-match queries |
| Dense-only retrieval | Retrieval that uses only vector similarity | It may miss query types served by a lexical channel |
| Long-tail query | A rare query pattern | It is a specific stress case for dense-only retrieval |
| Exact-match query | A query that depends on matching a precise term | It is another stress case for dense-only retrieval |
| Hybrid retrieval | Retrieval that combines dense and lexical channels | It requires a named fusion method and constant |
| Fusion method | The rule that combines rankings from retrieval channels | An unnamed fusion rule leaves ranking behavior undefined |
| Fusion constant | The fixed parameter used by the fusion rule | It makes the hybrid design reproducible |
| Retrieval depth $`k^{\prime}`$ | The number of candidates retrieved before later processing | It differs from rerank depth and generator context count |
| Rerank depth | The number of retrieved candidates sent to the reranker | It controls reranking work and candidate coverage |
| Generator $`k`$ | The number of chunks sent to the generator | It determines evidence count and prefill tokens |
| Index family | The data structure used for nearest-neighbor search | It trades recall, latency, memory, and rebuild cost |
| Recall target | A specified recovery target | Distinguish ANN exact-neighbor recall from task relevance |
| Metadata filter | A document-field eligibility or preference rule | Fallbacks must not relax mandatory access or policy constraints |
| Over-filtering fallback | A defined response to an empty/weak eligible result set | Abstain or repair permissible preferences; do not bypass authorization |
| Prefill token count | Fully serialized tokenized input | Equal-size chunk count estimates evidence tokens only |
| Evidence ordering | The placement of retrieved evidence in the prompt | The design must address evidence that lands in the middle of context |
| Serialization format | The structure used to place retrieved text in the prompt | It fixes where evidence sits relative to instructions |
| Attribution mechanism | The process that connects answer claims to evidence | Generating citations with the answer differs from attaching them later |
| Vector precision | The bytes used to store each vector value | It is part of vector storage sizing |
| $`N`$ | Number of stored vectors in this capacity calculation | Chunks can yield one or several vectors; state that mapping |
| $`d`$ | The number of dimensions in each stored vector | It is stated with $`N`$ and vector precision before computing vector bytes |
| Vector bytes $`b_{\mathrm{vec}}`$ | The storage used by one vector | The indexing checklist requires this number before other sizing |
| Graph-index overhead $`b_{\mathrm{link}}`$ | Per-vector storage for graph links | It must be counted separately from vector bytes |
| Inverted file (IVF) | An index family that uses postings and searched partitions | Its posting identifiers and search controls affect storage and recall |
| Posting identifier | A stored reference inside an IVF posting list | It adds index overhead beyond vector bytes |
| Compression code size | Bytes left after vector compression | The resulting recall cost must be measured, not assumed |
| Rebuild headroom $`h`$ | Capacity reserved for actual live/build overlap | A twenty-percent reserve is not a complete extra full-size index |
| Replication factor $`R`$ | The number of stored copies | It multiplies fleet capacity needs |
| Payload store | Text and filterable metadata under a placement policy | Can dominate after compression; the crossover is workload specific |
| Shard count | Logical corpus partitions | Ceiling division of aggregate bytes is a capacity lower bound, not the shard plan |
| Random access memory (RAM) | Usable memory on an index node | It limits how much index footprint one node can hold |
| $`\mathop{\mathrm{Recall}}\nolimits@k^{\prime}`$ | Candidate recall at retrieval depth | Define evidence judgments separately from exact-neighbor approximation |
| Exact search | A search that establishes the neighbor ground truth | It is the comparison target for recall |
| Held-out sample | Data reserved for evaluation | Recall must be measured on data outside the tuning path |
| Relevance judgment | A label about result usefulness | It measures a different question from exact-neighbor recall |
| $`\mathrm{ef}`$ | A per-query graph-index search control | It is reversible and should be tried before rebuilding |
| $`n_{\mathrm{probe}}`$ | The number of IVF partitions searched per query | It is a reversible search-time control |
| Build-time knob | A construction/training setting | Rebuild requirements depend on implementation and the setting changed |
| Production concurrency | The real number of simultaneous requests | Latency measured without it can misrepresent deployed behavior |
| Tombstone | A logical read-time deletion marker | Some implementations support other deletion/compaction behavior; verify visibility |
| Tombstone fraction | The share of entries marked deleted | Its measured recall effect sets the rebuild trigger |
| Rebuild cadence | How often the index is rebuilt | Operations must know its duration and placement |
| Hot-swap | Replacing an old index with a separately rebuilt one | It is an alternative to rebuilding in place |
| Embedding model version | Pinned document and query representations | Document-space changes can need re-embedding; tested versioned migrations can exist |
| Judgment set | The queries, judges, relevance scale, and agreement used for evaluation | It defines the denominator and labeling process |
| Inter-annotator agreement | Agreement among human judges | It shows whether relevance labels are stable |
| Training-data contamination | Evaluation examples appearing in generator training data | It can make the test set uninformative |
| Retrieval benchmark wearing RAG clothing | A task with closed-book-solvable questions | Can test retrieval benefit/harm, but alone does not isolate retrieval coverage |
| Normalized Discounted Cumulative Gain at $`k`$ ($`\mathop{\mathrm{nDCG}}\nolimits@k`$) | A ranked relevance score at the depth sent to the generator | It answers a different question from $`\mathop{\mathrm{Recall}}\nolimits@k^{\prime}`$ |
| Ceiling check | Fixed-generator oracle-context diagnostic | Not an absolute bound on all future retrieval or generation improvements |
| Large language model (LLM) judge | A model that assigns evaluation labels | Its agreement with humans must be known on a sample |
| Macro average | An average that gives each class equal weight | It prevents large classes from hiding weak judge agreement |
| Faithfulness | Whether an answer follows the supplied evidence | It can be high even when the evidence itself is wrong |
| Factuality | Whether an answer is true | It must be reported separately from faithfulness |
| Claim-level decomposition | Scoring individual claims instead of a whole answer | It exposes which claims lack support |
| Unverifiable-claim fraction | The share of claims that cannot be checked | It keeps unsupported content visible |
| Accuracy conditioned on answering | Accuracy only among non-abstained answers | It must be paired with abstention rate to show coverage |
| Length dependence | Error variation with answer length | Use matched/stratified analysis rather than declaring every comparison invalid |
| Stage attribution | Diagnosing stage contributions | Permit interacting causes and unresolved evidence |
| Macro, micro, and mezzo views | Three evaluation views at different aggregation levels | All three are needed because one end-to-end score cannot localize failure |
| Remove-the-evidence ablation | Controlled removal of retrieved evidence | Redundancy and distribution shift complicate causal interpretation |
| Closed-book baseline | Performance without retrieved evidence | It must appear beside the RAG result |
| Gold-context ceiling | Performance under specified ideal evidence | A conditional reference, not a universal mathematical ceiling |
| Shuffled-context control | Performance with context order or association disrupted | It is a sanity case for evidence use |
| Self-consistency | Stability across repeated runs | It is a system property beyond average quality |
| Corrupted-document robustness | Behavior when one document is damaged or wrong | It tests whether one bad source can destabilize the system |
| Denominator | The population behind a reported number | Every headline number needs one |
| Source reliability signal | Policy-defined evidence about source reliability | Recency and source agreement are not proofs of truth |
| Cross-source agreement | Support shared across sources | It can contribute to a reliability signal |
| Editorial provenance | The editorial origin and handling of a source | It can contribute to a reliability signal |
| Update recency | How recently a source changed | It can contribute to a reliability signal |
| $`k-1`$ other documents | The remaining retrieved documents when one document conflicts with a set of $`k`$ | The design must name a mechanism for resolving that contradiction |
| Temporal validity | Whether a fact remains valid at retrieval time | An old but once-correct document can produce a cited factual error |
| Credibility laundering | A weak claim gaining apparent authority through quotation by a stronger source | Source reputation alone does not prevent it |
| Post-hoc citation | A source link added after answer generation | Can verify support, but does not by itself establish the answer’s causal origin |
| Attributable claim | A claim supported by the cited evidence | It is one class in the attribution taxonomy |
| Extrapolatory claim | A claim that goes beyond the cited evidence | It is one class in the attribution taxonomy |
| Contradictory claim | A claim opposed by the cited evidence | It is one class in the attribution taxonomy |
| Prompt injection | Retrieved content trying to act as an instruction | Every datastore writer is a potential source |
| Structural separation | Keeping retrieved data distinct from prompt instructions | It helps the model treat content as data rather than commands |
| Blast radius | Observed attack impact under a threat model | A tested maximum is not necessarily the worst possible attack |
| Authenticated ingestion | Verifying who supplied content | It supports provenance policy at write time |
| Content credentials | Provenance information for media | Missing credentials are common and need a policy |
| Red-team test | An adversarial test of the retrieval path | Testing only the model misses retrieval-specific attacks |
| Entitlement | A user’s right to access content | It must be enforced during retrieval |
| Tenant boundary | Separation of customer data and processing | Both filters and separate indexes require correct authorization, routing and failure behavior |
| End-to-end removal | Deleting a document from retrieval, caches, summaries, and graphs | Removal is incomplete if derived copies remain |
| Approximate nearest neighbor (ANN) search | The approximate vector-search stage | It is one of the five measured latency stages |
| Rerank | Reorder retrieved candidates with a later scoring stage | It is one of the five measured latency stages |
| Prefill | Processing input before decoding | Contains linear model work and attention; measure whether it dominates |
| Decode | Producing output tokens | Its runtime share is workload and serving dependent |
| $`p_{95}`$ and $`p_{99}`$ | Latencies below which $`95\%`$ or $`99\%`$ of requests fall | The chosen production percentile matters more than a single-threaded median |
| Dominant stage | Critical-path or resource bottleneck under load | Largest isolated-stage time need not identify an overlapped pipeline bottleneck |
| Key-value (KV) cache | Cached compatible-prefix attention state | Reuse requires matching model, tokens, positions and access policy |
| Coarse-to-fine cascade | A sequence that sends few candidates to the expensive stage | It reduces expensive-stage work |
| Central processing unit and graphics processing unit (CPU and GPU) pipeline | Overlapping compatible independent work | Dependent search follows query encoding; cross-query pipelining can still help |
| Cost per query $`C_q`$ | Total query-serving cost across all five stages | It must include embedding and vector-store cost, not only generation tokens |
| Amortized indexing cost | Build cost allocated over a horizon and traffic | A recurring build need not be an accounting capital expense |
| Priced gate | Measured net cost/quality of a retrieval policy | Skipping can save work, but is not universally cheapest or quality preserving |
| Agentic or iterative loop | A design that retrieves over multiple rounds | It needs expected-round and tail-cost estimates |


## Core mechanics
These checklists are written to be run, not read. Each line must produce a concrete number or a named decision. A line that produces only an adjective remains unanswered.
The retrieval checklist is the interview checklist to internalize. Reproducing its five-decision skeleton and sizing questions verbally in under two minutes is a reasonable preparation target. The other four checklists are job-side design-review instruments.
Each checklist ends with exit criteria. If a design cannot meet them, report that the design is not finished.
### C.1 The Retrieval Design Checklist
Run this checklist in ten minutes. It contains five decisions. Give one number or named choice for every line. A vague answer is a finding, not a pass.
#### Phase 0: Size the problem, 2 minutes
**What**
1. State source documents, chunks, vectors per chunk, dimension and precision; document count alone does not determine memory.
2. State sustained and peak QPS over named windows, query mix, concurrency and resource-service demand.
3. Define TTFT and total-latency endpoints and their required percentiles under load.
4. State source-to-serving freshness and deletion-visibility requirements, including allowable failure behavior. Minutes, hours and days constrain options but do not each dictate one architecture.
5. Define quality metrics, cutoffs, eligible populations and uncertainty; name mandatory constraints separately.
**Why**
This phase turns scale, load, responsiveness, freshness, and quality into testable requirements.
**Failure without it**
The design cannot size memory or compute. It also cannot prove that latency, freshness, or accuracy passes.
**Cost and complexity**
The source gives this phase two minutes. It separates two sizing axes, corpus size and query rate, and two latency measures, time-to-first-token and total.
#### Phase 1: When to retrieve, 2 minutes
**What**
1. Define $`g(x)\in\{0,1\}`$ explicitly, including an always-retrieve policy if chosen.
2. Name available gate signals and when they can be computed; benchmark their cost and paired benefit/loss slices.
3. Define the no-retrieval path, its permitted sources of knowledge, abstention and escalation rules.
4. Monitor gate risk, coverage and query drift. Distribution shift can degrade a gate, but degradation is not inevitable on every shift.
**Why**
The gate controls which queries pay retrieval cost and which queries rely on the closed-book path.
**Failure without it**
Retrieval becomes an unmeasured default. A shifted query mix can silently degrade a learned controller.
**Cost and complexity**
The source gives this phase two minutes. A learned gate also needs an ongoing drift-detection plan.
#### Phase 2: Where and what, 2 minutes
**What**
1. Enumerate the stores. If there is more than one, state the source-selection rule and what happens when two stores disagree. Source pointer: section $`25.3`$.
2. State query transformation $`\Phi`$. Choose identity, compression, expansion, rewriting, or decomposition. State which queries trigger it. Source pointer: chapter 24.
3. For conversational traffic, confirm that rewriting resolves referents outside the current turn. Source pointer: section $`24.5`$.
4. State the chunking scheme and its evidence. Semantic chunking needs a measurement on the target corpus, not a benchmark number. Source pointer: section $`13.6`$.
**Why**
This phase fixes the searched sources, transformed query, conversation behavior, and retrievable unit.
**Failure without it**
Multi-store disagreement has no policy. Rewrites can lose the subject, and chunking becomes an unsupported label.
**Cost and complexity**
The source gives this phase two minutes. It requires corpus-specific chunking evidence and query-specific transformation rules.
#### Phase 3: How to retrieve, 2 minutes
**What**
1. Name scorers and lexical/exact-identifier capabilities; justify dense-only choices on representative slices.
2. For hybrid retrieval, define normalization or rank fusion, constants, weights and missing-list behavior.
3. Separate retrieved candidates $`k_0`$, candidates actually scored and final context items $`k`$.
4. Name the index, exact-neighbor recall target and task-relevance target at a measured latency/load.
5. Define filters and fallbacks. Never relax ACL, tenant or mandatory policy constraints merely to recover recall.
**Why**
This phase makes ranking channels, depth, index behavior, and filtering reproducible.
**Failure without it**
The design can hide exact-match weaknesses, conflate three depths, or over-filter until no useful evidence remains.
**Cost and complexity**
The source gives this phase two minutes. Hybrid fusion adds a method and constant. Recall must be tied to a measured latency.
#### Phase 4: How to present it, 2 minutes
**What**
1. Count actual tokenized query, instructions, evidence, headers, separators and demonstrations, plus reserved output. $`kL`$ estimates evidence only for $`k`$ equal-length $`L`$-token chunks.
2. Evaluate evidence order and complete support-chain coverage on the deployed model.
3. Define serialization and instruction/data boundaries; formatting is not an injection-proof security boundary.
4. Record source IDs, versions and spans, whether citations are inline or post-hoc, and independently measured support and coverage.
**Why**
This phase connects retrieved candidates to prompt cost, evidence position, instruction boundaries, and citations.
**Failure without it**
Context cost stays hidden. Evidence can land in a weak position, instructions can blur with data, and citation timing remains ambiguous.
**Cost and complexity**
The two-minute phase is a practice allocation. $`kL`$ is an equal-chunk evidence estimate, not total prefill tokens; count the fully serialized prompt with the actual tokenizer.
#### C.1 exit criteria
**What**
1. Every one of the five decisions has an answer, and at least two carry a number.
2. Name one thing the design gives up without being prompted.
3. State how to find which stage broke when quality drops. Source pointer: chapter 34.
**Why**
These gates test completeness, trade-off awareness, and debuggability.
**Failure without it**
The retrieval design remains vague, one-sided, or impossible to diagnose.
**Cost and complexity**
The exit gate requires five answered decisions, at least two numeric decisions, one volunteered sacrifice, and one diagnostic plan.
### C.2 The Indexing and Capacity Checklist
Run this checklist before provisioning and before every migration. Every line should produce a number. The source points to section A.8 for the arithmetic.
#### Sizing
**What**
1. Count $`N`$ stored vectors, dimension $`d`$, precision or packed code bytes, and per-item IDs.
2. Add only the selected index’s links, posting IDs, coarse centroids, PQ codebooks, maps and allocation/runtime overhead; do not attach every index family’s costs to every design.
3. State measured exact-neighbor and task-relevance effects of compression at matched search controls.
4. Give replication and rebuild overlap separately for indexes, payloads, metadata and caches. Twenty-percent reserve does not hold a complete second equal-size index.
5. Count source text and filterable metadata, compression and their own replication policy. There is no universal 30-fold payload crossover.
6. Derive per-partition fit and load. Ceiling division of aggregate bytes by usable RAM is a capacity lower bound, not a logical-shard or availability plan.[\[1\]](../part-12-systems-design/41_end_to_end_design_drills.md)
**Why**
Sizing must include vectors, index overhead, compressed codes, rebuild copies, replicas, payloads, and usable node memory.
**Failure without it**
Fleet memory can be understated. Rebuilds can exceed capacity, and aggressive vector compression can hide a dominant payload store.
**Cost and complexity**
Budget actual simultaneous live and build artifacts. An old/new full-index overlap needs another full copy wherever both coexist; an external build also consumes resources. Replication is artifact specific, and payload dominance depends on representation and placement rather than a universal compression threshold.
#### Recall and latency
**What**
1. Define ANN Recall@$`k`$ against exact search on the same eligible corpus and metric. Report semantic/evidence recall separately with judgments.
2. Record search-time controls such as $`\mathrm{efSearch}`$ or $`n_{\mathrm{probe}}`$ and the measured quality/latency curve.
3. Record build/training settings and which changes require retraining, re-encoding, rebuilding or only configuration updates in this implementation.
4. Test realistic traffic, concurrency, warm/cold states, filters and failure capacity.
5. Prefer low-cost reversible experiments when feasible, but let measured failure cause and constraints determine the order of remedies.[\[2\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)
**Why**
This order tries reversible controls before expensive rebuilds or an index-family change.
**Failure without it**
Recall can be measured against the wrong target. Latency can look good only because the test ignored concurrency.
**Cost and complexity**
Search-time changes are often reversible; build-setting changes can require reconstruction. Neither a full rebuild for every change nor one mandatory optimization order is universal. Record the implementation’s actual migration and rollback path.
#### Churn and operations
**What**
1. State insert/update/delete rates and the implementation’s physical deletion or tombstone semantics.
2. Measure effects of deletions on recall, service demand, tail latency and memory; do not assume monotone recall loss or one universal fraction threshold.
3. Specify rebuild placement, full peak resources, transfer/swap consistency and what queries see throughout.
4. Test stale caches, replicas and failures while enforcing deletion and authorization state.
5. Pin document and query encoders, tokenizer, pooling, normalization and index versions. Document-space changes usually need re-embedding; query-only or verified compatible migrations may not. Versioned partial migration requires an explicit compatibility/routing plan.[\[3\]](../part-05-indexing/17_operating_a_vector_store.md)
**Why**
The index is an operated system. Churn, deletion markers, rebuild behavior, and model versions change its live state.
**Failure without it**
Recall can decay without a trigger. Queries can hit an untested rebuild path, and a model change can invalidate the whole index unexpectedly.
**Cost and complexity**
Rebuild triggers use measured quality, latency, deletion visibility and capacity, not tombstone fraction alone. Some document-encoder changes require full re-encoding; query-only, compatible or dual-index migrations can permit other transitions.
#### C.2 exit criteria
**What**
1. Derive capacity by artifact and placement, including payload replicas, runtime and rebuild peaks.
2. Report graph/link and identifier shares under this layout; they do not shrink merely because vector codes shrink, but are not inherently incompressible.
3. Provide a timed rebuild/migration, deletion-verification and rollback runbook under target load.
**Why**
These gates demonstrate a scoped capacity plan, measured structural overhead and tested operations; they do not constitute a proof against every production failure.
**Failure without it**
Provisioning depends on an opaque estimate, and the rebuild path remains unmeasured.
**Cost and complexity**
Passing requires a derived capacity plan, structural-byte breakdown and timed operational runbooks. Explicitly label unknowns and unmeasured worst cases.
### C.3 The Evaluation Checklist
An evaluation that cannot localize a failure is a dashboard, not an instrument. This checklist builds an instrument. The source develops the method in chapter 34.
#### Before measuring anything
**What**
1. Work three representative real queries by hand, including ambiguous and insufficient-evidence cases, before choosing metrics.
2. Describe sampling, query count, judges, relevance/gain scale, adjudication and agreement; keep development and test data separate.
3. Document training-contamination checks and their access limits. A closed model’s unknown corpus cannot be certified absent from a finite prompt probe.
4. Measure closed-book solvability, evidence dependence and required support. Closed-book-solvable questions can still test retrieval benefit or harm; they do not by themselves isolate retrieval quality.[\[4\]](../part-10-evaluation/34_evaluating_the_system.md)
**Why**
This phase fixes the intended behavior, evaluation population, label process, and contamination checks before metric selection.
**Failure without it**
A metric can reward the wrong behavior. Contamination or closed-book solvability can make retrieval look useful when it was not tested.
**Cost and complexity**
The source requires three hand-worked real queries and a documented judgment set. It gives no fixed dataset size.
#### Retrieval layer
**What**
1. Report evidence Recall@$`k_0`$ and ranked relevance at final depth, with cutoffs, eligible corpus, gain, IDCG and unjudged/zero-IDCG policies.
2. Measure a fixed-generator oracle/gold-context reference on matched items. It estimates evidence headroom under that setup, not an absolute bound on all downstream or future improvements.
3. Validate model judges against human labels using classwise confusion, macro metrics, sample uncertainty and relevant slices; report the label process.[\[5\]](../part-10-evaluation/32_evaluating_retrieval.md)
**Why**
Separate candidate coverage, ordering, exact-neighbor approximation, evidence-use references and judge validity. No one metric substitutes for the others.
**Failure without it**
One retrieval metric can hide the other question. An unvalidated model judge can make the dashboard unreliable.
**Cost and complexity**
Budget exact-neighbor evaluation separately from human relevance judgments. Oracle-context and judge-validation runs need matched samples, explicit denominators and uncertainty.
#### Generation layer
**What**
1. Report context faithfulness and independent factuality separately; faithful use of a wrong source can still be false.
2. Define atomic claims and label support, contradiction, unverifiable and omitted claims without silently dropping unknowns.
3. Pair acceptance/abstention coverage with accepted-answer correctness and uncertainty, including zero-accepted cases.
4. Audit answer-length and claim-count effects. Use matched or stratified comparisons where needed; length dependence is a confound to address, not automatic invalidity of every comparison.
**Why**
These measures separate evidence use, truth, unsupported claims, coverage, and answer-length effects.
**Failure without it**
Wrong evidence can look faithful; refusals can inflate accepted-answer precision; answer-length and population shifts can confound headline comparisons unless controlled.
**Cost and complexity**
Claim-level scoring adds decomposition work. Conditional accuracy must be paired with abstention rate, and length must be checked across systems.
#### System layer
**What**
1. Classify a sample of failures across stages, allowing interacting and unresolved causes rather than forcing exactly one owner.
2. Define macro, micro and intermediate/slice aggregation and denominators.
3. Compare evidence removal, replacement and ordering under controlled decoding. Redundant evidence or distribution shift can complicate causal interpretation.
4. Run matched closed-book, oracle-context and shuffled-context controls with confidence intervals.
5. Measure production latency distributions, repeat-run stability, authorization failures and corrupted/poisoned-document robustness.[\[4\]](../part-10-evaluation/34_evaluating_the_system.md)
**Why**
System-level controls connect quality to plausible mechanisms and operational constraints. An ablation or descriptive decomposition alone does not prove a unique cause.
**Failure without it**
A quality drop stays unlocalized. The system may ignore retrieval, and one average can hide unstable or fragile behavior.
**Cost and complexity**
The checklist adds three sanity cases, one ablation, repeated runs, percentile latency, and a corrupted-document test.
#### C.3 exit criteria
**What**
1. Set a triage SLO, such as one working day, for a two-point observed decline. First check uncertainty and label/traffic changes; permit an honestly unresolved cause.
2. Give every headline number its denominator, sample, version, uncertainty and missing-data policy.
3. Show matched closed-book and RAG results with evidence/quality/cost slices.
**Why**
These gates test localization speed, measurement transparency, and retrieval value.
**Failure without it**
The dashboard cannot assign a quality drop, explain its population, or show improvement over no retrieval.
**Cost and complexity**
One working day is an operational triage target for the illustrative two-point change, not a statistical detection guarantee or promise that every cause is identifiable.
### C.4 The Credibility and Robustness Checklist
Retrieval places text the team did not write into a context the model treats as authoritative. This checklist covers unreliable sources, hostile sources, and sources that disagree. The source develops it across chapters 35 and 36.
#### Source quality
**What**
1. Record store, document/version and evidence span for each retrieved unit.
2. Define source reliability and authority policies, validation data and update ownership. Agreement and recency are signals, not proofs of truth; correlated sources can repeat one false claim.
3. Specify contradiction handling and whether to show competing evidence or abstain.
4. Preserve valid-time and ingestion/update-time metadata, plus the relevant jurisdiction or product version.
5. Trace quotations to original evidence where possible to detect credibility laundering.
**Why**
Source identity, evidence-based reliability, conflict handling, time validity, and laundering checks control what the model treats as authoritative.
**Failure without it**
Controls cannot be enforced by source. One contradiction, stale fact, or laundered claim can become a confident cited answer.
**Cost and complexity**
The checklist requires a reliability signal per source and a concrete contradiction mechanism. It gives no fixed numeric threshold.
#### Attribution
**What**
1. Define whether evidence preceded generation and whether citations were inline or post-hoc. Neither timing proves actual use, support or truth.
2. Resolve citation IDs/versions/spans and score attributable, extrapolatory, contradictory and unverifiable claims with explicit coverage denominators.
3. Validate automatic support judgments against human labels, with uncertainty and adversarial/ambiguous cases. Product grounding claims must match these measured properties.[\[6\]](../part-09-context-assembly/31_attribution_and_citation.md)
**Why**
Evidence availability, citation validity, support coverage and causal influence are different claims. Citation timing alone cannot establish or exclude all of them.
**Failure without it**
Product language can overstate post-hoc citations. An automatic score can look precise without known human agreement.
**Cost and complexity**
The source requires a human-scored sample for attribution and a human-agreement check for any automatic judge. It gives no fixed sample size.
#### Adversarial surface
**What**
1. Enumerate human and automated writers and upstream feeds as potential injection paths.
2. Separate instruction and data channels; add least-privilege tools, source authorization, sanitization where appropriate and verified output actions. A system-prompt sentence is not a security guarantee.
3. Measure harmful outcomes under one-chunk and coordinated attacks; state the tested threat model and unknown worst cases.
4. Authenticate ingestion and evaluate asset-bound credentials and signer trust; provenance credentials do not prove content truth, and absence does not prove fabrication.
5. Red-team retrieval, caches, tools and downstream actions, not only the base model.
**Why**
These checks cover write authority, instruction boundaries, single-document impact, provenance, and retrieval-specific attacks.
**Failure without it**
Untrusted content can act like an instruction. One poisoned chunk can have an unknown impact, and model-only tests can miss the retrieval path.
**Cost and complexity**
The source requires a measured one-chunk blast radius, authenticated ingestion policy, and a retrieval-specific red-team exercise.
#### Privacy and exposure
**What**
1. Enforce user and tenant entitlement before unauthorized evidence reaches the generator, including cache and reranking paths.
2. State the boundary mechanism, route validation, fail-closed behavior and tests. Neither a filter nor a separate index is inherently fail-open or fail-closed under every bug.
3. Specify deletion scope, measured visibility, replicas, caches, derived summaries/graphs, logs and backup handling; verify each required copy under policy.[\[7\]](../part-12-systems-design/38_distributed_and_federated_rag.md)
**Why**
The retrieval layer must stop unauthorized evidence before it reaches the generator and must remove every derived copy.
**Failure without it**
Private content can enter context. A filter bug can expose another tenant, and deletion can leave cached or derived material behind.
**Cost and complexity**
Separate stores may reduce cross-tenant risk but require correct routing and authorization. Either design needs tested failure behavior and a copy-aware removal policy.
#### C.4 exit criteria
**What**
1. State measured attacker outcomes and the threat model; do not call an untested worst case bounded.
2. Provide document removal with separate visibility, derived-copy, replica and retention deadlines plus verification evidence.
3. Keep grounding and safety claims within what evaluation and enforced controls demonstrate.
**Why**
These gates bound single-document harm, make deletion operational, and constrain product language.
**Failure without it**
The design cannot state its blast radius, complete removal, or defend its grounding claim.
**Cost and complexity**
Passing requires measured attack outcomes under a stated scope and verified removal operations; it does not establish universal worst-case security or instantaneous erasure of every copy.
### C.5 The Latency and Cost Checklist
Run this checklist with a stopwatch on a real query rather than on a diagram. The source points to section $`37.1`$ and section A.7 for stage arithmetic.
#### Build the budget
**What**
1. Trace embedding, search, reranking, prefill and decode, plus gates, network, queues, retries and merge/orchestration work.
2. Report TTFT and end-to-end distributions at the relevant load and percentiles, including warm/cold behavior and errors. A median is useful but not a tail SLO.
3. Identify the actual critical-path and resource bottlenecks; do not assume generation or search always dominates.
4. Keep per-request stage traces: the sum of independently measured stage percentiles is not generally the end-to-end percentile.
**Why**
The budget exposes each stage, the production tail, the dominant constraint, and the product-facing latency measure.
**Failure without it**
An estimate can miss by an order of magnitude. A median can hide the production tail, and an optimization can target the wrong stage.
**Cost and complexity**
The source requires five timed stages at production concurrency. It calls for $`p_{95}`$ or $`p_{99}`$ when that percentile matters.
#### Reduce it in the right order
**What**
1. Compare context reduction with other remedies at matched evidence support and quality. Prefill includes linear model work and quadratic attention terms; total latency is not universally quadratic.
2. Reuse KV state only when token prefixes, model, positions, attention behavior and authorization are compatible. A repeated chunk in a different prefix is not automatically a reusable causal cache.
3. Test candidate/context cascades with their coverage and scheduling costs.
4. Overlap independent work; query embedding precedes dependent search for that query, although different queries or stages can pipeline.
5. Compare smaller models, different precision, batching and index changes on a measured cost/quality/SLO frontier rather than one mandatory order.[\[8\]](../part-12-systems-design/37_latency_cost_and_systems.md)
**Why**
This order reduces avoidable work before replacing quality-bearing components.
**Failure without it**
The design can sacrifice quality while leaving context, repeated work, candidate count, and overlap opportunities untouched.
**Cost and complexity**
Context reduction, compatible-prefix caching, cascades and pipelines change resource and quality trade-offs. Measure those changes; a smaller model or cheaper index need not worsen every task, and no ordering is universally optimal.
#### Cost per query
**What**
1. Price attributable resource service and provider charges for all paths, plus fixed-capacity utilization. With rates $`r_j`$ per resource-hour and service $`s_j`$ seconds, an allocated variable proxy is $`\sum_j r_js_j/3600`$, not summed wall latency times one rate.
2. Add or separately report amortized ingestion, embedding, build, refresh, storage, licenses and replication under a stated time horizon and request count.
3. Evaluate retrieval-gate savings net of gate cost, missed retrieval benefits and drift; skipping is not always the cheapest or safest optimization.
4. For iterative policies report distributions of calls, tokens, cost, latency and failures, not only expected rounds.[\[8\]](../part-12-systems-design/37_latency_cost_and_systems.md)
**Why**
The calculation exposes full serving cost, recurring build cost, gate value, and multi-round tail cost.
**Failure without it**
Generation tokens can hide embedding, vector-store, indexing, and repeated-retrieval costs.
**Cost and complexity**
Report per-request variable cost and allocated fixed/build/storage cost separately before a combined total. Specify utilization, currency, time horizon and traffic; latency and resource-service cost are not interchangeable.
#### C.5 exit criteria
**What**
1. Name the dominant stage and its budget share as a percentage.
2. Derive one cost-per-query number with the stages broken out.
3. For every proposed optimization, name the constraint it addresses and what it trades away.
**Why**
These gates tie optimization to measured dominance, complete cost, and explicit trade-offs.
**Failure without it**
The design can optimize a minor stage, omit costs, or hide a quality sacrifice.
**Cost and complexity**
Passing requires one percentage, one staged cost-per-query total, and one constraint-and-trade-off statement per optimization.
## Diagrams
The source lists zero captioned figures and source tables. This study page includes decoder and key-number tables summarizing the five checklists, not measured experiment results.
## Whiteboard pack
### What to draw
1. Draw one large box labeled “Design review.”
2. Draw five cards inside it from left to right.
3. Label the cards “Retrieval,” “Indexing,” “Evaluation,” “Credibility,” and “Latency and cost.”
4. Under Retrieval, write “size, gate, sources, search, presentation.”
5. Under Indexing, write “memory, recall, churn, rebuild.”
6. Under Evaluation, write “test set, retrieval, generation, system.”
7. Under Credibility, write “sources, attribution, attacks, privacy.”
8. Under Latency and cost, write “five stages, optimize, price.”
9. Draw an arrow from every card to one bottom box labeled “Exit criteria.”
10. Inside the bottom box, write “number or named choice.”
### Spoken script
Run five cards rather than merely naming them. Retrieval fixes requirements, routing, sources, scoring and presentation. Capacity separates artifacts, copies, load and operational peaks. Evaluation checks populations, evidence use and stage hypotheses without inventing guarantees. Credibility verifies permissions, source support and scoped attacks. Latency/cost traces actual dependencies and prices attributable resources. Every unknown has an owner and a validation plan; an unchecked box is not evidence of failure or success.
## Interview traps
### Probe 1: Why is “always retrieve” not a complete retrieval design?
Always-retrieve is an explicit gate with measured cost, not a neutral default. A complete answer also names the closed-book path, abstention behavior, controller-drift detection, sources, transformations, search depths, index target, filters, prompt placement, and attribution timing.
### Probe 2: Why can vector compression fail to solve capacity planning?
Compression shrinks vector codes, not automatically identifiers, links, payloads, runtime or copy counts. Their shares depend on the exact layout; graph links are not inherently incompressible and no universal 30-fold crossover exists. Ceiling bytes provide a capacity lower bound, not a shard/replica/load plan.
### Probe 3: Why is one end-to-end quality number insufficient?
It cannot localize retrieval, generation, or system failure. The evaluation checklist pairs retrieval metrics, claim-level generation measures, abstention, stage attribution, macro, micro, and mezzo views, ablations, sanity controls, latency, consistency, and corrupted-document robustness.
### Probe 4: When is a cited answer still not safely grounded?
A citation can be invalid, unsupported, stale or tied to an unreliable source. Inline generation does not prove evidence use; post-hoc linking can verify support but not by itself establish the answer’s origin. Define and evaluate the precise grounding claim, enforce authorization and track source versions and removal.
### Probe 5: Why not optimize search latency first?
Optimize the measured critical path and constrained resource under production load, not a presumption that search or generation must dominate. Test context, caching, cascade, overlap and component alternatives at comparable quality and risk; independently measured stage percentiles cannot simply be added.
## Key numbers


| Number, threshold, or trade-off | Source commitment | Design meaning |
| --- | --- | --- |
| 5 checklists | Retrieval, indexing and capacity, evaluation, credibility and robustness, latency and cost | The appendix covers five review surfaces |
| 1 interview checklist | Internalize the retrieval checklist | The other four primarily serve on-the-job design reviews |
| Under 2 minutes | Reasonable target for verbally reproducing the retrieval skeleton and sizing questions | Interview recall should be concise |
| 10 minutes | Total retrieval design checklist time | The checklist is a timed design round |
| 5 decisions | Size, retrieval timing, sources and transformations, search, presentation | Every decision needs an answer |
| 2 minutes per phase | Time for each of the five retrieval phases | Five equal phases fill the ten-minute checklist |
| At least 2 numeric decisions | Minimum numeric content in the C.1 exit criteria | Naming choices alone is insufficient |
| Corpus size versus query rate | Memory and compute are different sizing axes | Both must be stated |
| Freshness scales | Source-to-serving lag and failure policy | Do not infer one mandatory architecture per timescale |
| 2 latency measures | Time-to-first-token and total latency | The product must name which measure matters |
| Always retrieve versus a selective gate | Always retrieving has measured cost and is not a neutral default | State $`g(x)`$ explicitly |
| Closed-book answer versus abstention | The no-retrieval path must name its behavior | Coverage is a design decision |
| Multiple stores versus one store | Multiple stores need a selection rule and disagreement behavior | Source routing cannot remain implicit |
| Semantic label versus corpus measurement | Semantic chunking needs evidence on the target corpus | A benchmark number does not answer the checklist line |
| Dense-only versus a lexical channel | Dense-only retrieval needs justification for long-tail and exact-match queries | Channel choice has query-specific risk |
| Hybrid retrieval | The design must state the fusion method and its constant | Combining channels adds a reproducibility requirement |
| $`k^{\prime}`$, rerank depth, $`k`$ | Retrieval candidates, reranked candidates, and generator chunks | The three depths must be stated separately |
| Recall target at named latency | Index recall and latency must be committed together | Neither metric stands alone |
| Filters and fallback | Preserve mandatory authorization | Relax only permitted preference constraints |
| Equal-size evidence $`kL`$ | Evidence estimate, not total prefill | Count actual serialized input including prefixes and instructions |
| Inline versus post-hoc links | State timing and independent support checks | Neither timing establishes internal causal influence or truth |
| Old/new overlap | Budget full artifacts wherever simultaneously resident | External builds and swap transfer also use capacity |
| Compression versus measured recall | State resulting code size and measured recall cost | Do not assume the quality loss |
| Payload crossover | Workload- and placement-specific | No universal thirtyfold compression threshold |
| Ceiling division | Aggregate-byte capacity lower bound | Not a logical-shard, throughput or availability plan |
| Exact-neighbor recall versus relevance judgments | The source says these measure different things | Use exact search for the recall target |
| Search/build/family remedies | Prefer inexpensive reversible experiments when appropriate | Diagnose the measured failure rather than obeying a universal order |
| In-place rebuild versus separate hot-swap | State where rebuilding happens | Query behavior during rebuild must also be tested |
| Versioned representation migration | Document and query changes have different effects | Partial/dual-index or compatible migration can exist with explicit routing |
| 3 real queries | Hand-written correct-answer examples before metric selection | Evaluation starts from concrete intended behavior |
| $`\mathop{\mathrm{Recall}}\nolimits@k^{\prime}`$ and $`\mathop{\mathrm{nDCG}}\nolimits@k`$ | Two retrieval-layer measures that must both be reported | They answer different questions |
| 3 attribution classes | Attributable, extrapolatory, contradictory | Attribution quality is measured at claim level |
| Faithfulness versus factuality | Report them separately | A faithful answer can follow a wrong document |
| Abstention versus production coverage | Report abstention rate and accuracy conditioned on answering | More refusal can improve precision while worsening production behavior |
| Answer-length dependence | Potential comparison confound | Control or stratify lengths rather than automatically invalidate all results |
| 3 evaluation views | Macro, micro, mezzo | One end-to-end number cannot localize failure |
| Closed-book/oracle/shuffled controls | Conditional diagnostic references | Oracle context is not an absolute bound on all future improvement |
| Two-point observed drop | Check uncertainty before assigning a cause | Stage interactions and unresolved causes are legitimate findings |
| One working day | Illustrative triage SLO | Not a statistical identifiability guarantee |
| 2021 | Example year for a once-correct but now stale document | A citation does not repair temporal invalidity |
| $`k-1`$ | Other documents contradicted by one retrieved document | The design must name a conflict mechanism |
| 1 poisoned chunk | Unit used for the blast-radius check | Security impact must be measured at single-document scale |
| One attacker document | Measure under an explicit threat model | The tested maximum is not necessarily the worst possible effect |
| Filter versus separate tenant index | Test routing, authorization and fail-closed behavior | Neither design inherently prevents every cross-tenant failure |
| 5 latency stages | Query embedding, ANN search, rerank, prefill, decode | Measure every stage on a real query |
| $`p_{95}`$ or $`p_{99}`$ | Usually relevant latency percentiles | Report the product-relevant tail at production concurrency |
| Measured versus proxy latency | Report assumptions and benchmark deviations | No universal tenfold error in intuition is assumed |
| Prefill work | Linear model and attention terms | Total prefill/runtime is not universally quadratic |
| Optimization alternatives | Measured cost/quality/risk frontier | No mandatory context-first order for every system |
| Component alternatives | Benchmark smaller models and cheaper indexes | Neither has a universal fixed quality penalty |
| Serving versus ingestion/build cost | State horizon, utilization and allocation | Avoid treating wall latency as resource cost or recurring builds as inherently capex |
| Expected rounds versus tail rounds | Price both for agentic or iterative retrieval | Mean cost alone misses the tail |
| 1 cost per query $`C_q`$ | Derived total across all five stages | Break the total into its stage costs |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A2\_question\_bank](A2_question_bank.md) · [A4\_the\_rag\_card\_and\_index\_datasheet](A4_the_rag_card_and_index_datasheet.md)
