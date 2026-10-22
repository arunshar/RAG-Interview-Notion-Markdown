# Appendix D: The RAG Card and Index Datasheet
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A3\_design\_checklists](A3_design_checklists.md) · [A5\_annotated\_reading\_list](A5_annotated_reading_list.md)
This appendix supplies two handoff templates for a Retrieval-Augmented Generation (RAG) system, with the RAG Card documenting scope, retrieval decisions, measured behavior, and known failures while the Index Datasheet documents artifact composition, representation, structure, footprint, performance, and lifecycle.
## TL;DR
- Maintain separate, linked system and artifact records with versioned measurements, owners and last-verified dates.
- Update the RAG Card after policy, routing, generator, evaluation or source changes. Update the Index Datasheet after any serving, lifecycle or representation change, not only rebuilds.
- Fill each field or record why it is unknown, inapplicable or not measured, with an owner and plan.
- Separate relevance, exact-neighbor approximation, claim support, truth, selective correctness and latency/cost populations.
- The filled records are illustrative source scenarios, not production benchmarks independently reproduced here. Arithmetic below explicitly assumes fp32 reference vectors, 96-byte codes and eight-byte posting IDs.
- Cross-record inconsistencies are findings: gate/refusal semantics, omitted stores, date/count mismatches and insufficient rebuild headroom must be resolved.
## The story
Think of a retrieval system handoff as transferring a ship to a new crew.
The RAG Card is the bridge log. It states the vessel’s mission, forbidden waters, captain, operating decisions, measured behavior, and known hazards.
The Index Datasheet is the engine-room ledger. It records what physical cargo is aboard, how it is packed, how much space it consumes, how fast the machinery runs, and what happens during maintenance.
The records have different scopes, not exclusive clocks. Routing or answer-policy changes primarily affect the card; search controls, deletion policy, encoder compatibility and live corpus changes can invalidate the datasheet without a rebuild. Cross-link every affected release and measurement.
They also have different keepers. The team that owns the user-facing system and on-call rotation may not own the underlying store, embedding, or rebuild job.
An unknown field is a validation gap, not automatically proof that a system is unsafe or broken. Missing baselines limit claims of added value, while a missing tombstone threshold calls for implementation-specific quality, latency and deletion tests rather than assuming one universal deletion failure mode.
The most important warning can span books or rows. An authoritative store plus unreviewed automated ingest creates an injection surface. A small compressed index plus a large payload means text, not vectors, drives capacity.
The handoff succeeds when a new crew can answer what the system does, what it costs, who owns each moving part, what evidence supports it, and which conditions make it fail.
## Decoder table


| Term or field | Decode | Required evidence or decision |
| --- | --- | --- |
| Retrieval-Augmented Generation (RAG) | Generator augmented with retrieved evidence | The card documents the full system rather than only the generator |
| RAG Card | One-page system record | Scope, five decisions, measurements, and failures |
| Index Datasheet | Physical index record | Contents, representation, cost, performance, and lifecycle |
| Model card | Preceding model documentation practice | Records training context and inappropriate uses |
| Datasheet for datasets | Preceding dataset documentation practice | Records provenance, composition, and intended use |
| Datastore | Store searched by retrieval | Its contents and writers affect system behavior |
| Generator | Model that answers from retrieved context | Receives the final presented evidence |
| Last accurate date | Date the card still described reality | Makes staleness visible |
| System name and version | Identity of one deployed system state | Binds measurements to a concrete release |
| Intended use | Question types the system is built to answer | Positive operating boundary |
| Out of scope | Question types it answers badly | Specific negative boundary |
| Owner | Responsible team and on-call rotation | Operational accountability |
| When | Retrieval gate decision | Gate family, fire rate, and unnecessary-retrieval cost |
| Where | Datastore decision | Every store, size, owner, and conflict rule |
| What | Query-transformation decision | Transformation and triggering query subset |
| How | Retrieval method decision | Scorer, index, $`k^{\prime}`$, rerank depth, lexical channel |
| Present | Generator-context decision | Final $`k`$, order, serialization, and prompt position |
| Retrieval gate | Rule that decides whether retrieval runs | Requires family, fire rate, and cost when unnecessary |
| Always-on retrieval | Retrieval on every query | Must report cost on queries that did not need it |
| Query transformation | Rewrite applied before retrieval | Must name both the rewrite and triggering queries |
| Identity transformation | Leaving the query unchanged | Worked fallback when no conversational rewrite runs |
| Rerank depth | Candidate count scored by the reranker | Sits between initial retrieval and final context depth |
| Lexical channel | Exact-token retrieval path | Worked card uses BM25 beside dense retrieval |
| Dense channel | Embedding-based retrieval path | Worked card fuses it with BM25 |
| Cross-encoder reranker | Pairwise query-document scorer | Reduces the worked candidate set before presentation |
| Context ordering | Placement order of final evidence | Worked card places strongest evidence first and last |
| Serialization | Format used for retrieved blocks | Worked card uses JSON |
| Prompt position | Location of retrieved text relative to instructions | Makes instruction and evidence layout explicit |
| $`k^{\prime}`$ | Initial retrieval depth | Candidate depth before reranking |
| $`k`$ | Context depth after reranking | Evidence count passed to the generator |
| Best Matching 25 (BM25) | Lexical scorer in the worked card | Exact-token retrieval channel |
| Reciprocal Rank Fusion (RRF) | Rank-based list fusion | Combines lexical and dense channels |
| JavaScript Object Notation (JSON) | Worked serialization format | Places retrieved blocks after instructions |
| $`\operatorname{Recall}@k^{\prime}`$ | Relevant evidence found by retrieval depth | Requires a described judgment set |
| $`\operatorname{nDCG}@k`$ | Normalized Discounted Cumulative Gain at final depth | Measures ranked quality at generator depth |
| Judgment-set size | Count of evaluated queries or items | Defines evidence volume |
| Judge | Human or other judging source | Defines who assigned relevance |
| Relevance scale | Labels used for relevance | Makes metric interpretation possible |
| Agreement | Inter-judge consistency | Shows reliability of relevance labels |
| Kappa ($`\kappa`$) | Agreement statistic shown in the worked card | Reports the worked three-way human-judgment consistency |
| Faithfulness | Whether answer claims follow provided evidence | Report separately at claim level |
| Factuality | Whether answer claims are factually correct | Report separately from faithfulness |
| Claim level | Evaluating individual answer claims | Prevents one answer-level score from hiding unsupported claims |
| Closed-book baseline | Accuracy with retrieval disabled | Measures the generator-only path |
| Gold-context ceiling | Fixed-generator performance with specified ideal evidence | Conditional diagnostic, not an absolute bound on future improvements |
| Abstention | Refusal behavior | Refusal rate plus accuracy when answering |
| Refusal rate | Fraction of requests not answered | Must be paired with accuracy on answered requests |
| Accuracy conditioned on answering | Accuracy after excluding refusals | Separates safer refusal behavior from answer quality |
| $`p_{50}`$ | Fiftieth percentile | Reports median index latency |
| $`p_{95}`$ | Ninety-fifth percentile | Tail metric for total and first-token latency |
| $`p_{99}`$ | Ninety-ninth percentile | Tail metric for freshness lag |
| TTFT | Time to first token | Separates prompt and prefill delay from total time |
| Stage cost | Cost per query by pipeline stage | Prevents one blended cost number |
| Freshness lag | Source change to retrievability | Report median and $`p_{99}`$ where requested |
| Attribution quality | How citations attach and whether claims are attributable | Method plus measured attributable rate |
| Inline citation | Citation generated with the answer | Distinguished from a citation attached afterwards |
| Attributable rate | Supported answers or claims under an explicit rubric | Do not mix answer-level and claim-level denominators |
| Adversarial surface | Write access and poisoning impact | Writers plus one-chunk blast radius |
| Write access | People or processes allowed to alter a store | Exposes reviewed and automated ingestion paths |
| Automated ingest | Programmatic datastore write path | Becomes an injection surface when unreviewed |
| Authoritative store | Store the generator is expected to trust | Makes an unreviewed writer especially risky |
| Injection surface | Path by which hostile content can enter retrieval | Emerges from write access and generator trust together |
| Poisoned chunk | One adversarial retrieval unit | Supplies the blast-radius measurement unit |
| Corpus snapshot | Source systems, counts, and date | Fixes index composition in time |
| Document count | Number of source documents | Distinguishes source items from derived chunks |
| Chunk count | Number of retrieval units | Drives index and payload scale |
| Snapshot date | Date represented by the corpus | Binds composition to a point in time |
| Chunking | Scheme, target, overlap, and selection evidence | Shows whether the policy was chosen or inherited |
| Target chunk size | Intended tokens per chunk | Defines the worked chunking granularity |
| Chunk overlap | Tokens repeated across neighboring chunks | Must be recorded with target size |
| Chunk prefix | Heading retained with chunk text | Worked example preserves document structure |
| Dropped content | Excluded modalities, boilerplate, and oversize documents | Makes silent exclusions visible |
| Silent exclusion | Content removed without an explicit record | Can mimic retrieval failure even when the source held the answer |
| Metadata | Filterable fields | Cardinality and fill rate for each field |
| Filterable field | Metadata usable in retrieval filtering | Requires cardinality and fill rate |
| Cardinality | Count of distinct field values | Describes a filter’s selectivity |
| Embedding model | Pinned query/document representation | Compatibility and version routing determine migration options |
| $`d`$ | Embedding dimensionality | Sets the worked vector width |
| $`L_2`$ normalization | Unit-length vector normalization | Must match the selected distance metric |
| Distance metric | Rule used to compare vectors | Must agree with normalization |
| Cosine | Worked vector similarity measure | Used with $`L_2`$-normalized embeddings |
| Partial migration | A versioned representation transition | Uncontrolled incompatible mixing is invalid; compatible or dual-index migrations can exist |
| Precision | Stored numeric precision and quantization | Requires measured recall cost |
| Quantization | Compressed numeric representation | Must include its measured recall cost |
| HNSW | Hierarchical Navigable Small World index | Graph-family option with link memory |
| IVF | Inverted File index | Coarse-partitioned index family |
| IVF-PQ | Inverted File with Product Quantization | Compressed index used in the example |
| Disk-resident index | Index stored beyond memory | Must include build parameters |
| Build parameters | Settings used to construct the chosen index | Needed to reproduce the artifact |
| Per-vector bytes | Vector/code, ID and applicable link bytes | Plus family-level, allocation and runtime terms; structural share is not inherently incompressible |
| PQ code bytes | Compressed vector representation | Worked example assigns $`96\ \text{bytes}`$ per vector |
| Vector compression | Raw vector bytes divided by compressed code bytes | Must not be confused with full per-vector or fleet compression |
| Identifier bytes | Posting identifier storage | The revised layout explicitly uses eight bytes per ID |
| Link bytes | Graph-neighbor storage | Zero for the worked IVF-PQ index |
| Link share | Scoped graph-neighbor share | Does not shrink automatically with vector coding; topology/layout can change it |
| $`n_{\mathrm{list}}`$ | IVF coarse-centroid count | Sets the worked partition count |
| $`m`$ in IVF-PQ | Product Quantization subvector count | Sets the worked code length |
| $`k`$ in IVF-PQ | Centroids per Product Quantization subspace | Distinct from retrieval depth $`k`$ |
| Centroids | Coarse or quantization representatives | Contribute memory outside per-vector codes |
| Codebook | Stored Product Quantization centroids | Adds a separate memory term |
| $`M_{\mathrm{index}}`$ | Standalone index footprint | Excludes replicas, rebuild, and payload |
| $`M_{\mathrm{index,budget}}`$ | Replicated standalone index subtotal plus stated reserve | Not complete fleet memory or full rebuild peak; payload placement is separate |
| $`R`$ | Replication factor | Multiplies the index footprint |
| $`h`$ | Explicit reserve fraction | Twenty-percent reserve cannot hold a complete extra equal-sized index |
| Rebuild headroom | Capacity for simultaneous build/live artifacts | Budget external build, transfer and swap peaks explicitly |
| Payload store | Retrieved text and related content | Keep separate from vector index footprint |
| Shards and replicas | Partition and redundancy counts | Include nonrandom partition key |
| Partitioning key | Rule that assigns items to shards | Must be stated when sharding is not random |
| Exact-search recall | Approximate recall against exact search | Different from semantic relevance |
| Exact search | Exhaustive nearest-neighbor reference | Supplies the comparison target for approximate recall |
| Held-out sample | Evaluation items excluded from index tuning | Supplies the exact-search recall test set |
| $`\mathrm{ef}`$ | HNSW search-time parameter | Needs a recall-latency curve |
| $`n_{\mathrm{probe}}`$ | IVF search-time probe count | Needs at least three measured values |
| Recall-latency curve | Recall and latency across search settings | Requires at least three values |
| Production concurrency | Simultaneous serving load | Required context for $`p_{50}`$ and $`p_{95}`$ latency |
| Build | Wall-clock time, cost, and location | State in-place versus separate box |
| In-place build | Rebuild on the serving environment | Must state what queries return during it |
| Separate-box build | Build away from the serving index | Worked example hot-swaps the artifact afterwards |
| Hot swap | Replace the served artifact after an external build | Reduces in-place overlap requirements |
| Churn | Insert, update, and delete rates | Describes index change pressure |
| Upsert | Insert or update operation | Worked lifecycle reports a daily rate |
| Tombstone | Logical deletion marker | Track policy, current fraction, and failure threshold |
| Tombstone fraction | Share of entries logically marked deleted | Quality and latency effects depend on implementation and deletion pattern |
| Rebuild trigger | Condition that starts a rebuild | Should precede the observed recall-fall region |
| Rebuild cadence | Scheduled replacement frequency | Part of the lifecycle record |
| Rebuild | Artifact replacement process | Trigger, cadence, duration, and query behavior |
| Fill rate | Fraction of records with a metadata field | Exposes unreliable filters |
| Conflict rule | Which store wins disagreement | Prevents undefined multi-store behavior |
| Fire rate | Fraction of queries that trigger retrieval | Makes gate behavior measurable |
| Blast radius | Impact of one poisoned chunk | Quantifies adversarial exposure |
| USD | United States dollar amount | States worked query and build costs without a currency symbol |
| `M` and `k` count suffixes | Million and thousand in corpus or churn counts | Distinct from model names and retrieval or quantization symbols |
| MB, GB, ms, minutes, and hours | Megabytes, gigabytes, milliseconds, minutes, and hours | Keep memory, latency, build, and freshness units distinct |


## Core mechanics
### Introduction
Retrieval systems pass between teams more often than teams rewrite them. A new owner asks predictable questions, so the two templates answer those questions in advance.
The RAG Card documents system behavior; the Index Datasheet documents artifact state and operation. Keep separate owners and version links, but update both whenever a change invalidates their claims, including runtime search settings or source/permission changes between rebuilds.
The records adapt model-card and dataset-datasheet documentation practices to retrieval systems. Their purpose is to preserve intended use, assumptions, measurements, accountability and known gaps, not to certify an artifact merely because every template cell is populated.
The template itself is not the value. The value appears when a field cannot be filled. An unfillable field creates the fastest available retrieval-system audit and an interview question in either direction.
### D.1 The RAG Card
#### Purpose, ownership, and use
The RAG Card answers what the system does and does not do. It exposes behavior that is otherwise hidden behind the generator, including the datastore and five retrieval decisions.
Review the card after any material system change, with a named owner and on-call rotation. Store the evaluation versions and evidence behind its last-accurate date.
Fill every field. A field that cannot be filled is the finding.
#### Template 1: Identity and scope


| Field | Prompt to fill |
| --- | --- |
| System name and version | Give the system name and version. Include the date the card was last accurate. |
| Intended use | Name the question types this system is built to answer. |
| Out of scope | Name question types it will answer badly. Be specific. `General knowledge` is not an answer. |
| Owner | Name the team and on-call rotation. |


#### Template 2: The five decisions


| Decision | Prompt to fill |
| --- | --- |
| When | State the retrieval gate, gate family, and measured fire rate. If always-on, say so and give the measured cost on queries that did not need retrieval. Source section $`25.1`$. |
| Where | List every store, its size, its owner, and the conflict-resolution rule between stores. Source section $`25.3`$. |
| What | State the query transformation and the queries on which it is applied. Source chapter 24. |
| How | State the scorer, index family, retrieval depth $`k^{\prime}`$, rerank depth, and whether a lexical channel exists. |
| Present | State final $`k`$, generator ordering rule, serialization format, and retrieved text position relative to instructions. Source chapter 30. |


#### Template 3: Measured behavior


| Field | Prompt to fill |
| --- | --- |
| Retrieval | Report judged evidence Recall@$`k^{\prime}`$ and nDCG@$`k`$, corpus eligibility, query count, sampling, gain, zero-IDCG/unjudged policy, judges, agreement statistic and uncertainty. Keep ANN recall separate. |
| Generation | Report faithfulness and factuality separately at claim level. Source chapter 33. |
| Closed-book baseline | Report end-to-end accuracy with retrieval disabled. Without it, system value is unmeasured. |
| Gold-context reference | Report matched fixed-generator oracle-evidence accuracy and uncertainty. Interpret as conditional evidence headroom, not an absolute bound on all improvements. |
| Abstention | Report overall and branch-specific refusal rates, accepted-answer accuracy, denominators, uncertainty and empty-accepted-set policy. |
| Latency and cost | Report relevant TTFT/end-to-end percentiles under load, errors/retries, measured stage traces and resource/provider cost allocation. Do not add independent stage percentiles. |


#### Template 4: Known failure modes


| Field | Prompt to fill |
| --- | --- |
| Query types that fail | Name each failing query type and give an example. |
| Freshness lag | Report time from a document change until the change is retrievable. |
| Attribution quality | Define evidence availability, source/version/span resolution, inline/post-hoc method, support precision, claim coverage and unknowns. Timing alone is not a grounding certificate. |
| Adversarial surface | State who can write into the datastore and the measured blast radius of one poisoned chunk. Source chapter 36. |


#### Filled RAG Card example, abbreviated


| Field | Value |
| --- | --- |
| System | Support Assistant retrieval, v4.2, accurate as of 2026-06-01 |
| Intended use | Product how-to and troubleshooting from published docs |
| Out of scope | Pricing, contractual terms, and anything account-specific |
| When | Illustrative classifier fires on $`68\%`$ of requests. Other requests may answer only under the allowed closed-book policy or abstain; they are not all assumed to refuse. |
| Where | Docs contain $`1.2\,\mathrm{M}`$ chunks. Tickets contain $`400\,\mathrm{k}`$ resolved-only chunks. Docs win on conflict. |
| What | Conversational rewrite on turns greater than one. Identity otherwise. |
| How | BM25 plus dense; RRF constant $`c_{\mathrm{RRF}}=60`$; retrieve 100, score 20 with the cross-encoder, present five. State that recall is measured before this candidate truncation. |
| Present | $`k=5`$, most-relevant first and last, JSON blocks after the instruction |
| $`\operatorname{Recall}@100`$ | Scenario value $`0.91`$ on $`2{,}000`$ judged queries; reported $`\kappa=0.71`$. Macro/micro aggregation, multi-judge statistic and intervals still need specification. |
| $`\operatorname{nDCG}@5`$ | $`0.68`$ |
| Closed-book | $`0.42`$ end-to-end |
| Gold-context | Fixed-generator scenario reference $`0.88`$, not a universal ceiling. |
| Deployed | Scenario all-request accuracy $`0.79`$ with refusals counted incorrect; overall refusal $`9\%`$. Accepted-answer accuracy is $`0.79/0.91\approx0.868132`$ under that convention. |
| $`p_{95}`$ latency | Scenario $`1.9\,\mathrm{s}`$ total and $`640\,\mathrm{ms}`$ TTFT, separately estimated percentiles; reported cost $`\$0.0041`$/query needs allocation/load details. |
| Freshness lag | $`45`$ minute median, $`4`$ hour $`p_{99}`$ |
| Attribution | Inline links; 162 of 200 sampled answers ($`81\%`$) labeled attributable in this scenario. Specify the all-claims-support rubric and sampling; this is answer-level, not claim-level coverage. |
| Write access | Docs team is reviewed. Ticket ingest is automated and unreviewed. |


#### Cross-field audit
Audit the automated, unreviewed ticket feed as a possible injection path even though documentation wins conflicts. Inline citations do not prove support or safety. The worked gate has been clarified to permit policy-authorized no-retrieval answers or abstention; otherwise a 68-percent retrieval rate with every skipped request refused would force at least 32-percent overall abstention, contradicting the example’s 9 percent.
### D.2 The Index Datasheet
#### Purpose, ownership, and use
The Index Datasheet adapts dataset documentation of provenance, composition, and intended use to a physical artifact with a memory footprint, rebuild cost, recall curve, and pinned embedding representation. It lets downstream users judge fitness instead of guessing.
Update the datasheet when its content, build, representation, search controls, deletion/retention policy or measured performance changes. A rebuild is one trigger, not the only one; preserve the previous version and link the serving release.
The datasheet exists because dashboard summaries do not preserve every assumption that determines migration cost.
#### Template 1: Composition


| Field | Prompt to fill |
| --- | --- |
| Corpus | List source systems, document count, chunk count, and snapshot date. |
| Chunking | State scheme, target size, overlap, and evidence that the scheme was chosen rather than inherited. Source chapter 13. |
| What was dropped | List excluded records/modalities and their counts, policy and sampled effects. Silent exclusions are one possible cause of missing evidence, not a universally measured most-common cause. |
| Metadata | List every filterable field, its cardinality, and its fill rate. Source section $`17.4`$. |


Composition fixes what the artifact contains at one point in time. Counts without a snapshot date cannot establish freshness. A chunking scheme without selection evidence can be inherited debt rather than a tested choice.
Dropped content is an explicit negative inventory. It separates retrieval failure from ingestion exclusion.
#### Template 2: Representation


| Field | Prompt to fill |
| --- | --- |
| Embedding model | Pin query/document encoders, versions, tokenizer, pooling, dimension and projection. State the compatibility or dual-index routing plan for migration. |
| Normalization | State whether vectors are $`L_2`$-normalized and name the distance metric that assumes it. |
| Precision | State stored precision and quantization. Include the measured recall cost. Source chapter 16. |


Pin the complete query/document representation and metric. Arbitrarily scoring incompatible model spaces together is invalid, but dual-index routing, calibrated fusion and verified compatible migrations can support a partial transition. Query-only changes need not force every document vector to change.[\[1\]](../part-08-training/28_training_the_retriever.md)
Quantization is incomplete without its measured recall cost.
#### Template 3: Structure and footprint


| Field | Prompt to fill |
| --- | --- |
| Index family | Name HNSW, IVF, IVF-PQ, disk-resident, or another family. Include build parameters. |
| Per-vector bytes | Separate codes/vector values, IDs and links, plus family-level centroids/codebooks. Structural bytes do not shrink merely with vector quantization; they are not inherently incompressible. |
| Footprint | Report standalone $`M_I`$ subtotal, all-in resident/runtime measurement, reserved capacity, peak old/new overlap and each artifact’s replication/placement. Keep payload copies explicit. |
| Shards and replicas | Give counts and the partitioning key when partitioning is not random. Source section $`17.2`$. |


Vector-code compression is not whole-index or fleet compression. Count IDs, family-specific links, centroids, codebooks, maps, allocation and runtime, then replicate each artifact under its own policy. A reserved fraction is not automatically enough for simultaneous old/new artifacts.[\[2\]](../part-12-systems-design/41_end_to_end_design_drills.md)
The payload must remain a separate line so text cannot disappear inside an index-only total.
#### Template 4: Performance


| Field | Prompt to fill |
| --- | --- |
| Recall | Report $`\operatorname{Recall}@k`$ against exact search on a held-out sample at the production search parameter. State that this differs from relevance. |
| Search parameters | Give current $`\mathrm{ef}`$ or $`n_{\mathrm{probe}}`$ and a recall-latency curve across at least three values. |
| Latency | Report $`p_{50}`$ and $`p_{95}`$ at production concurrency, not one query in isolation. |
| Build | Report wall-clock build time, cost, and whether the build is in place or on a separate box. |


Exact-neighbor recovery compares ANN with the same metric and eligible corpus; human evidence relevance addresses a different target. Document ties, filters, sample selection and ground-truth computation for both.
One search parameter cannot show a trade-off curve. One single-query latency cannot represent production concurrency.
#### Template 5: Lifecycle


| Field | Prompt to fill |
| --- | --- |
| Churn | Report insert, update, and delete rates. |
| Deletion | State deletion API/markers, consistency, read-path verification, replicas/caches/derived copies and retention/backup policy. Report measured quality/latency effects where tombstones apply. |
| Rebuild | State trigger, cadence, duration, and what queries return during the rebuild. |
| Freshness | Report median and $`p_{99}`$ lag from source change to retrievability. |


Lifecycle fields define what happens after the build completes. Delete rate without tombstone policy cannot predict degradation. A rebuild duration without query behavior leaves availability unspecified.
Freshness reports the full source-to-index path rather than the age of a source document alone.
#### Filled Index Datasheet example, abbreviated


| Field | Value |
| --- | --- |
| Corpus | Product docs, snapshot 2026-05-28. $`180\,\mathrm{k}`$ documents become $`1.24\,\mathrm{M}`$ chunks. |
| Chunking | Recursive, $`512`$ tokens, $`64`$ overlap. Headings retained as chunk prefix. |
| Dropped | Images, code blocks over $`200`$ lines, and $`3.1\%`$ of documents over the size cap |
| Embedding | e5-base-v2, $`d=768`$, unit normalization and cosine; fp32 is the explicit raw-vector comparison assumption for the byte calculation, not a new quality measurement. |
| Index | IVF-PQ, $`n_{\mathrm{list}}=4{,}096`$, $`m=96`$, $`k_{\mathrm{sub}}=256`$; eight-bit subcodes and eight-byte posting IDs in this revised layout. |
| Bytes per vector | $`96\,\mathrm{B}`$ code $`+8\,\mathrm{B}`$ ID $`=104\,\mathrm{B}`$; no HNSW links. Other per-entry/allocation costs omitted. |
| $`M_I`$ | $`128.96\,\mathrm{MB}`$ codes/IDs; $`12.582912\,\mathrm{MB}`$ coarse centroids; $`0.786432\,\mathrm{MB}`$ PQ codebook; subtotal $`142.329344\,\mathrm{MB}`$. |
| $`M_{\mathrm{index,budget}}`$ | $`R=2`$, reserve $`h=0.2`$: $`341.590426\,\mathrm{MB}`$. Not full old/new peak. The $`2\,\mathrm{GB}`$ payload has a separate placement/replication policy. |
| Shards | One logical shard is an illustrative choice, not a demonstrated node plan. Usable RAM, runtime, payload placement, load and replica/failure capacity must establish fit. |
| $`\operatorname{Recall}@10`$ | $`0.94`$ versus exact at $`n_{\mathrm{probe}}=32`$ |
| Recall curve | $`0.87`$ at $`8`$, $`0.94`$ at $`32`$, $`0.97`$ at $`128`$ probes |
| Latency | $`p_{50}`$ $`4\ \mathrm{ms}`$, $`p_{95}`$ $`11\ \mathrm{ms}`$ at $`40`$ concurrent |
| Build | Source scenario: 38 minutes and $`\$14`$, separate build box then swap. Build-box resources and transfer/old-new overlap must be budgeted separately. |
| Churn | About $`2\,\mathrm{k}`$ upserts per day and about $`200`$ deletes per day |
| Tombstones | $`1.8\%`$ now. Recall measured flat to $`6\%`$. Rebuild triggers at $`5\%`$. |
| Freshness | $`45`$ minute median, $`4`$ hour $`p_{99}`$ |


#### Cross-field capacity audit
The revised arithmetic example uses $`1{,}240{,}000`$ vectors with 96-byte PQ codes and eight-byte posting IDs. Codes plus IDs use $`128.96\,\mathrm{MB}`$; coarse centroids use $`12.582912\,\mathrm{MB}`$ and the PQ codebook $`0.786432\,\mathrm{MB}`$. Their scoped subtotal is $`142.329344\,\mathrm{MB}`$ before maps, allocation and runtime. A $`2\,\mathrm{GB}`$ text payload is about $`14.052`$ times that subtotal; text is the larger scoped term, not necessarily the entire fleet bottleneck.[\[3\]](../part-05-indexing/16_compression_and_index_economics.md)
For an explicitly assumed fp32 uncompressed reference, $`768\times4=3{,}072\,\mathrm{B/vector}`$ versus 96 code bytes gives exactly $`32`$-fold vector-code compression. The 104-byte code-plus-ID layout does not have that same ratio. The source’s four-byte-ID toy layout was 100 bytes per vector; a Faiss-style eight-byte-ID layout is 104 bytes, so its source totals cannot be copied unchanged.[\[3\]](../part-05-indexing/16_compression_and_index_economics.md)
With $`R=2`$ and reserved index headroom $`h=0.2`$, the index capacity budget is $`R(1+h)M_I=341.5904256\,\mathrm{MB}`$. This is not a full old/new overlap budget: if both complete versions coexist at both replicas, their index subtotal is $`2RM_I=569.317376\,\mathrm{MB}`$. Payload placement is separate: one shared $`2\,\mathrm{GB}`$ payload plus the reserve gives $`2.3415904256\,\mathrm{GB}`$; two payload copies give $`4.3415904256\,\mathrm{GB}`$. All omit the payload store’s other replication/runtime and external build resources.[\[2\]](../part-12-systems-design/41_end_to_end_design_drills.md)
The source scenario reports recall flat through a measured $`6\%`$ tombstone fraction and a $`5\%`$ trigger. That is scoped evidence, not a guarantee under all delete patterns or a measurement of physical erasure. Test latency, authorization/deletion visibility and headroom too. The 45-minute median and four-hour $`p_{99}`$ freshness are distributions, not maximum deadlines.
#### Handoff audit sequence
1. Check the card’s last-accurate date and the datasheet’s snapshot date.
2. Match every card store to a datasheet corpus or to an explicitly separate artifact.
3. Match the card’s scorer and index family to the datasheet’s pinned representation and build parameters.
4. Match retrieval depth, rerank depth, and final context depth to the metrics reported at those depths.
5. Keep human relevance Recall separate from approximate-index Recall against exact search.
6. Compare end-to-end $`p_{95}`$ and TTFT with index-only $`p_{50}`$ and $`p_{95}`$ at production concurrency.
7. Compare card freshness with datasheet freshness to verify one source-to-retrieval path.
8. Read store ownership, write access, review status, and poisoned-chunk blast radius together.
9. Read vector compression, link share, fleet memory, and payload on separate lines.
10. Assign every missing quantity to an owner rather than leaving the field silently blank.
#### Cross-record consistency matrix


| RAG Card field | Index Datasheet field | Consistency question |
| --- | --- | --- |
| System version and last-accurate date | Snapshot date | Do the documents describe the same operating period? |
| Intended use | Corpus and dropped content | Does the index contain the modalities needed by the intended questions? |
| Out of scope | What was dropped | Are exclusions reflected in the negative system boundary? |
| Owner | Shards, replicas, build, and rebuild | Is artifact operation assigned as clearly as system on-call? |
| When | Latency and build cost | Is gate behavior justified by the downstream cost it avoids? |
| Where | Corpus and metadata | Does every routed store have composition and filter evidence? |
| What | Chunking and representation | Does transformation match the artifact being searched? |
| How | Index family and search parameters | Do serving settings match the documented index? |
| Present | Recall and recall curve | Do final evidence depth and retrieval settings support the claimed behavior? |
| Retrieval metrics | Exact-search Recall | Are relevance and approximation quality kept separate? |
| Latency and cost | Latency and build | Do endpoints, workloads, load, service-cost allocation and per-request traces agree? Stage percentiles are not additive. |
| Freshness lag | Freshness, churn, and rebuild | Can the lifecycle explain the observed lag? |
| Attribution quality | Dropped content and payload | Does every claimed source/version/span exist, remain authorized and support the complete claim? |
| Adversarial surface | Corpus writers and lifecycle | Are automated write paths measured and bounded? |


Use these consistency checks without merging the records. The card’s rounded 1.2-million documentation chunks and the datasheet’s 1.24-million May snapshot need explicit date/rounding reconciliation; the 400,000-ticket store needs its own datasheet. Card policy and dates do not by themselves establish that every artifact is covered.
The strongest handoff evidence is reconciled evidence. A system-level number should point to an artifact-level condition, while an artifact-level change should identify which card decision or measurement it can invalidate.
## Diagrams
Manifest accounting is 0 numbered figures and 0 numbered tables. The source contains two unnumbered blank templates and two unnumbered abbreviated filled examples. All four are recreated as Markdown tables under Core mechanics. No figure number, table number, or source caption is invented.
### Unnumbered template accounting
- Blank RAG Card: four category tables, 19 field rows.
- Filled RAG Card: 17 illustrative field rows; values are not independently reproduced measurements.
- Blank Index Datasheet: five category tables, 19 field rows.
- Filled Index Datasheet: 16 field rows; revised byte arithmetic uses explicitly stated assumptions, not a new benchmark.
## Whiteboard pack
### What to draw
1. Draw two adjacent boxes labeled RAG Card and Index Datasheet.
2. Write decision change above the card and index rebuild above the datasheet.
3. Divide the card into identity, five decisions, measured behavior, and failures.
4. Put When, Where, What, How, and Present down the card’s center.
5. Add baseline, ceiling, abstention, latency, freshness, attribution, and write access.
6. Divide the datasheet into composition, representation, structure, performance, and lifecycle.
7. Add model pinning, vector-id-link bytes, exact-search recall, concurrency, and tombstones.
8. Draw separate index and payload boxes, then circle the larger payload in the example.
9. Connect unreviewed ingest to the adversarial surface.
10. Mark every blank field as an audit finding with an owner and evidence gap.
### Spoken script
Draw two version-linked records, with named owners and dated evidence. The card records scope, routing, context, baselines, support, selective behavior and operational failures. The datasheet records composition, representation compatibility, bytes by artifact, exact-neighbor recall, load, lifecycle and deletion verification. Reconcile stores and snapshots, gate/refusal denominators, payload copies and rebuild peaks. A filled template does not remove these evidence gaps.
## Interview traps
### 1. Why are there two templates instead of one?
They describe different scopes and may have different owners, but their change triggers overlap. Update whichever claims are invalidated by a release, source/permission change, search setting, index rebuild or measurement update; link both to the actual serving version.
### 2. Which five-decision answer is incomplete?
`When` needs gate family, fire rate, and unnecessary-retrieval cost, while `Where` needs every store, size, owner, and conflict rule. `What` needs transformation and triggering queries, while `How` needs scorer, index, $`k^{\prime}`$, rerank, and lexical channel. `Present` needs final $`k`$, order, serialization, and position relative to instructions.
### 3. What measurements must not be conflated?
Keep judged evidence relevance, ANN exact-neighbor recovery, claim support, truth and selective accuracy separate. A fixed-generator gold-context run is a diagnostic reference, not a theorem bounding all improvements. Percentiles need load and endpoint definitions; separately measured stage percentiles generally do not sum to the end-to-end percentile.[\[4\]](../part-10-evaluation/34_evaluating_the_system.md)
### 4. How does the card reveal an adversarial failure?
Read fields together. The worked card says citations are generated inline and ticket ingestion is automated and unreviewed, so treating that store as authoritative makes the write path an injection surface. The blank template also requires the measured blast radius of one poisoned chunk.
### 5. Why can a compressed index still require more capacity than expected?
Include codes, IDs, centroids, codebooks, links where applicable, runtime and the actual replica/rebuild placements. In the revised layout, $`142.329344\,\mathrm{MB}`$ is a scoped standalone subtotal and $`2\,\mathrm{GB}`$ is a separately placed payload. Exactly $`32`$-fold vector-code compression is not a whole-index or fleet reduction, and 20-percent reserve cannot hold a full extra equal-size artifact.
## Key numbers


| Template | Number or value | Meaning |
| --- | --- | --- |
| Source manifest | 0 figures, 0 tables | Templates are unnumbered source layouts |
| Documentation precedents | Model cards in 2019 and datasheets for datasets in 2021 | Cited traditions behind the two records |
| RAG Card scope | One page and 5 retrieval decisions | System-level handoff record |
| RAG Card blank template | 4 categories, 19 prompts | Identity 4, decisions 5, behavior 6, failures 4 |
| Index Datasheet blank template | 5 categories, 19 prompts | Composition 4, representation 3, structure 4, performance 4, lifecycle 4 |
| Filled RAG Card | 17 rows | Abbreviated worked example |
| Filled Index Datasheet | 16 rows | Abbreviated worked example |
| Card version | v4.2, accurate 2026-06-01 | Worked system identity |
| Retrieval gate | $`68\%`$ fires | No-retrieval branch may answer under policy or abstain; overall refusal is not forced to 32 percent |
| Stores | $`1.2\,\mathrm{M}`$ doc chunks and $`400\,\mathrm{k}`$ resolved ticket chunks | Docs win conflicts |
| Query rewrite | Turns greater than $`1`$ | Identity transformation otherwise |
| Retrieval funnel | Retrieve 100, score 20, present five | RRF constant $`c_{\mathrm{RRF}}=60`$; do not conflate fusion constant and context count |
| Recall | $`0.91`$ at $`100`$ | $`2{,}000`$ queries, three-way human judged, $`\kappa=0.71`$ |
| nDCG | $`0.68`$ at $`5`$ | Ranked final-context quality |
| Accuracy | $`0.42`$, $`0.88`$, $`0.79`$ | Scenario baseline, fixed-generator oracle reference and deployed accuracy; state scoring population |
| Abstention | $`9\%`$ overall; accepted accuracy $`0.868132`$ | Conditional arithmetic assumes 79-percent all-request accuracy and refusal counted incorrect |
| Card latency | $`1.9`$ seconds total, $`640\ \mathrm{ms}`$ TTFT | $`p_{95}`$ worked value |
| Card cost | $`\mathrm{USD}\ 0.0041`$ per query | Worked end-to-end cost |
| Card freshness | $`45`$ minute median, $`4`$ hour $`p_{99}`$ | Source-to-retrieval lag |
| Attribution | $`162/200=81\%`$ answers | Answer-level scenario rubric; inline links do not prove causal use or truth |
| Adversarial measurement | One poisoned chunk | Unit for the required blast-radius measurement |
| Corpus | $`180\,\mathrm{k}`$ documents to $`1.24\,\mathrm{M}`$ chunks | Snapshot 2026-05-28 |
| Chunking | $`512`$ tokens, $`64`$ overlap | Recursive with heading prefix |
| Dropped | Code blocks over $`200`$ lines, $`3.1\%`$ over cap | Images also excluded |
| Embedding | $`d=768`$; fp32 raw reference | The uncompressed arithmetic assumption is now explicit |
| IVF-PQ | $`n_{\mathrm{list}}=4{,}096`$, $`m=96`$, $`k_{\mathrm{sub}}=256`$ | 96 eight-bit subcodes; no HNSW links |
| Bytes per vector | $`96+8=104\,\mathrm{B}`$ | Revised code plus eight-byte-ID layout, excluding other per-entry/runtime costs |
| Vector-code compression | $`3{,}072/96=32`$ | fp32 raw vector to PQ code only; not all-index/fleet compression |
| Index subtotal | $`142.329344\,\mathrm{MB}`$ | $`128.96\,\mathrm{MB}`$ codes/IDs plus exact coarse-centroid and PQ-codebook terms |
| Reserved index budget | $`341.590426\,\mathrm{MB}`$ | Two copies and 20-percent reserve, not full old/new overlap |
| Payload | $`2\,\mathrm{GB}`$ per stated payload copy | About $`14.052`$ times index subtotal; replica and placement policy still required |
| Logical shards | One in the illustration | Not a proven usable-RAM, throughput or availability plan |
| Exact Recall | $`0.94`$ at $`10`$ and $`n_{\mathrm{probe}}=32`$ | Approximate versus exact search |
| Recall curve | $`0.87`$ at $`8`$, $`0.94`$ at $`32`$, $`0.97`$ at $`128`$ | Three probe settings |
| Search-curve minimum | At least 3 values | Required to show the recall-latency trade-off |
| Index latency | $`p_{50}`$ $`4\ \mathrm{ms}`$, $`p_{95}`$ $`11\ \mathrm{ms}`$ | Measured at $`40`$ concurrent |
| Build | 38 minutes; $`\$14`$ | Source scenario; external resources and old/new transfer peak separately budgeted |
| Churn | About $`2\,\mathrm{k}`$ upserts and $`200`$ deletes daily | Worked lifecycle rate |
| Tombstones | $`1.8\%`$ now; tested to $`6\%`$; trigger $`5\%`$ | Scoped quality observation, not a universal deletion guarantee |
| Datasheet freshness | $`45`$ minute median, $`4`$ hour $`p_{99}`$ | Matches worked card lag |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A3\_design\_checklists](A3_design_checklists.md) · [A5\_annotated\_reading\_list](A5_annotated_reading_list.md)
