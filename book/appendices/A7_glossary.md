# Appendix G: Glossary
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A6\_notation\_quick\_reference](A6_notation_quick_reference.md)
This appendix maps overlapping RAG vocabulary while stating local conventions, mathematical assumptions and limits. Aliases identify related usage, not universal equivalence; follow the cited chapter for the precise mechanism.
## TL;DR
- Three literatures often give the same RAG object different names: information retrieval, representation learning, and systems.
- An alias is not always an exact synonym. A role, a product label, and one implementation can sound interchangeable while meaning different things.
- Faithfulness asks whether an answer follows the retrieved context. Factuality asks whether the answer is true of the world.
- Index recall compares approximate search with exact search under one scorer. Retrieval relevance compares results with human judgment.
- Retrieval depth $`k^{\prime}`$ feeds the reranker. The separate count $`k`$ reaches the generator.
- Use the book’s stated convention and source pointer when a term is contested. Do not average incompatible definitions.
## The story
Imagine a railway station shared by three rail companies. One company calls a boarding area a platform, another calls it a gate, and the third calls it a bay. The object stays the same, but a traveler can miss the train by treating a familiar object as unfamiliar.
The alias list is the station’s translation board. It groups names that point to one object. It also posts warnings when a label is broader than the object. A reranker, for example, is a job at the station. A cross-encoder is one worker that can perform that job.
The glossary is the station handbook. Each entry gives the local meaning and the location of the full operating rule. The handbook follows this station’s convention instead of averaging every railway’s usage.
Three red warning cards prevent the worst transfers. The first separates a conductor who repeats the printed schedule faithfully from a train that actually arrives on time. That is faithfulness versus factuality.
The second separates a routing machine that reproduces its own exact route from a route that takes a passenger where people say they need to go. That is index recall versus retrieval relevance.
The third separates passengers sent to inspection from passengers allowed onto the train. Those counts are $`k^{\prime}`$ and $`k`$. They serve different objectives.
The traveler succeeds by reading the translation board, checking each warning card, and opening the handbook at the cited rule. Vocabulary then becomes a navigation tool instead of a trap.
## Decoder table


| Technical term | Plain-English meaning | Why it matters |
| --- | --- | --- |
| Chunk | Also called passage, segment, span, node, or document. It is the unit stored, embedded, and retrieved | Document can mean either the source file or the retrieval unit. Chunk size and overlap are design parameters. Chapter 13 |
| Datastore | Retrievable content, metadata and its serving/search structures | Index, corpus, knowledge base and vector store are related but not universally synonymous; enumerate their scope |
| Embedding | A mapping or representation, often a dense vector | Latent and encoding are broader terms; specify representation compatibility |
| Bi-encoder | Independently encode query and document | Scoring still interacts with both; towers need not share weights, unlike typical siamese usage |
| Cross-encoder | Joint query-document encoding for scoring | Reranking is a role; no generic one-vector document index reproduces every cross-encoder score |
| Approximate nearest-neighbor search | Approximate neighbor recovery under a metric | Vector/nearest-neighbor search may be exact; semantic search is broader than this algorithm |
| Retrieval depth $`k^{\prime}`$ | Also called candidate set size, fan-out, recall set, or first-stage depth | It feeds the reranker. It is distinct from $`k`$, the count reaching the generator |
| Query reformulation | Also called query rewriting, query expansion, query transformation, or query understanding | Expansion adds terms. Rewriting replaces terms. The umbrella name varies by paper |
| Relevance judgment | A query-item relevance label or qrel | A gold label can be incomplete or disputed; state pooling, gain and unjudged-item policies |
| Hard negative | A genuinely non-relevant item scored as confusingly relevant | Not every distractor is hard, mined or correctly labeled; audit false negatives |
| Faithfulness | Also called groundedness, consistency, or attribution to source. It asks whether an answer follows retrieved context | It is distinct from factuality, which asks about the world. Section $`33.1`$ |
| Factuality | Also called correctness, truthfulness, or factual accuracy. It asks whether an answer is true of the world | An answer can be faithful to a wrong document and still be non-factual. Section $`33.1`$ |
| Abstention | Declining an answer under an explicit policy | Uncertainty abstention, safety refusal and human deferral can require different denominators/costs |
| Retrieval gate | Also called adaptive retrieval, active retrieval, when-to-retrieve, or routing. It is $`g(x)`$, which decides whether to retrieve | Routing usually also covers where to retrieve. The gate is decision one. Section $`25.1`$ |
| Lost in the middle | A measured evidence-position effect | A middle trough is empirical, not universal or identical to every positional bias |
| Learned sparse | Also called sparse neural, expansion-based sparse, or Sparse Lexical and Expansion Model (SPLADE)-style retrieval | It predicts vocabulary-term importance while retaining sparse retrieval structure |
| Late interaction | Match independently encoded token or patch representations later | Multi-vector retrieval is broader; not all variants use ColBERT MaxSim |
| Prompt injection | Adversarial instructions crossing an instruction/data trust boundary | Direct and indirect forms differ; indirect channels include retrieval and tool results |
| Agentic RAG | A controller chooses retrieval actions and stopping | Expected and tail work depend on rounds, context growth, retries and parallelism |
| Attribution | The property that a statement is supported by a cited source | It differs from citation, which is the act of pointing. Chapter 31 |
| BM25 | The book’s name for a lexical ranker using inverse document frequency (IDF) times saturating, length-normalized term frequency | It states the standard lexical scoring convention. Section A.1 |
| Calibration | Agreement between stated model confidence and observed accuracy | Failure prediction only needs to rank confidence, so it is a different requirement. Section $`11.4`$ |
| Contrastive Language-Image Pre-training (CLIP) | Jointly trained image and text encoders place matching image-text pairs near each other | It is the default multimodal retriever in the source and carries three biases worth naming. Section $`39.3`$ |
| Closed-book | No retrieved evidence is supplied for answering | Use a matched baseline; this glossary does not establish how often evaluations omit it |
| ColBERT | A contextual-token late-interaction retriever using configured MaxSim | Precomputable document vectors trade storage and fine-grained matching; accuracy gains are empirical |
| Generative retrieval | Generate document identifiers with validity and access handling | Cost is not universally constant in corpus size; update/quality limitations are design dependent |
| Graph RAG | Graph entities/relations or communities augment chunk evidence | Relational-query gains must justify extraction, resolution and maintenance costs |
| Hallucination | A generated statement unsupported by the context or the world | The useful taxonomy separates factuality from faithfulness and intrinsic from extrinsic. Section $`4.4`$ |
| High-bandwidth memory | Accelerator-attached high-throughput memory | Decode transfers active weights and KV state, not necessarily all installed memory each token |
| HNSW | Layered proximity graphs with probabilistic levels and greedy routing | Graph storage can dominate compressed codes but links are not inherently incompressible |
| HyDE | Generate a hypothetical document/answer and retrieve with its embedding | The generated text is not itself verified evidence |
| Hybrid retrieval | Combine complementary channels via a defined combiner | Parallel lexical/dense RRF is common, not the only topology or fusion rule |
| In-context learning | A model conditions on examples or evidence in its prompt without weight updates | RAG depends on this mechanism. Chapter 7 |
| Inverted file index (IVF) | It partitions space into Voronoi cells around centroids | It searches only the nearest $`n_{\mathrm{probe}}`$ cells. Section $`15.3`$ |
| Locality-sensitive hashing | Hash families with specified near/far collision guarantees | Sublinear bounds and comparisons with graph indexes depend on parameters and workload |
| Modularity | One null-model-relative graph-community objective | Not all community algorithms maximize it; a heuristic need not find its global maximum |
| Parametric memory | Information represented by learned parameters | No native fact-key delete; rollback, editing, unlearning and retraining have distinct guarantees |
| Product quantization | $`m`$ subquantizers with $`k`$ centroids per subspace | At most $`k^m`$ combinations; $`km`$ subspace centroids contain $`kd`$ scalars for equal subspaces |
| Indirect prompt injection | Adversarial instructions embedded in an external data channel | Retrieved text, tool results and other untrusted sources can carry them |
| $`\operatorname{Recall}@k`$ | Relevant eligible items recovered within cutoff | Specify judged versus exact-neighbor reference, aggregation and zero-denominator policy |
| Reranking | Score/reorder first-stage candidates | May improve ordering, not guaranteed; pure reranking cannot add absent evidence |
| Reciprocal rank fusion | Unweighted score $`\sum_{\ell:d\in L_\ell}1/(c+r_\ell(d))`$ | One-based ranks, stated constant and absent-list zero; not a context-depth formula |
| Self-RAG | A generator emits reflection tokens that control retrieval and critique its own output | The generator learns control as well as generation. Section $`26.6`$ |
| Semi-parametric | A system combines parametric model weights with a non-parametric datastore | This is the frame behind every RAG design question in the source. Section $`1.1`$ |
| SPLADE | A learned sparse retriever family with vocabulary weights | Can reweight or expand queries/documents; exact activation/pooling differs by variant |
| Tombstone | A logical deletion marker | Measure read-time enforcement and maintenance; accumulation does not universally force one repair strategy |
| Vocabulary mismatch | A user and a relevant document use different words | Query expansion, document expansion, and dense retrieval attack the gap differently. Section $`24.1`$ |


## Core mechanics
### Introduction: why a glossary is needed
What: Two obstacles get in a reader’s way. Only one is conceptual.
Why: Three independent literatures use different names for the same objects. All three vocabularies remain active.
Failure without the aliasing table: A familiar object can sound unfamiliar in a paper, vendor document, or interview question.
What the glossary adds: It gives a needed definition at the moment it is needed and points to the full argument.
Convention: Each definition states this book’s usage. It does not average the field.
Qualifier: The source calls out contested distinctions instead of smoothing them over. These distinctions test whether the candidate’s vocabulary is real.
Cost or complexity: The source states no numeric implementation cost for the glossary itself.
### G.1 The Same Idea Under Four Names
Retrieval-Augmented Generation sits where information retrieval, representation learning, and systems meet. The three literatures developed independently. Their names survive in current papers, vendor documentation, and interview questions.
The source uses a bi-encoder and a two-tower model as its example. A candidate who misses the alias hears a familiar question as an unfamiliar one.
Terms on one alias row denote the same broad idea. The caveat column preserves the places where that equivalence is incomplete.
#### Chunk
Also called: passage, segment, span, node, and document.
Qualifier: Document is ambiguous. It can mean the source file or the retrieval unit.
Failure without the qualifier: A design can confuse the stored unit with the original file.
Cost or complexity: No separate cost is stated in G.1.
#### Datastore
Also called: index, corpus, knowledge base, vector store, and non-parametric memory.
Qualifier: An index is a search structure; a corpus is its content collection, a datastore also includes serving data and metadata, and a vector store is one implementation. “Knowledge base” may include structured facts, so these are not exact synonyms.
Failure without the qualifier: A discussion can merge retrievable content with the structure used to search it.
Cost or complexity: No separate cost is stated in G.1.
#### Embedding
Also called: dense vector, representation, latent, and encoding.
Qualifier: An embedding is a mapping/representation, often but not always a dense vector. “Latent” and “encoding” are broader terms; compatibility depends on the model, projection, dimension and metric.
Failure without the aliases: Representation-learning vocabulary can hide a familiar vector object.
Cost or complexity: No separate cost is stated in G.1.
#### Bi-encoder
Also called: dual encoder, two-tower, siamese network, and dense retriever.
Qualifier: A two-tower/dual encoder independently represents each side; its towers may or may not share weights. “Siamese” commonly implies weight sharing. DPR is one instance, and not every dense retriever is a standard bi-encoder.
Failure without the qualifier: One named implementation can be mistaken for the entire architecture class.
Cost or complexity: No separate cost is stated in G.1.
#### Cross-encoder
Also called: interaction model, joint encoder, reranker, and scorer.
Qualifier: Reranker is a role. A cross-encoder is one way to fill that role.
Failure without the qualifier: A pipeline position can be mistaken for one architecture.
Cost or complexity: No separate cost is stated in G.1.
#### ANN search
Also called: vector search, similarity search, semantic search, and nearest-neighbor search.
Qualifier: Vector and nearest-neighbor search can be exact or approximate. “Semantic search” is broader than ANN and need not denote one index algorithm or the entire same pipeline.
Failure without the qualifier: A full product can be confused with its approximate search component.
Cost or complexity: No separate cost is stated in G.1.
#### Retrieval depth $`k^{\prime}`$
Also called: candidate set size, fan-out, recall set, and first-stage depth.
Qualifier: $`k^{\prime}`$ differs from $`k`$, the count that reaches the generator.
Failure without the qualifier: One depth can be optimized as if it served two separate objectives.
Cost or complexity: The source gives no fixed values for $`k^{\prime}`$ or $`k`$ in this appendix.
#### Query reformulation
Also called: query rewriting, query expansion, query transformation, and query understanding.
Qualifier: Expansion adds alternative terms or representations; rewriting transforms the query and may also add or remove information. They can overlap, so preserve the actual transformation and its trigger.
Failure without the qualifier: Two different transformations can be treated as identical operations.
Cost or complexity: No separate cost is stated in G.1.
#### Relevance judgment
Also called: qrel, label, annotation, ground truth, and gold.
Qualifier: A qrel is a query-document relevance judgment, not infallible truth. Gold labels can be incomplete or disputed; record the judgment scale, pool, adjudication and unjudged-item policy.
Failure without the aliases: Evaluation vocabulary can hide that each name denotes a relevance judgment.
Cost or complexity: No separate cost is stated in G.1.
#### Hard negative
Also called: distractor, confusing negative, and mined negative.
Qualifier: A distractor may be irrelevant to an answer without being a mined high-scoring training negative. Training mixtures such as RAFT deliberately include distractors, but not every distractor is hard and not every mined negative is truly irrelevant.
Failure without the qualifier: An ordinary distractor can be confused with a deliberately selected training example.
Cost or complexity: No separate cost is stated in G.1.
#### Faithfulness
Also called: groundedness, consistency, and attribution to source.
Qualifier: Faithfulness is distinct from factuality, which asks about the world.
Failure without the qualifier: Support from context can be mistaken for truth about the world.
Cost or complexity: No separate cost is stated in G.1.
#### Factuality
Also called: correctness, truthfulness, and factual accuracy.
Qualifier: An answer can be faithful and non-factual.
Failure without the qualifier: A wrong source can make a supported answer false.
Cost or complexity: No separate cost is stated in G.1.
#### Abstention
Also called: refusal, deferral, rejection, and “I don’t know.”
Qualifier: An uncertain-answer abstention, a policy refusal and deferral to a human can have different causes, costs and metrics. State which decisions are counted together.
Failure without the aliases: Different labels can hide the same decision to decline an answer.
Cost or complexity: No separate cost is stated in G.1.
#### Retrieval gate
Also called: adaptive retrieval, active retrieval, when-to-retrieve, and routing.
Qualifier: Routing usually also covers where to retrieve.
Failure without the qualifier: A broader routing decision can be reduced to a yes-or-no gate.
Cost or complexity: No separate cost is stated in G.1.
#### Lost in the middle
Also called: positional bias, serial position effect, and primacy and recency.
Qualifier: Primacy, recency and broader positional bias are related effects, not exact synonyms. A middle trough is an empirical pattern for specified models and tasks, not a universal context law.
Failure without the aliases: The position effect can look like several unrelated phenomena.
Cost or complexity: No separate cost is stated in G.1.
#### Learned sparse
Also called: sparse neural, expansion-based sparse, and SPLADE-style.
Qualifier: Learned sparse retrieval is broader than expansion-based or SPLADE implementations; some methods reweight existing terms without expanding them. State the representation and sparse scorer.
Failure without the aliases: A learned sparse method can be mistaken for a different retrieval family.
Cost or complexity: No separate cost is stated in G.1.
#### Late interaction
Also called: multi-vector retrieval, token-level matching, and ColBERT-style.
Qualifier: Late interaction delays query-document matching until after independent token or patch encoding. Multi-vector retrieval is a broader representation class; it does not always use ColBERT-style MaxSim.
Failure without the aliases: A per-token retrieval architecture can sound like a separate method family.
Cost or complexity: No separate cost is stated in G.1.
#### Prompt injection
Also called: indirect injection, context injection, and instruction hijacking.
Qualifier: Indirect injection arrives through an external data channel such as retrieval, a tool result, file or webpage, rather than an authorized instruction. The broader term prompt injection includes direct attacks too.
Failure without the qualifier: Retrieved instructions can be confused with instructions supplied directly by the user.
Cost or complexity: No separate cost is stated in G.1.
### Three distinctions worth keeping
Not every pair of terms is synonymous. The source identifies three collapses that can lose an interview exchange.
#### Faithfulness is not factuality
What: Faithfulness asks whether the answer follows from retrieved context. Factuality asks whether it is true of the world.
Example: A perfectly faithful answer to a wrong document is faithful and false.
Failure without the distinction: The two failure types receive the same diagnosis even though they require different fixes.
Cross-reference: Section $`33.1`$.
#### Recall is not relevance
What: Index recall compares approximate search with exact search under the same scoring function. Retrieval relevance compares results with human judgment.
Example: An index with $`0.99`$ recall against a bad scorer retrieves the wrong documents very reliably.
Failure without the distinction: Strong agreement with exact search can be misreported as strong agreement with people.
Cross-reference: Section $`15.6`$.
#### $`k`$ is not $`k^{\prime}`$
What: Retrieval depth $`k^{\prime}`$ feeds the reranker. The separate count $`k`$ reaches the generator.
Why: The two variables serve separate objectives.
Interview signal: The question “How do you choose $`k`$?” tests whether the candidate separates them.
Cross-reference: Section $`22.2`$.
### G.2 Glossary of Terms
The following entries keep the source’s exact alphabetical order. Each definition uses the book’s convention and retains its source pointer.
#### Abstention
What: Declining to answer instead of producing a low-confidence answer.
Why: It gives the system an alternative to an answer it does not support confidently.
Failure or qualifier: Chapter 11 compares five illustrative mechanism families. They are not an exhaustive theorem about all possible abstention methods; measure their risk/coverage trade-offs.
Cost or complexity: The appendix states the mechanism count but no shared cost.
Cross-reference: Chapter 11.
#### Agentic RAG
What: A controller chooses retrieval actions, queries and a stopping rule over one or more steps. The controller may be the generator, another model or explicit logic; tool calling is one implementation.
Why: Retrieval becomes a per-step decision rather than a fixed single stage.
Failure or qualifier: Report both expected and tail cost, latency and quality under a bounded stopping policy; neither tail nor mean is always the sole relevant constraint.
Cost or complexity: Total request cost is $`\sum_{t=1}^{R}C_t`$ for random round count $`R`$. It is proportional to rounds only when per-round cost is fixed; accumulated context, retries and parallelism change the profile.[\[1\]](../part-07-query-control/26_iterative_recursive_and_agentic_retrieval.md)
Cross-reference: Section $`26.8`$.
#### ANN search
What: Approximate nearest-neighbor search.
Why: Trading exactness for speed makes billion-scale retrieval possible.
Failure or qualifier: Approximation makes recall a tunable rather than a fixed guarantee.
Cost or complexity: The source states the speed-exactness trade but no numeric cost here.
Cross-reference: Chapter 15.
#### Attribution
What: The property that a statement is supported by a cited source.
Why: It tests support rather than merely displaying a pointer.
Failure or qualifier: Citation is the act of pointing. It is distinct from attribution.
Cost or complexity: No separate cost is stated.
Cross-reference: Chapter 31.
#### Bi-encoder
What: An architecture that encodes the query and document independently.
Why: Document vectors can be precomputed and indexed, which makes retrieval tractable.
Failure or qualifier: Standard bi-encoders omit joint query-document cross-attention during encoding. Their similarity scorer still interacts with both representations, and training can couple the two encoders.[\[2\]](../part-06-retrieval-ranking/20_dense_retrieval.md)
Cost or complexity: Precomputed document representations avoid a joint encoder pass per corpus pair. Retrieval quality and speed relative to other architectures remain workload dependent.
Cross-reference: Section $`20.1`$.
#### BM25
What: The standard lexical ranking function in the source.
Why: It combines inverse document frequency weighting with saturating, length-normalized term frequency.
Failure or qualifier: Specify the IDF variant, term-frequency/length normalization, query-term handling and corpus statistics; “BM25” alone does not fix every implementation convention.
Cost or complexity: No numeric cost is stated.
Cross-reference: Section A.1.
#### Calibration
What: Agreement between stated model confidence and observed accuracy.
Why: A calibrated confidence value matches empirical outcomes.
Failure or qualifier: Discrimination ranks failure risk, whereas calibration concerns probability values. A monotone recalibration can preserve discrimination while changing calibration; neither alone guarantees safe selective answering.
Cost or complexity: No separate cost is stated.
Cross-reference: Section $`11.4`$.
#### Chunk
What: The unit stored, embedded, and retrieved.
Why: It defines the object that moves through retrieval.
Failure or qualifier: Chunk size and overlap are design parameters, not defaults.
Cost or complexity: No fixed size or overlap is stated here.
Cross-reference: Chapter 13.
#### CLIP
What: Contrastive Language-Image Pre-training. An image encoder and a text encoder train together so matching image-text pairs land near each other in one shared space.
Why: It is the default multimodal retriever in the source.
Failure or qualifier: Chapter 39 examines three named bias mechanisms in its setting, not an exhaustive list or a guarantee that every CLIP-derived model shares identical failures.
Cost or complexity: No numeric compute or storage cost is stated here.
Cross-reference: Section $`39.3`$.
#### Closed-book
What: Answering from parametric knowledge with no retrieved context.
Why: It supplies the baseline every RAG evaluation needs.
Failure or qualifier: Record the baseline explicitly. Its absence limits claims of retrieval benefit; this glossary does not establish how often published evaluations omit it.
Cost or complexity: No separate cost is stated.
Cross-reference: Section $`30.2`$.
#### ColBERT
What: ColBERT-style late interaction retains contextual token representations and sums, over query tokens, their maximum similarity to document-token vectors under the configured masking/normalization.
Why: It permits fine-grained matching while precomputing document representations; accuracy relative to a cross-encoder must be measured on the target task.
Failure or qualifier: The recovery comes with index-time cost.
Cost or complexity: One vector per token increases index work. The appendix gives no numeric factor.
Cross-reference: Section $`21.3`$.
#### Cross-encoder
What: A model that scores a query and document jointly in one forward pass.
Why: Joint query-document encoding permits cross-attention that can improve pair scoring; architectural interaction alone does not guarantee higher accuracy.
Failure or qualifier: A general cross-encoder score cannot be reconstructed simply from one query-independent document vector. It is commonly used after candidate retrieval, but caching, factorization and other architectures have different trade-offs.
Cost or complexity: It uses one forward pass for each scored query-document pair. No numeric cost is stated here.
Cross-reference: Chapter 22.
#### Datastore
What: The non-parametric component made of retrievable text plus its index. The source writes it as $`D`$.
Why: It holds knowledge outside model weights.
Failure or qualifier: A datastore may be one store or a governed union; enumerate contents, indexes, payloads, metadata and permissions rather than assuming one deployment topology.
Cost or complexity: No numeric storage cost is stated here.
Cross-reference: Section $`1.1`$.
#### Faithfulness
What: Whether an answer follows from retrieved context.
Why: It tests support by the evidence given to the generator.
Failure or qualifier: It is not the same as factuality.
Cost or complexity: No separate cost is stated.
Cross-reference: Section $`33.1`$.
#### Generative retrieval
What: A model generates document identifiers, often with constrained decoding, rather than using the usual dense or lexical candidate-search path. It still needs identifier validity structures and access to retrievable evidence.
Why: Retrieval becomes direct identifier generation.
Failure or qualifier: Retrieval quality, corpus updates and identifier design are empirical/design challenges, not a proved universal quality ceiling or an assertion that no incremental method can exist.
Cost or complexity: Decoding cost depends on model size, identifier length, beam width, trie/validity checks and corpus updates. Avoiding an explicit full-corpus score scan does not establish constant total cost independent of corpus size.[\[3\]](../part-06-retrieval-ranking/23_generative_retrieval.md)
Cross-reference: Chapter 23.
#### Graph RAG
What: A RAG design uses extracted entities, relations and often community summaries alongside source chunks. Graph and flat evidence can coexist rather than being exclusive alternatives.
Why: It serves relational and global-sense-making queries.
Failure or qualifier: Relational or global queries are promising use cases, but extraction, resolution, update and serving costs must be justified by measured downstream gains. No query-family label guarantees payoff.
Cost or complexity: The appendix names a cost but gives no numeric amount.
Cross-reference: Chapter 40.
#### Hallucination
What: A generated statement unsupported by the context or the world.
Why: The term names unsupported generation.
Failure or qualifier: The useful taxonomy separates factuality from faithfulness and intrinsic from extrinsic.
Cost or complexity: No separate cost is stated.
Cross-reference: Section $`4.4`$.
#### Hard negative
What: A non-relevant document that scores highly.
Why: Training on it sharpens the decision boundary.
Failure or qualifier: Mining can strongly affect quality, but its dominance is empirical rather than universal. Audit false negatives, exposure bias and training/evaluation leakage.
Cost or complexity: No numeric mining cost is stated here.
Cross-reference: Section $`20.3`$.
#### HBM
What: High-bandwidth memory attached to an accelerator.
Why: Low-batch dense-model decoding may be dominated by transferring model weights and KV state from accelerator memory. It does not read the entire installed HBM capacity on every token.
Failure or qualifier: Bandwidth, compute, batching, active parameters, caching, context length and parallelism determine the bottleneck. Decode is not universally bandwidth-bound.
Cost or complexity: Estimate actual bytes transferred per step and effective bandwidth. Weight payload is one term; KV traffic, cache reuse and hardware/runtime overhead can matter.[\[4\]](../part-12-systems-design/37_latency_cost_and_systems.md)
Cross-reference: Section $`37.1`$.
#### HNSW
What: A multilayer proximity-graph ANN index with probabilistic levels and greedy routing, commonly explained by analogy to skip lists. It is not a literal ordered skip-list conversion.
Why: The layered graph supports fast search.
Failure or qualifier: Links, IDs, level storage and allocation can be substantial. Link bytes do not shrink automatically with vector quantization, but topology and encodings can change them; they are not inherently incompressible.[\[5\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)
Cost or complexity: The source states memory pressure but gives no numeric amount here.
Cross-reference: Section $`15.2`$.
#### HyDE
What: Generate a hypothetical document/answer for a query, embed that representation and use it to retrieve real evidence. The generated text is not itself verified evidence.
Why: The generated answer moves the query into document space.
Failure or qualifier: The embedding may emphasize a useful semantic neighborhood or amplify an incorrect assumption. Measure retrieval gains, generation cost and exact-identifier failures.
Cost or complexity: No numeric generation or retrieval cost is stated here.
Cross-reference: Section $`24.6`$.
#### Hybrid retrieval
What: Combine complementary retrieval channels, often lexical and dense, using score/rank fusion or a learned combiner. Parallel execution is common, not mandatory; a shared engine may also implement a hybrid.
Why: The pipeline combines the two retrieval paths.
Failure or qualifier: RRF is one fusion option. Define weights, constants, normalizers, eligibility and missing-list behavior for the actual system.
Cost or complexity: It runs two retrieval paths. No numeric overhead is stated here.
Cross-reference: Section $`21.6`$.
#### In-context learning
What: A model conditions on examples or evidence in the prompt without weight updates.
Why: RAG rides on this mechanism.
Failure or qualifier: The definition excludes weight updates.
Cost or complexity: No numeric prompt cost is stated here.
Cross-reference: Chapter 7.
#### IVF
What: Inverted file index. It partitions the space into Voronoi cells around centroids.
Why: Search visits only the nearest $`n_{\mathrm{probe}}`$ cells.
Failure or qualifier: Voronoi-cell language assumes the specified Euclidean coarse assignment. Other metrics, approximate routing, residual coding and filtering affect which lists are searched and what the recall target means.
Cost or complexity: $`n_{\mathrm{probe}}`$ controls the searched subset. No fixed value is stated here.
Cross-reference: Section $`15.3`$.
#### Lost in the middle
What: An observed degradation when supporting evidence is placed in middle positions relative to matched beginning/end positions for a specified model/task/context setup.
Why: Evidence position can change whether the model uses it.
Failure or qualifier: Evidence in the middle can receive lower effective use than evidence near the ends.
Cost or complexity: No numeric accuracy gap is stated here.
Cross-reference: Chapter 30.
#### LSH
What: Locality-sensitive hashing. Its hash functions collide more often as inputs become more similar.
Why: Retrieval becomes a bucket lookup.
Failure or qualifier: LSH families provide specified near/far collision guarantees; sublinear query bounds require parameter and distribution assumptions, approximation and output costs. Practical memory/speed rankings against graph indexes are benchmark dependent.
Cost or complexity: The source states the theory-practice trade but no numeric memory amount here.
Cross-reference: Section $`15.4`$.
#### Modularity
What: A graph-community objective comparing within-community edge weight with a specified null model, optionally with a resolution parameter. It is one objective, not the definition of every community-detection algorithm.
Why: A GraphRAG implementation may use it to select communities, while other implementations optimize objectives such as CPM or use different grouping rules.
Failure or qualifier: State weighted/directed conventions and resolution. A heuristic result need not globally maximize modularity, and its communities are not guaranteed to match semantic ground truth.[\[6\]](../part-12-systems-design/40_graph_rag.md)
Cost or complexity: No numeric computation cost is stated here.
Cross-reference: Section A.5.
#### Parametric memory
What: Knowledge stored in model weights.
Why: It names the model-side store in the RAG frame.
Failure or qualifier: Weights do not expose a native fact-key delete. Checkpoint rollback, model editing, unlearning and retraining exist with different guarantees and costs; fixed parameter count is not a universal fixed factual-capacity theorem.
Cost or complexity: Update expense is stated without a numeric amount here.
Cross-reference: Section $`1.1`$.
#### Product quantization
What: Split a vector into subvectors and quantize each against its own codebook.
Why: With $`m`$ subquantizers and $`k`$ centroids per subspace, the Cartesian product has at most $`k^m`$ reconstructed combinations from $`km`$ subspace centroids. Duplicate centroids can reduce distinct vectors.
Failure or qualifier: For equal subspaces, require $`m\mid d`$. The $`km`$ centroids have dimension $`d/m`$ each, so their total is $`kd`$ scalar values, not $`kmd`$. Quantization error does not by itself determine ranking recall.
Cost or complexity: Centroid storage is $`kd\,b_{\mathrm{value}}`$ bytes. Fixed-width subcodes use $`m\lceil\log_2 k\rceil`$ bits before byte packing/alignment, plus IDs and applicable index/runtime terms.[\[7\]](../part-05-indexing/16_compression_and_index_economics.md)
Cross-reference: Section A.6.
#### Prompt injection, indirect
What: An adversary places instructions in an untrusted external data channel, such as retrieved text or a tool result, to redirect a system beyond the user-authorized task.
Why: It identifies a trust-boundary failure; retrieval is a common entry point but not the only indirect data channel.
Failure or qualifier: It is a major RAG threat model, not the only security risk. Instruction/data formatting alone does not establish immunity.
Cost or complexity: No attack rate or defense cost is stated here.
Cross-reference: Section $`36.4`$.
#### $`\operatorname{Recall}@k`$
What: Relevant items returned within top $`k`$, divided by all relevant eligible items under the stated judgments. Record macro/micro aggregation and the zero-relevant-item policy.
Why: It measures coverage at a chosen retrieval depth.
Failure or qualifier: ANN recall instead compares recovery of an exact-neighbor reference under the same metric, corpus and eligibility. The word recall alone does not determine which denominator is intended.
Cost or complexity: The source gives no fixed $`k`$ in this definition.
Cross-reference: Chapter 32.
#### Reranking
What: A second scoring stage over the first stage’s candidates.
Why: A stronger second-stage scorer can improve ordering within a candidate set; it cannot guarantee an accuracy gain or recover omitted evidence without another retrieval action.
Failure or qualifier: It does not replace first-stage candidate generation.
Cost or complexity: It trades compute for accuracy. No numeric budget is stated here.
Cross-reference: Chapter 22.
#### Retrieval gate
What: The function $`g(x)`$ that decides whether to retrieve at all.
Why: It controls decision one in the retrieval flow.
Failure or qualifier: The definition covers whether to retrieve, while routing can also cover where.
Cost or complexity: No threshold or gate cost is stated here.
Cross-reference: Section $`25.1`$.
#### RRF
What: In unweighted RRF, $`S(d)=\sum_{\ell:d\in L_\ell}1/(c+r_\ell(d))`$, with one-based ranks, a stated $`c\ge0`$ and zero contribution when absent. This constant is not context depth $`k`$.[\[8\]](../part-06-retrieval-ranking/21_learned_sparse_and_multi_vector_retrieval.md)
Why: It fuses rankings whose raw scores are not comparable.
Failure or qualifier: It uses ranks rather than directly comparable raw scores; weights, duplicate policies, tie handling and truncated-list support still matter.
Cost or complexity: The appendix states the sum but no list-count limit.
Cross-reference: Section A.2.
#### Self-RAG
What: Train a generator to emit reflection tokens that control retrieval and critique its output.
Why: The generator learns when to retrieve and how to assess its response.
Failure or qualifier: Control depends on emitted reflection tokens.
Cost or complexity: No numeric training or serving cost is stated here.
Cross-reference: Section $`26.6`$.
#### Semi-parametric
What: A system that combines parametric weights with a non-parametric datastore.
Why: This combination frames every RAG design question in the source.
Failure or qualifier: Neither store alone represents the full system.
Cost or complexity: No numeric cost is stated here.
Cross-reference: Section $`1.1`$.
#### SPLADE
What: A learned sparse family that commonly uses a masked-language-model vocabulary head, activation and pooling to produce term weights; variants can encode queries and documents differently.
Why: It can reweight literal terms and activate absent vocabulary terms while using sparse retrieval structures; expansion is learned, not a guarantee of relevance.
Failure or qualifier: The prediction remains vocabulary-based and sparse.
Cost or complexity: No numeric expansion or index cost is stated here.
Cross-reference: Section $`21.1`$.
#### Tombstone
What: A logical deletion marker checked while reading or searching an index. It need not physically reclaim the vector or every derived copy.
Why: It records a deletion inside the index.
Failure or qualifier: Effects on recall, service work and latency depend on implementation and deletion pattern. Some systems compact, repair or physically delete; a full rebuild is not universally forced at a fixed fraction.
Cost or complexity: Measure deletion visibility, replica/cache enforcement, retained bytes and maintenance cost separately; no universal rebuild threshold follows from the term.[\[9\]](../part-05-indexing/17_operating_a_vector_store.md)
Cross-reference: Section $`17.1`$.
#### Vocabulary mismatch
What: The gap between the words a user types and the words a document uses.
Why: A relevant document can express the idea with different vocabulary.
Failure or qualifier: Query expansion, document expansion, and dense retrieval attack the gap differently.
Cost or complexity: No numeric mismatch rate or remedy cost is stated here.
Cross-reference: Section $`24.1`$.
## Diagrams
The manifest lists zero figures and zero counted source tables for this appendix. The unnumbered G.1 table contains 18 alias rows, all preserved in the Decoder table and Core mechanics.
Source layout defect: The final two alias rows extend below the visible page boundary in the source Portable Document Format (PDF). The PDF content layer retains Late interaction and Prompt injection, so both clipped rows are preserved here. This does not change the manifest’s numbered table count.
## Whiteboard pack
### What to draw
1. Draw one station labeled “RAG vocabulary.”
2. Draw three rail lines entering it. Label them “information retrieval,” “representation learning,” and “systems.”
3. Draw one shared object with several name tags around it.
4. Add a translation board labeled “alias rows plus caveats.”
5. Draw two boxes labeled “faithfulness” and “factuality.” Put “context” under the first and “world” under the second.
6. Draw two boxes labeled “index recall” and “retrieval relevance.” Put “exact search” and “human judgment” beneath them.
7. Draw $`k^{\prime}`$ candidates entering a reranker, then $`k`$ items reaching a generator.
8. Add a handbook arrow labeled “definition plus source pointer.”
### Spoken script
This glossary is a translation board for three literatures that named the same RAG objects differently. First, I map aliases but keep the caveats. A reranker is a role, while a cross-encoder is one way to perform it. Then I protect three distinctions. Faithfulness asks whether an answer follows the context, while factuality asks whether it is true. Index recall compares approximate with exact search, while relevance uses human judgment. Finally, $`k^{\prime}`$ feeds the reranker and $`k`$ reaches the generator. Each glossary entry states this book’s convention and points to the section with the full argument.
## Interview traps
### 1. Are all terms on an alias row perfect synonyms?
No. The rows identify one broad idea, but the caveats preserve scope. DPR is one bi-encoder instance, a reranker is a role rather than a cross-encoder synonym, and semantic search can name the whole product pipeline.
### 2. Can an answer be faithful and false?
Yes. Faithfulness asks whether the answer follows the retrieved context. If the retrieved document is wrong, an answer can follow it perfectly and still be non-factual about the world.
### 3. Does index recall of $`0.99`$ prove relevant retrieval?
No. Index recall compares approximate search with exact search under the same scorer. A bad scorer can retrieve the wrong documents very reliably, while retrieval relevance still fails against human judgment.
### 4. Why does the source separate $`k^{\prime}`$ from $`k`$?
$`k^{\prime}`$ is the first-stage depth sent to the reranker. $`k`$ is the count sent to the generator. They serve different objectives, so one shared depth hides a design decision.
### 5. When should you reject a field-wide average definition?
State the local convention and competing meanings, but do not use a book convention to override mathematical scope or a library’s actual behavior. In particular, distinguish evidence support from truth, exact-neighbor recovery from judged recall, and candidate depth from final context depth.
## Key numbers


| Source item | Number or formula | Meaning and claim limit |
| --- | --- | --- |
| Introduction obstacles | 2 obstacles, only 1 conceptual | Aliasing is conceptual. Point-of-need definition is ordinary navigation |
| Naming traditions | 3 literatures | Information retrieval, representation learning, and systems developed separate vocabularies |
| G.1 title | 4 names | The title emphasizes several names for one idea. It does not claim every row has exactly four aliases |
| G.1 alias inventory | 18 rows, final 2 clipped | The PDF content layer retains the two rows that extend below the visible page boundary |
| Explicit distinction tests | 3 | Faithfulness versus factuality, recall versus relevance, and $`k`$ versus $`k^{\prime}`$ |
| Retrieval depths | $`k^{\prime}`$ to the reranker, $`k`$ to the generator | The two depths serve separate objectives |
| Bad-scorer example | Index recall $`0.99`$ | High agreement with exact search does not establish human relevance |
| Abstention | Five illustrative families in Chapter 11 | Not an exhaustive mathematical count of all refusal policies |
| Agentic RAG | $`\sum_{t=1}^{R}C_t`$ | Random rounds with variable work; expectation and tail both matter |
| ANN search | Approximation/latency trade-off | Billion-scale feasibility depends on layout, hardware and required recall |
| CLIP | Three named bias mechanisms in the chapter | Illustrative analysis, not an exhaustive universal property |
| ColBERT | Token/patch-level vectors under configured tokenization | Variable vector count and compression; no guaranteed cross-encoder-equivalent accuracy |
| Cross-encoder | Joint pair encoding | Common second-stage scorer; cost and indexability depend on architecture, pair length and caching |
| Generative retrieval | Identifier decoding and update costs | No universal constant-time or fixed-quality-ceiling guarantee |
| High-bandwidth memory | Active bytes per decode step | Not the full installed memory on every token; bottleneck depends on batching/cache/context |
| Hybrid retrieval | Two or more complementary channels | Parallelism and RRF are options, not defining requirements |
| Inverted file index | $`n_{\mathrm{probe}}`$ nearest cells | The appendix gives no fixed probe count |
| LSH | Conditional sublinear query bounds | State approximation, parameter, output and workload assumptions |
| Product quantization | At most $`k^m`$ combinations; $`kd`$ scalars | Count subspace centroid dimensions, code bits, packing, IDs and runtime separately |
| $`\operatorname{Recall}@k`$ | Top $`k`$ | Index and retrieval recall use different reference standards |
| Reranking | Candidate-only scoring | Additional compute may improve order but does not guarantee accuracy or missing evidence recovery |
| Retrieval gate | $`g(x)`$, decision 1 | The function decides whether to retrieve at all |
| RRF | $`\sum_{\ell:d\in L_\ell}1/(c+r_\ell(d))`$ | Ranks start at one; absent lists contribute zero; state constant/weights/ties |
| Semi-parametric system | 2 stores | Parametric weights and a non-parametric datastore form the complete system |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A6\_notation\_quick\_reference](A6_notation_quick_reference.md)
