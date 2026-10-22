# 00.03 Preface
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_02\_contents](00_02_contents.md) · [00\_04\_how\_to\_use\_this\_book](00_04_how_to_use_this_book.md)
This preface explains how the book turns Retrieval-Augmented Generation (RAG) from a component list into a production reasoning discipline for artificial intelligence systems.
## TL;DR
- Treat RAG as a conditional design choice, not an automatic improvement or an empirically established most-common pattern.
- Retrieval can help or damage an answer; retain each cited study’s model, task, baseline and denominator.
- Diagnose components and interactions using versioned traces and controlled tests; a final score does not identify one unique cause.
- The contents list twelve parts, forty-one chapters and 247 numbered main-chapter subsections. These counts do not prove every revised section has an identical teaching layout.
- Re-derive numbers from units, assumptions and scope. Payload formulas are not complete memory budgets, and arithmetic cannot validate empirical or legal assumptions by itself.
## The story
Imagine a hospital where a physician answers questions from memory and a records clerk fetches patient charts. The physician is the language model. The records clerk is the retriever. Together they form the RAG team.
A component-only description retrieves the top $`k`$ results and passes them to a generator. A design must also ask whether evidence is needed, which sources are eligible, and whether answering without retrieval is permitted by the product’s support and safety requirements.
In the cited Mallen et al. setting, retrieval changed roughly $`10\%`$ of otherwise-correct answers into wrong ones. That denominator is the correct closed-book subset, not every user request, and the rate is not universal. Distracting evidence can hurt, while suitable evidence can help; measure the actual model, corpus and task.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
The hospital must diagnose failures across the full care path. A quality drop can begin with the question, the chart search, the ranking, the packet given to the physician, or the final answer. Knowing that the team uses retrieval does not reveal which stage caused an end-to-end accuracy regression, which means a decline in the final measured answer quality.
The relevant literature spans retrieval, language modeling, databases and system engineering. Individual papers and tools can cross these boundaries; no single source should be assumed either complete or incapable of discussing neighboring stages.
The source contents enumerate twelve parts, forty-one chapters and 247 main-chapter subsections. The book advocates first-principles explanation, worked arithmetic, practical choices and interview practice. This is a teaching template, not a verified claim that every revised Notion section contains identical blocks.
The residency assumes that the trainee can already code and knows basic machine learning. It does not teach deep-learning foundations, transformers, or backpropagation. It is not a complete retrieval survey. It also avoids training the reader on one vendor, framework, embedding application programming interface (API), or company interview process because those details can change every quarter.
The rotations follow the whole patient journey. They begin with what the physician already memorized and when that memory fails. They cover representation, indexing, ranking, question routing, repeated retrieval, training, context assembly, citation, evaluation, credibility, security, cost, and scale. The final rotation uses end-to-end design drills, including the hard case where the system became worse without an obvious cause.
The HNSW example illustrates a conditional memory balance: vector-only quantization shrinks vector payload while leaving an unchanged graph allocation fixed. Graph links can therefore dominate that subtotal. They are not inherently incompressible; topology, ID encoding and graph representation can also change storage.[\[2\]](../part-05-indexing/16_compression_and_index_economics.md)
A trainee who memorizes that one archive needs 32 gigabytes knows one case. A trainee who understands the memory formula can recompute the answer when vector dimension or compression changes. The book trains that second habit.
The residency also has a strict evidence rule. It attributes numbers to named sources or derives them in view. It omits numbers that it cannot attribute or derive. It starts rounds with cases that challenge familiar advice, such as retrieval losing to a closed-book answer on popular facts or citations failing to prove that an answer is grounded in its evidence.
The hospital metaphor ends with one operating principle. A fluent description of the staff is not enough. A strong engineer decides when each staff member should act, measures what happened, finds the failing stage, and recomputes the plan when the constraints change.
## Decoder table


| Technical term | Plain-English meaning | Why it matters |
| --- | --- | --- |
| Retrieval-Augmented Generation (RAG) | A system that fetches external material and gives it to a language model before the model answers | The book studies when this pattern helps, when it hurts, and how to design it |
| Machine learning | Systems that learn patterns from data | The book assumes the reader already knows its basics |
| Deep learning | The model family named as assumed background | The book does not teach it from the beginning |
| Transformer | A language-model architecture named as assumed background | The book does not explain it |
| Backpropagation | A model-training method named as assumed background | The book does not teach it |
| Language model | The component that produces the final text answer | Retrieval only helps if the model uses the added material well |
| Model memory | Knowledge reflected in the model before external retrieval | It affects the decision to retrieve or answer closed-book |
| Retriever | The component that searches for material related to a query | Its choices determine what evidence reaches the model |
| Embedding | A numeric representation of text | It lets a system compare queries and documents by location in a numeric space |
| Nearest neighbor | A stored vector judged most similar to a query vector | Nearest-neighbor search is the common retrieval step named in weak RAG descriptions |
| Top $`k`$ | Highest-ranked results under a specified scorer | Candidate depth and final context depth can differ; actual token count is separate |
| Prompt | The instructions and context given to the language model | Retrieved documents influence the answer through the prompt |
| Context | The material available to the model while it creates an answer | Added context can help, distract, or increase unsupported output |
| Hallucination | A defined form of unsupported or incorrect content | Specify support against evidence versus world truth; they are different constructs |
| Closed-book answer | An answer produced from the model’s stored knowledge without retrieval | It can outperform retrieval for some high-popularity facts |
| Corpus | The collection of documents available for search | Changes to the corpus can degrade an index over time |
| Corpus churn | Documents being added, removed, or changed | Churn can weaken graph-based search behavior |
| Hierarchical Navigable Small World (HNSW) index | A graph-based structure for approximate nearest-neighbor search | Its graph can become the main memory cost after vector compression |
| Approximate nearest-neighbor index | A search structure that seeks likely close vectors instead of an exact full comparison | The book covers how these indexes work and what they cost in memory |
| HNSW graph | Stored navigation links and level structures | Vector-only quantization leaves unchanged links fixed, not inherently incompressible |
| Quantization | A way to store vectors with fewer bytes | It reduces vector memory and can expose the graph as the remaining cost |
| Product quantization | A compression method that stores a vector as a short code | The preface uses a 64-byte code to show how the memory balance changes |
| fp32 | $`4\,\mathrm{B/dimension}`$ | Contributes $`4d`$ bytes of vector payload only |
| Vector dimension | The number of numeric coordinates in an embedding | Increasing dimension raises vector memory directly |
| End-to-end accuracy | Final answer accuracy across the complete pipeline | A drop at the system output does not identify which stage failed |
| Regression | A measured decline in system performance | Engineers must locate the stage that owns it |
| Five-stage pipeline | An illustrative decomposition of a production path | Interactions and simultaneous changes may prevent a unique owning-stage diagnosis |
| Five-decision skeleton | The structure used for design answers throughout the book | It forces a candidate to state decisions rather than list components |
| Information retrieval | The field that studies search, ranking, and retrieval evaluation | Its depth does not by itself explain how a generator uses retrieved text |
| Generator | The component that writes the answer | Generator behavior must be studied together with retrieval behavior |
| Index | The data structure used to find documents quickly | Treating it as a black box hides memory and quality trade-offs |
| Black box | A component whose internal choices are hidden from the surrounding explanation | It prevents end-to-end reasoning about the index or generator |
| Vector database | A system that stores and searches vector representations | Product documentation explains controls but not the whole RAG decision surface |
| Orchestration framework | Software that connects pipeline components | A short framework demo does not explain production-scale failure |
| Embedding application programming interface (API) | A software interface that produces embeddings | The book avoids tying its arithmetic to one changing product interface |
| Trade-off surface | The set of gains and losses created by a design choice | Knowing controls is not enough without knowing what each control sacrifices |
| Continual learning | Updating a model as new data arrives | The book compares it with retrieval as a way to add or refresh knowledge |
| Model editing | Changing selected knowledge inside a model | It is another competitor to retrieval |
| Long context | Giving the model a large amount of material directly | It can replace retrieval in some designs and creates its own costs |
| Generative retrieval | Using generation as part of identifying what to retrieve | It appears both as a retrieval alternative and as a later ranking approach |
| In-context learning | A model adapting its response from examples or information in its current prompt | Retrieval depends on this mechanism to influence generation |
| Parameter | A learned internal value in a model | The book compares the economics of model parameters with an external datastore |
| Datastore | The external collection that a retrieval system searches | It offers a different place to hold information than model parameters |
| Training data | The examples used to fit a model or retrieval system | The book covers its privacy, legal, and manufacturing questions |
| Confabulation | Producing plausible content that is not supported | The generator chapters contrast it with what the model memorized |
| Prompting | Choosing instructions and context for a model | It shapes how the generator responds to retrieved material |
| Sensitivity | The degree to which an answer changes when its input changes | It is part of the generator behavior covered by the book |
| Representation | The form used to encode text for search | Poor representation weakens every later retrieval stage |
| Chunking | Splitting documents into searchable pieces | Chunk boundaries affect what a retriever can return |
| Semantic chunking | Meaning-related boundary selection | Gains can fail to transfer; compare matched real-document baselines |
| Ranking | Ordering candidate documents by expected usefulness | The model often sees only the highest-ranked candidates |
| Best Matching 25 (BM25) | A term-based ranking method | It begins the ranking progression described in the book |
| Learning to rank | Training a model to order search results | It moves ranking from fixed formulas toward learned decisions |
| Dense bi-encoder | A retrieval model that separately encodes queries and documents as dense vectors | It is one step in the ranking progression described in the preface |
| Learned sparse retrieval | A learned retrieval method that uses sparse representations | It is one step in the ranking progression described in the preface |
| Multi-vector retrieval | Representing an item with more than one vector | It is one step in the ranking progression described in the preface |
| Reranking | Reordering an initial candidate set | It is one step in the ranking progression described in the preface |
| Query understanding | Interpreting what a question asks before retrieval | It shapes reformulation, routing, and control flow |
| Query reformulation | Rewriting a question to make search work better | A weak query can cause failure before document ranking begins |
| Routing | Choosing which retrieval path or resource should handle a query | Routing controls whether and where retrieval happens |
| Control flow | The order, branches, and repeated steps in a pipeline | It controls how many times the system retrieves |
| Iterative retrieval | Searching more than once while building an answer | Some questions need evidence gathered in several steps |
| Self-reflective loop | A control loop in which the system judges its own need for more work | It decides whether another retrieval step is needed |
| Agentic loop | A control loop in which a model selects and repeats actions | It can decide how many times to retrieve |
| Learned controller | A trained routing/gating policy | Shift can change calibration or action quality; degradation is empirical, not universal |
| Query distribution shift | Change in the request distribution | Monitor policy-specific outcomes and evidence needs |
| Context assembly | Selecting and arranging retrieved material for the model | Successful search can still fail if the final context is poorly built |
| Context window | The text positions available to the model for the current answer | The book covers positional bias inside this window |
| Positional bias | The model treating information differently because of where it appears in the context | Evidence placement can change how well retrieved content is used |
| Attribution | Connecting an answer claim to its supporting source | It helps users see where answer content came from |
| Citation | A reference to a source attached to an answer | A citation alone does not prove that the answer is grounded |
| Post-hoc citation | A source reference attached after generation | May support a claim but does not by itself prove historical model use |
| Grounding | Making answer claims depend on supplied evidence | It is a stronger requirement than merely displaying citations |
| Retrieval evaluation | Measuring whether search returned useful material | It tests one layer of the system |
| Generation evaluation | Measuring the quality of the produced answer | Good retrieval does not guarantee good generation |
| System evaluation | Measuring the combined behavior that users experience | The whole system is not simply the sum of separate retrieval and generation scores |
| Source credibility | The trustworthiness of retrieved material | Unreliable evidence can produce unreliable answers |
| Provenance | A record of where data or a claim came from | It supports traceability and trust |
| Prompt injection | Malicious instructions placed in content that reaches the model | Retrieved text can attack the system through its prompt |
| Datastore poisoning | Corrupting the collection that retrieval searches | Poisoned material can steer later answers |
| Memory bandwidth | The rate at which stored data can be moved for computation | The book treats it as a durable systems constraint |
| High-dimensional geometry | The behavior of vectors with many coordinates | It shapes the limits and choices of vector retrieval |
| Token economics | The cost consequences of processing pieces of text | Context and generation choices affect operating cost |
| Abstention | Choosing not to answer when confidence or evidence is inadequate | A system needs a safe option when retrieval or generation is unreliable |


## Core mechanics
### 1. The real first decision
- What it is: Decide whether and how to retrieve under the task’s evidence requirements.
- Why it exists: External material can help or distract; a model’s apparent confidence is not sufficient proof of truth.
- Failure without it: The system assumes every retrieval is beneficial or bypasses evidence when citations/current facts are mandatory.
- Evidence: The roughly ten-percent harm figure is conditional on the cited study’s otherwise-correct subset, not a universal production rate.[\[1\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
The common recipe embeds documents, searches for nearest neighbors, and places the top k results in the prompt. That recipe names components. It does not state a decision rule.
The source presents retrieval as common and component-only explanations as insufficient. It supplies no population study establishing that almost every language-model product includes retrieval or a universal date when interview expectations changed.
Production interviews probe this gap. They ask when not to retrieve. They also ask why more documents can increase hallucination.
### 2. Diagnose the full pipeline
- What it is: Trace a regression through the question, evidence, ranking/assembly and answer path under the actual architecture.
- Why it exists: End-to-end accuracy reveals an outcome, not a unique causal decomposition.
- Failure without it: The team changes a familiar stage without testing rival explanations.
- Example scope: The five-stage decomposition is illustrative; a four-percentage-point accuracy decline is not four percent relative decline. Several stages or their interactions may change.[\[3\]](../part-10-evaluation/34_evaluating_the_system.md)
Corpus churn can affect graph search, maintenance work and eligibility, depending on the implementation and deletion pattern. Vector compression can expose unchanged link allocation as a larger share of storage; it does not prove the graph is an incompressible term or that every deletion requires a full rebuild.[\[4\]](../part-05-indexing/17_operating_a_vector_store.md)
The source does not call these exotic questions. It frames them as ordinary Tuesday on-call work when answer quality falls four points overnight.
### 3. Join the separated literatures
- What it is: Treat model memory, retrieval, generation, context, evaluation, trust, and cost as one path.
- Why it exists: Each existing source family explains only part of that path.
- What breaks without it: Engineers optimize one component while treating neighboring components as black boxes.
- Scale stated in the source: Information retrieval contributes 50 years of depth. A framework demo may fit in 20 lines but fail to explain behavior at 10 million documents.
Retrieval has a long history in ranking and evaluation; modern work also studies generation and end-to-end systems. Language-model papers, database documentation and framework tutorials cover different scopes. Treat the source’s fifty-year, twenty-line and ten-million-document examples as illustrations, not exact limits on a literature or a demonstrated failure threshold.
### 4. Use a fixed learning skeleton
- What it is: Teach each idea through first principles, evidence, a numeric example, a production decision, and interview framing.
- Why it exists: The repeated structure makes the material useful for both study and fast reference.
- What breaks without it: A reader may memorize vocabulary without gaining the flexibility to reason under changed constraints.
- Scale stated in the source: The skeleton repeats across 247 sections and frames questions at three seniority levels.
The intended reader already knows how to code and understands basic machine learning. The book can be read from start to finish or opened at one relevant page.
### 5. Keep the boundary explicit
- What it is: Define the book’s scope rather than teach every foundation or product.
- Why it exists: The focus is reasoning about RAG under stated constraints.
- Failure without it: A reader treats a worked example as a vendor benchmark or universal deployment prescription.
- Change boundary: Algebraic identities remain valid under their definitions, but architectural coefficients, hardware throughput, pricing and workload assumptions must be refreshed; there is no fixed quarterly change law.
The book does not explain transformers or teach backpropagation. It does not cover the entire retrieval literature. It does not target one vector database, orchestration framework, embedding API, or employer.
Its generic reasoning rests on constraints that remain relevant across products. The preface names memory bandwidth, high-dimensional geometry, and token economics.
### 6. Cover the complete RAG path
- What it is: Organize the subject into 12 parts and 41 chapters.
- Why it exists: A design answer must connect model behavior, retrieval behavior, control, evaluation, trust, and scale.
- What breaks without it: A candidate can discuss a local component but cannot reason about the system around it.
- Scale stated in the source: Part XII closes with seven end-to-end design drills.
Part I maps the interview and compares retrieval with continual learning, model editing, long context, and generative retrieval. It introduces the five-decision skeleton for later design answers.
Parts II and III cover the generator. Topics include memorized knowledge, confabulation, training-data privacy and law, parameter versus datastore economics, in-context learning, prompting, sensitivity, and abstention.
Parts IV through VI build the retrieval stack. They cover representation, chunking, approximate nearest-neighbor indexes, index memory, and ranking. The ranking path runs from BM25 through learning to rank, dense bi-encoders, learned sparse retrieval, multi-vector retrieval, reranking, and generative retrieval.
Part VII covers query understanding and control flow. It includes reformulation, routing, iterative retrieval, self-reflective loops, and agentic loops that choose how many times to retrieve.
Part VIII trains both retrieval and generation and creates training data. Parts IX and X cover positional bias, attribution, citation, retrieval evaluation, generation evaluation, and full-system evaluation.
Part XI covers credibility, provenance, injection and poisoning. Part XII includes systems topics and seven end-to-end design drills, one of which diagnoses an unexplained quality decline. Their interview frequency is not established by this preface.
### 7. Derive the memory answer
Use decimal storage units: $`1\,\mathrm{GB}=10^9\,\mathrm B`$. Let $`N_v`$ be vector count, $`d`$ dimension and $`M`$ the common HNSW graph parameter. For four-byte link IDs, allocated base-layer capacity $`2M`$ costs $`8M`$ bytes per vector. The subtotal $`B_{\mathrm{sub}}=N_v(b_{\mathrm{vec}}+8M)`$ includes vector/code payload and base-layer links only. It excludes upper layers, labels/IDs, codebooks, source text, metadata, allocation, replicas and runtime; it is not process memory or an all-in capacity plan.[\[5\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[2\]](../part-05-indexing/16_compression_and_index_economics.md)
For $`N_v=10^7`$, $`d=768`$ and $`M=16`$, fp32 vector payload is $`4d=3{,}072\,\mathrm B`$ and base links are $`8M=128\,\mathrm B`$. The scoped subtotal is $`3{,}200\,\mathrm{B/vector}`$ and $`B_{\mathrm{sub}}=32\,\mathrm{GB}`$.
At $`d=1{,}536`$ and unchanged $`M=16`$, fp32 payload is $`6{,}144\,\mathrm B`$. With the same $`128\,\mathrm B`$ link allocation, the subtotal is $`6{,}272\,\mathrm{B/vector}`$ and $`62.72\,\mathrm{GB}\approx62.7\,\mathrm{GB}`$ for $`10^7`$ vectors. The vector term exactly doubles; the combined subtotal increases by slightly less than a factor of two.
Replacing vector payload with a $`64\,\mathrm B`$ PQ code and holding the same base-link allocation gives $`64+128=192\,\mathrm{B/vector}`$, or $`1.92\,\mathrm{GB}`$ for $`10^7`$ vectors. Base links are $`128/192=2/3\approx66.67\%`$ of this subtotal only. Other index terms still need separate budgets; vector quantization alone does not compress unchanged links.
There are three worked cases: one baseline and two changed configurations. Recompute the scoped subtotal each time, then add omitted components before using it to choose machines.
### 8. Enforce an evidence rule
- What it is: Attribute each number to a named source or derive it in the prose.
- Why it exists: Evidence makes a surprising claim checkable.
- What breaks without it: Folklore and fluent explanation can pass as measured fact.
- Source policy stated in the source: Numbers without attribution or derivation are omitted.
Claims are attributed by author and year where they appear. The annotated reading list in Appendix E collects the sources.
### 9. Lead with contradictions
- What it is: Put measured challenges to common RAG advice near the front.
- Why it exists: Counterexamples reveal whether a candidate understands conditions and trade-offs.
- What breaks without it: The candidate repeats a popular technique without asking whether it pays on the real data.
- Evidence posture stated in the source: The supporting number appears beside a claim that rejects common advice.
The preface highlights conditional counterexamples: retrieval can underperform closed-book, chunking gains can fail to transfer, citations can fail to support claims, and controllers can degrade under shift. Each requires a matched evaluation; none establishes a universal failure rate or proves that valid post-hoc evidential support is impossible.[\[6\]](../part-04-representations/13_chunking_and_granularity.md)[\[7\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)[\[8\]](../part-09-context-assembly/31_attribution_and_citation.md)
The source says these contradictions separate a real answer from a fluent one. It closes with the dateline Los Angeles, California, August 2026.
## Diagrams
The source pages contain no figures or tables. The following source-derived map summarizes the prose coverage and does not replace an original visual.
```plain text
[What the model memorized]
             |
             v
[Representation and chunking]
             |
             v
[Indexing and ranking]
             |
             v
[Query routing and control flow]
             |
             v
[Context assembly and generation]
             |
             v
[Attribution and three-layer evaluation]
             |
             v
[Credibility, security, cost, and scale]
             |
             v
[End-to-end design drills]
```
## Whiteboard pack
### What to draw
1. Draw a box labeled `Question` on the far left.
2. Draw a decision diamond labeled `Retrieve?` to its right.
3. Draw a bypass arrow labeled `No` from the diamond to a box labeled `Model answer`.
4. Draw a downward arrow labeled `Yes` to a box labeled `Represent and search`.
5. Add boxes labeled `Index`, `Rank`, and `Assemble context` in a row after search.
6. Connect `Assemble context` to the same `Model answer` box.
7. Draw three boxes below the pipeline labeled `Retrieval evaluation`, `Generation evaluation`, and `System evaluation`.
8. Add a side box labeled `Trust, security, cost, scale` with arrows into the full pipeline.
9. Write `Derive when constraints change` across the top.
### Spoken script
Retrieval-Augmented Generation is not just embed, search, and put the top results in a prompt. The first decision is whether retrieval should happen at all, because added documents can overturn a correct answer or increase unsupported content. Then I trace the full path from representation and indexing through ranking, context assembly, and generation. I evaluate retrieval, generation, and the combined system separately. I also check credibility, security, cost, and scale. The key interview skill is to derive the answer from current constraints, such as vector dimension and compression, instead of repeating one memorized capacity number.
## Interview traps
### 1. Why is `embed, search, add top k` an incomplete RAG design answer?
It names components but omits the decision that controls whether the system should retrieve. A strong answer starts with when retrieval helps, when it can hurt, and how the system will make or evaluate that choice.
### 2. When would you not retrieve?
Compare matched retrieval/no-retrieval actions under the product’s evidence policy. The cited roughly $`10\%`$ harm figure concerns an otherwise-correct subset in one study; it is not a rule that every popular fact should bypass retrieval. A current or source-backed answer may require retrieval even when the model seems confident.
### 3. Why can adding documents increase hallucination?
Retrieved text becomes context, not guaranteed truth. Distracting or misleading material can steer the generator away from a supported answer, so the system must evaluate evidence selection and answer grounding.
### 4. Why can the HNSW graph dominate after quantization?
Only the vector payload shrinks in this comparison; base links remain $`128\,\mathrm B`$. A $`64\,\mathrm B`$ code gives a $`192\,\mathrm B`$ subtotal with link share $`2/3`$. This excludes other index terms and does not establish that links themselves cannot be compressed or redesigned.
### 5. A RAG system lost four accuracy points overnight. Where do you look first?
Check corpus/ACL freshness, workload composition, labels, versions and stage traces; reproduce the decline on a matched set and intervene on plausible stages. The illustrative five-stage map and three evaluation layers guide diagnosis but cannot guarantee exactly one owning stage.[\[3\]](../part-10-evaluation/34_evaluating_the_system.md)
## Key numbers


| Number | Context | Meaning |
| --- | --- | --- |
| 2023 | Mallen et al. attribution | The year attached to the retrieval-harm result |
| Roughly $`10\%`$ | Cited otherwise-correct subset | Conditional retrieval harm in a reported setting |
| Five stages | Production pipeline | The example asks which stage owns an end-to-end regression |
| Four percentage points | Illustrative absolute accuracy drop | Different from four percent relative decline |
| 50 years | Information-retrieval literature | The preface describes this depth in ranking and evaluation |
| 20 lines | Framework tutorial | A small demonstration can fit in this amount of code without explaining production behavior |
| 10 million documents | Framework scale test | The preface uses this size to expose what a small tutorial does not explain |
| One idea | Per section | Each section develops the reasoning behind one idea |
| Three seniority levels | Interview framing | Each section frames its idea for three levels of interview question |
| 247 numbered subsections | Counted in the source contents | An inventory, not proof of uniform teaching format |
| One page | Reference use | The book can be opened at one relevant page instead of read only in sequence |
| Quarterly change | Source illustration | Not a measured universal update cadence |
| 12 parts | Book structure | The parts span the full RAG path |
| 41 chapters | Book structure | The chapters organize the material inside the 12 parts |
| Five decisions | Design-answer skeleton | Part I introduces this structure for later system designs |
| Three evaluation layers | Retrieval, generation, and system | The combined system requires its own evaluation layer |
| Seven drills | End-to-end design practice | Part XII closes with this many design drills |
| $`N_v(b_{\mathrm{vec}}+8M)`$ | Vector/code plus base-link subtotal | Not all-in HNSW process memory |
| $`d=768`$ | fp32 payload | $`3{,}072\,\mathrm{B/vector}`$ |
| $`M=16`$ | Four-byte allocated base links | $`2M`$ slots give $`128\,\mathrm{B/vector}`$ |
| $`3{,}200\,\mathrm{B/vector}`$ | First scoped subtotal | Vector payload plus base links only |
| $`N_v=10^7`$ | Assumed count | Same count for all three storage cases |
| $`3.2\times10^{10}\,\mathrm B`$ | First scoped subtotal | Decimal byte total |
| $`32\,\mathrm{GB}`$ | First scoped subtotal | Excludes all other index/runtime terms |
| $`d=1{,}536`$ | Second case | Exactly doubles fp32 vector payload |
| $`6{,}272\,\mathrm{B/vector}`$ | Second scoped subtotal | $`6{,}144+128`$ bytes |
| $`62.72\,\mathrm{GB}`$ | Second scoped subtotal | Approximately $`62.7\,\mathrm{GB}`$ after rounding |
| $`64\,\mathrm B`$ | Assumed PQ code | Code only; add IDs, codebooks and other structure |
| $`192\,\mathrm{B/vector}`$ | Compressed scoped subtotal | $`64+128`$ bytes |
| $`1.92\,\mathrm{GB}`$ | Compressed scoped subtotal | Code plus base links for ten million vectors |
| $`2/3\approx66.67\%`$ | Base-link share | Share of the scoped subtotal, not all-in process memory |
| Two changed configurations | After one baseline | Three worked cases are not three changes |
| Three worked cases | Baseline, larger dimension and PQ code | All recomputed under declared storage scope |
| One subtotal formula | Vector/code plus fixed base allocation | Add omitted terms before capacity decisions |
| August 2026 | Preface dateline | The month and year printed at the end of the source unit |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_02\_contents](00_02_contents.md) · [00\_04\_how\_to\_use\_this\_book](00_04_how_to_use_this_book.md)
