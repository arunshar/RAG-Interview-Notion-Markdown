# Appendix F: Notation Quick Reference
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A5\_annotated\_reading\_list](A5_annotated_reading_list.md) · [A7\_glossary](A7_glossary.md)
This appendix provides a compact, calculation-oriented reference for the source’s symbols, overload rules, and conventions while leaving the extended discussion to the front matter.
## TL;DR
- Preserve case, primes, subscripts, function arguments and units.
- This reference writes $`m`$ for PQ subquantizers and $`M`$ for the HNSW neighbor parameter. Libraries may reuse uppercase $`M`$ for PQ; map the API’s meaning explicitly.
- Keep $`R@k`$, $`P@k`$ and $`P_k`$ distinct: recall, precision and segmentation error.
- Write the logarithm base. Use natural logs for nats/log loss and the stated BM25 form, and $`\log_2`$ for DCG and code-bit counts.
- Decimal and binary storage units are different. Vectors are columns unless a derivation explicitly uses row-vector stacking.
- The $`d=768`$, nominal $`110`$-million-parameter encoder is an illustrative comparison setup, not a universal architecture requirement.
## The story
Think of the notation sheet as the control board in a rail yard.
Each symbol is a track label. $`x`$ brings user input into the yard. $`q=\Phi(x)`$ sends a query toward retrieval. $`c_1,\ldots,c_k`$ carries contexts onward, and $`y`$ marks the generated answer at the end of the route.
Subscripts and primes are track markers. $`k`$ and $`k^{\prime}`$ do not stop at the same platform. Uppercase $`D`$, lowercase $`d`$ , and primed $`D^{\prime}`$ also lead to different destinations.
The overload table is the switch chart. It tells the operator whether $`d`$ means a document or dimensionality, whether $`k`$ counts contexts or PQ centroids, and whether $`N`$ counts vectors or model parameters.
The switch chart chooses $`m`$ for PQ here and $`M`$ for HNSW. That is a local notation policy, not a claim about every paper or library. Faiss, for example, also uses uppercase $`M`$ as its PQ subquantizer parameter.[\[1\]](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
The conventions state units, vector orientation and the recurring setup. Logarithm bases must follow the quantity: bits and DCG use base $`2`$, while nats, natural-log BM25 and natural-log losses use base $`e`$. A change of base changes reported numeric values and units.
The board works only when the operator reads the local route before interpreting a reused label.
## Decoder table
The three blocks below preserve the source layout and all 57 primary symbol-meaning entries.
### Block 1: System, retrieval, and index notation


| Symbol | Meaning | Why it matters |
| --- | --- | --- |
| $`\theta`$ | Generator parameters | Locates the trainable or frozen generator state |
| $`D`$ | Datastore | Names the corpus available to retrieval |
| $`D_1,\ldots,D_m`$ | Component stores | Separates federated or multimodal stores |
| $`x`$ | User input | Marks the pipeline input before encoding |
| $`q=\Phi(x)`$ | Query issued to the retriever | Connects user input to retrieval geometry |
| $`g(x)`$ | Retrieval gate with values $`\{0,1\}`$ | Decides whether retrieval runs |
| $`\mathop{\mathrm{RET}}\nolimits`$ | Retriever | Names the evidence-selection component |
| $`\mathop{\mathrm{GEN}}\nolimits_{\theta}`$ | Generator | Names the answer-producing component |
| $`c_1,\ldots,c_k`$ | Retrieved contexts | Identifies evidence passed downstream |
| $`y`$ | Generated answer | Marks the pipeline output |
| $`k`$ | Contexts reaching the generator | Sets the final context depth |
| $`k^{\prime}`$ | Retrieval depth before reranking | Separates candidate depth from final depth |
| $`N`$ | Vectors in the index | Sets index scale |
| $`d`$ | Embedding dimension | Sets vector width and memory cost |
| $`M`$ | HNSW neighbor parameter in this reference | A configured slot/connectivity parameter, not every node’s realized out-degree |
| $`M_0`$ | Base-layer neighbor-slot capacity, often $`2M`$ | A construction/allocation convention, not exact realized degree |
| $`\mathrm{ef}`$ | HNSW search candidate-width parameter | Not necessarily graph hops, examined nodes or hardware parallelism |
| $`n_{\mathrm{probe}}`$ | Inverted File (IVF) cells probed | Trades search breadth for latency |
| $`m`$ | PQ sub-quantizers | Splits a vector for compression |
| $`k`$ | PQ centroids per subspace | Sets the local codebook size |
| $`b`$ | Bits per sub-code when $`k=2^b`$; total PQ bits are $`mb`$ | Declare total packed bytes $`\lceil mb/8\rceil`$ and padding separately |
| $`R@k`$ | Recall at a declared retrieval cutoff | Distinguish judged-relevance recall, ANN-neighbor recall and full evidence-set coverage |


### Block 2: Scoring and evaluation notation


| Symbol | Meaning | Why it matters |
| --- | --- | --- |
| $`s(q,d)`$ | Relevance score | Orders a document for a query |
| $`\mathop{\mathrm{rank}}\nolimits(d)`$ | Position in a ranked list | Connects a score to rank-based metrics |
| $`f`$ | Term frequency | Controls lexical evidence strength |
| $`k_1,b`$ | Best Matching 25 (BM25) saturation and length normalization | Control repetition and document-length effects |
| $`\mathop{\mathrm{avgdl}}\nolimits`$ | Mean document length | Anchors BM25 length normalization |
| $`E_Q,E_P`$ | Query and passage encoders | Map both sides into retrieval space |
| $`L_q,L_p`$ | Query and passage token lengths | Set encoder and interaction cost |
| $`\mathop{\mathrm{MaxSim}}\nolimits`$ | Late-interaction operator | Aggregates token-level similarity |
| $`\mathop{\mathrm{MRR}}\nolimits`$ | Mean reciprocal rank of the first relevant hit | Permits multiple relevant items; zero when none appears within cutoff |
| $`\mathop{\mathrm{DCG}}\nolimits`$ | Discounted Cumulative Gain | Discounts useful results by rank |
| $`\mathop{\mathrm{nDCG}}\nolimits@k`$ | Normalized discounted cumulative gain at a declared cutoff | State gain, judgment universe, tie and zero-IDCG policies |
| $`P@k`$ | Precision at $`k`$ | Measures useful-result density |
| $`P_k`$ | Segmentation error | Measures chunk-boundary error rather than precision |
| $`\mathop{\mathrm{IoU}}\nolimits`$ | Intersection over Union for chunk-span overlap | Measures overlap between predicted and reference spans |
| $`F_1`$ | $`2PR/(P+R)`$ when $`P+R\gt 0`$ | State zero-denominator and macro/micro averaging policies |


### Block 3: Training, serving, and graph notation


| Symbol | Meaning | Why it matters |
| --- | --- | --- |
| $`\mathcal{L}`$ | Training loss | Defines the optimization target |
| $`\tau`$ | Positive softmax temperature $`\tau\gt 0`$ | Relative to fixed scores: smaller sharpens, larger flattens |
| $`D^{\prime}`$ | Retrieved subset in training | Separates training evidence from the full store |
| $`\lambda(d,x)`$ | Normalized document weight for a stated ensemble | Define candidate support and normalization; not every mixture uses this same convention |
| $`P_R`$ | Retriever distribution | Describes retrieval preferences |
| $`Q_{\mathrm{LM}}`$ | Generator preference distribution | Describes generator preferences |
| $`\gamma,\beta`$ | Retriever and generator temperatures | Calibrate the two distributions |
| $`r`$ | Low-Rank Adaptation (LoRA) rank | Sets adapter capacity |
| $`A,B`$ | For $`W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}`$, $`A\in\mathbb R^{r\times d_{\mathrm{in}}}`$, $`B\in\mathbb R^{d_{\mathrm{out}}\times r}`$ | $`W+(\alpha/r)BA`$ is common; factor order follows the chosen orientation |
| $`E_q,E_d`$ | Shared-space embeddings | Enable cross-modal or shared-space scoring |
| $`t_i`$ | Stage latency, distinct from attributable device service time | Parallel stages need a critical path; resource cost needs service/capacity allocation |
| $`C_q`$ | Cost per query | Connects design choices to serving economics |
| $`B`$ | Serving batch size | Controls throughput and memory use |
| $`s`$ | Tokens in context or answer | Sets sequence-dependent serving cost |
| $`L`$ | Transformer layers | Sets model depth and repeated compute |
| $`G`$ | Knowledge graph | Names the graph used for grounded retrieval |
| $`\lvert V\rvert,\lvert E\rvert`$ | Entities and edges | Quantify graph size |
| $`\omega`$ | Graph grounding operator | Maps a query into graph-grounded form |
| $`h`$ | Traversal depth in hops | Bounds graph search and reasoning depth |
| $`\theta_m`$ | Example modality-specific admission threshold | Validate modality/score comparability; no theorem forbids a calibrated global policy |


## Core mechanics
### Purpose and use
The full notation chapter carries the conventions and extended overload discussion. This appendix is the compact version. It gives symbols and meanings for use during a derivation rather than during continuous reading.
Read the exact glyph before assigning meaning. Case distinguishes $`D`$ from $`d`$. A prime distinguishes $`D^{\prime}`$ from $`D`$ and $`k^{\prime}`$ from $`k`$. Subscripts distinguish $`M_0`$, $`P_R`$, $`Q_{\mathrm{LM}}`$, $`\theta_m`$, and related symbols from their base letters.
Function arguments also establish scope. $`s(q,d)`$ is a relevance score, while standalone $`s`$ counts tokens in a context or answer.
The calligraphic symbol $`\mathcal{L}`$ is training loss. Plain $`L`$ counts Transformer layers.
### Overloaded symbols


| Symbol | Meaning A | Meaning B or scope condition |
| --- | --- | --- |
| $`d`$ | A document or chunk | Embedding or hidden dimensionality |
| $`k`$ | Retrieved contexts | PQ centroids per sub-quantizer |
| $`m`$ | PQ sub-quantizers | Datastore count or modality index |
| $`M`$ | HNSW neighbor parameter in this reference | Other notations, including Faiss PQ, reuse $`M`$; never translate by case alone |
| $`b`$ | BM25 length-normalization parameter | Sub-code bits, total code bits and bytes/parameter must be distinguished by scope |
| $`N`$ | Vectors in the index | Model parameter count |
| $`P`$ | Model parameter count | Instance price per hour |
| $`G`$ | Knowledge graph | Use $`\mathop{\mathrm{schema}}\nolimits(G)`$ for its schema where confusion is possible |


The overload table is a scope rule. It does not authorize swapping meanings within one derivation. Use the surrounding topic to select the intended meaning.
### Conventions


| Convention | Rule | Scope condition |
| --- | --- | --- |
| Logarithms | Write $`\ln`$ or $`\log_2`$ as appropriate. | Nats/log loss and displayed BM25 use natural logs; DCG and bit counts use base $`2`$. |
| Storage | Use decimal units. $`1\ \text{gigabyte (GB)}=10^9\ \text{bytes}`$. | This is the default storage convention. |
| Quoted storage source | Preserve binary units. | Apply this only when the source is quoted in binary units and the text says so. |
| Latency | Measure in milliseconds. | Applies when no different unit is written. |
| Throughput | Measure in queries per second. | Applies when no different unit is written. |
| Cost | Measure in United States dollars per query. | Applies when no different unit is written. |
| Vectors and transpose | Column vectors by default; $`\mathsf T`$ is transpose. | A stated row-stacked matrix convention overrides the default; check dimensions. |
| Recurring default | Illustrative $`d=768`$ and nominal $`110\times10^6`$ parameters. | Model parameter count does not by itself determine embedding dimension or FLOPs. |


### Scope-sensitive reading sequence
1. Match exact glyphs, arguments and units.
2. Define overloaded symbols locally and map any library parameter explicitly.
3. Check dimensions and vector orientation before multiplying matrices.
4. State the logarithm base and whether the quantity is in bits or nats.
5. Separate document count, indexed vector count and model parameters.
6. Treat $`d=768`$ and $`110\times10^6`$ parameters as illustrative inputs, not universal facts.
### Distinctions that must survive transcription


| Pair or family | Required distinction |
| --- | --- |
| $`D`$, $`d`$, $`D^{\prime}`$ | Datastore, document or dimension by scope, and retrieved training subset |
| $`k`$, $`k^{\prime}`$, $`k_1`$ | Context count or PQ centroid count, pre-rerank depth, and BM25 saturation |
| $`M`$, $`M_0`$, $`m`$ | HNSW upper/base slot parameters and this reference’s PQ subquantizer count |
| $`b`$ | Distinguish BM25 normalization, bits per sub-code, total code bits and bytes per parameter |
| $`P@k`$, $`P_k`$, $`P_R`$, $`P`$ | Precision at $`k`$, segmentation error, retriever distribution, and an overloaded parameter or price symbol |
| $`R@k`$, $`\mathop{\mathrm{MRR}}\nolimits`$, $`\mathop{\mathrm{DCG}}\nolimits`$, $`\mathop{\mathrm{nDCG}}\nolimits@k`$, $`F_1`$ | Distinct evaluation quantities |
| $`s(q,d)`$, $`s`$ | Relevance score and token count |
| $`\mathcal{L}`$, $`L`$, $`L_q`$, $`L_p`$ | Training loss, Transformer layer count, and query or passage token lengths |
| $`E_Q`$, $`E_P`$, $`E_q`$, $`E_d`$ | Query and passage encoders versus shared-space embeddings |
| $`G`$, $`\lvert V\rvert`$, $`\lvert E\rvert`$, $`\omega`$, $`h`$ | Graph, entity count, edge count, grounding operator, and hop depth |


## Diagrams
The manifest records 0 numbered figures and 0 numbered tables. The source contains two unnumbered reference layouts. The main symbol layout is recreated in the three Decoder table blocks. The overloaded-symbol layout is recreated under Core mechanics. No source number or caption is invented.
### Source layout accounting
1. Main notation layout: 57 symbol-meaning entries across three horizontal blocks.
2. Overloaded symbols layout: 8 symbols with two meanings or a scope restriction.
3. Conventions text: 8 rules covering bases, units, orientation, and the recurring default.
## Whiteboard pack
### What to draw
1. Draw system flow $`x\to q=\Phi(x)\to\mathop{\mathrm{RET}}\nolimits\to c_{1:k}\to\mathop{\mathrm{GEN}}\nolimits_\theta\to y`$.
2. Separate index, evaluation, training, serving and graph symbols.
3. Mark local overloads and library-specific parameter names.
4. Distinguish PQ sub-code bits from total bits and HNSW slots from realized degree.
5. Add explicit log bases, storage units, latency versus service time and vector orientation.
6. State the example dimension and parameter count without claiming a universal setup.
### Spoken script
Read the notation sheet by scope, not by a familiar letter alone. Trace input, query transformation, retrieval, contexts and generation. Then separate index sizes from model parameters, first-hit metrics from set coverage, and stage latency from attributable resource time. In this reference, PQ uses $`m`$ and HNSW uses $`M`$, but a library can reuse uppercase $`M`$ for PQ. Distinguish bits per sub-code from total code bits, and use dimension checks for matrix products. Finish with explicit logarithm bases, decimal or binary units, and the illustrative encoder setup.
## Interview traps
### 1. Are $`m`$ and $`M`$ interchangeable in product quantization?
Not within one locally defined derivation. This reference uses $`m`$ for PQ and $`M`$ for HNSW, but this is not universal: Faiss PQ also names its subquantizer parameter $`M`$. Read the API and map it explicitly; HNSW parameters are not exact realized degrees.[\[1\]](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
### 2. Does $`k`$ always mean the same retrieval depth?
No. $`k`$ can mean contexts reaching the generator or PQ centroids per sub-quantizer, while $`k^{\prime}`$ is retrieval depth before reranking. $`k_1`$ is the BM25 saturation parameter, so the exact mark and local scope choose the meaning.
### 3. Which metric symbols are easy to collapse incorrectly?
$`R@k`$ is Recall at $`k`$, while $`P@k`$ is Precision at $`k`$. $`P_k`$ is segmentation error. $`\mathop{\mathrm{MRR}}\nolimits`$, $`\mathop{\mathrm{DCG}}\nolimits`$, $`\mathop{\mathrm{nDCG}}\nolimits@k`$, $`\mathop{\mathrm{IoU}}\nolimits`$, and $`F_1`$ also retain separate meanings, so similar letters do not make them interchangeable.
### 4. What defaults apply when a derivation omits units?
Do not repair a missing definition by guessing. State $`\ln`$ for natural-log quantities such as nats and the displayed BM25 IDF, and $`\log_2`$ for DCG and bit counts. Use explicitly decimal $`\mathrm{GB}`$ or binary $`\mathrm{GiB}`$. Milliseconds, QPS and dollars/query are convenient defaults only when dimensionally appropriate; stage wall time is not automatically billable device time.
### 5. Is $`d=768`$ a universal requirement?
No. It is the book’s recurring default with a $`110`$ million-parameter encoder, and its purpose is to make worked examples comparable without repeating the setup. A written alternative overrides a convention default.
## Key numbers


| Item | Value | Meaning |
| --- | --- | --- |
| Primary reference entries | 57 | All symbol-meaning entries in the main layout |
| Primary source blocks | 3 | System and index, evaluation, and training or serving or graph notation |
| Explicit overloaded symbols | 8 | $`d`$, $`k`$, $`m`$, $`M`$, $`b`$, $`N`$, $`P`$, and $`G`$ |
| Convention rules | 8 | Base, storage, unit, orientation, and recurring-default rules |
| Retrieval gate values | $`2`$ | $`g(x)`$ takes values $`\{0,1\}`$ |
| Default logarithm base | $`2`$ | Applies unless another base is written |
| Decimal gigabyte | $`10^9`$ bytes | Binary units require an explicit quoted-source exception |
| Default embedding dimension | $`768`$ | Recurring setup for comparable worked examples |
| Default encoder size | $`110`$ million parameters | Paired with $`d=768`$ in the recurring setup |
| Manifest numbered visuals | 0 figures, 0 tables | The source layouts are unnumbered |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [A5\_annotated\_reading\_list](A5_annotated_reading_list.md) · [A7\_glossary](A7_glossary.md)
