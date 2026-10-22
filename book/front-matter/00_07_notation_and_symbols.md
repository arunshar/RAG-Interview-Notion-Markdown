# 00.07 Notation and Symbols
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_06\_for\_practicing\_engineers](00_06_for_practicing_engineers.md) · [00\_08\_acknowledgments](00_08_acknowledgments.md)
This unit standardizes the symbols, abbreviations, overloads, and measurement conventions used throughout the retrieval-augmented generation (RAG) book.
## TL;DR
- The semi-parametric frame separates knowledge in generator parameters $`\theta`$ from knowledge in the retrievable datastore $`D`$ .
- The main flow is $`x`$ to $`q`$ to a retrieval gate, a ranked retrieval result, $`k`$ contexts, and generated answer $`y`$ .
- Retrieval and ranking notation covers relevance, rank, term statistics, encoders, token lengths, loss, and temperature.
- Index notation names vector count, dimension, graph degree, search breadth, product quantization, and recall targets.
- Evaluation and training notation distinguishes ranking quality, chunking quality, retrieval distributions, generator supervision, and low-rank factors.
- Multimodal, graph, latency, and cost symbols connect retrieval choices to modality thresholds, graph size, stage timing, and query price.
- Several conventional symbols are overloaded. The source keeps them and requires context to identify the intended meaning.
## The story
Learned generator parameters are $`\theta`$; datastore $`D`$ remains externally addressable. Component stores are $`D_1,\ldots,D_m`$. Transform input $`x`$ into retriever query $`q=\Phi(x)`$; the gate $`g(x)\in\{0,1\}`$ selects whether retrieval runs. These are local definitions, not a claim that weights and stores have the same update operations.
When retrieval runs, write $`\operatorname{RET}(q,D)`$ for the ranked candidates. Candidate depth $`k'`$ precedes filtering/reranking and final depth $`k\le k'`$; these are requested upper limits, and fewer eligible contexts may remain. The generator conditions on the request and retained $`c_1,\ldots,c_k`$ to generate $`y`$ token by token.
The library has several control panels. Retrieval scores and ranks items. The vector index controls how broadly the shelves are searched. Evaluation checks ranked results and chunk boundaries. Training aligns the retriever’s distribution with the generator’s preference. Multimodal admission rules and graph grounding decide which rooms and relationships can join the search.
The checkout counter records each stage latency $`t_i`$ and query cost $`C_q`$ . Some labels repeat across panels. The label $`d`$ can mean a document at the search desk or a dimension at the index panel. The label only becomes safe when its panel and first-use definition are visible.
## Decoder table


| Technical term | Plain-English meaning | Why it matters |
| --- | --- | --- |
| The Semi-Parametric Frame: $`\theta`$ | Generator parameters, the parametric component | Knowledge stored in weights. |
| The Semi-Parametric Frame: $`D`$ | Datastore, the non-parametric component | Retrievable at test time. |
| The Semi-Parametric Frame: $`D_1,\ldots,D_m`$ | Individual stores composing $`D`$ | Source selection ranges over these stores. |
| The Semi-Parametric Frame: $`x`$ | User input | What the user actually typed. |
| The Semi-Parametric Frame: $`q=\Phi(x)`$ | Query issued to the retriever | $`\Phi`$ is the query transformation. |
| The Semi-Parametric Frame: $`g(x)\in\{0,1\}`$ | Retrieval gate | Decision one is whether to retrieve. |
| The Semi-Parametric Frame: $`\operatorname{RET}(q,D)`$ | Retriever | Returns a ranked list. |
| The Semi-Parametric Frame: $`\operatorname{GEN}_\theta`$ | Generator | The language model. |
| The Semi-Parametric Frame: $`c_1,\ldots,c_k`$ | Retrieved contexts passed to the generator | These are the contexts retained after truncation. |
| The Semi-Parametric Frame: $`y`$ | Generated answer | $`y_t`$ is its t-th token. |
| The Semi-Parametric Frame: $`k`$ | Number of contexts passed to $`\operatorname{GEN}`$ | The top-k cutoff. |
| The Semi-Parametric Frame: $`k'`$ | Retrieval depth before reranking | $`k'\ge k`$. Truncation is decision five. |
| Retrieval and Ranking: $`d`$ | A document or chunk | Context disambiguates it from dimension. |
| Retrieval and Ranking: $`s(q,d)`$ | Relevance score of $`d`$ for $`q`$ | No additional convention is stated. |
| Retrieval and Ranking: $`\operatorname{rank}(d)`$ | Position of $`d`$ in a ranked list | Positions are 1-indexed. |
| Retrieval and Ranking: $`f`$ | Term frequency in a document | Used by BM25. |
| Retrieval and Ranking: $`k_1,b`$ | BM25 saturation and length-normalization parameters | Typically $`k_1\approx1.2`$ and $`b\approx0.75`$. |
| Retrieval and Ranking: $`\operatorname{avgdl}`$ | Average document length in the collection | Used by BM25. |
| Retrieval and Ranking: $`E_Q(\cdot),E_P(\cdot)`$ | Query and passage encoders | Dense passage retrieval (DPR) convention. Bi-encoder. |
| Retrieval and Ranking: $`L_q,L_p`$ | Query and passage length in tokens | Drives encoder floating-point operations (FLOPs). |
| Retrieval and Ranking: $`\operatorname{MaxSim}`$ | Late-interaction operator | ColBERT. |
| Retrieval and Ranking: $`\mathcal L`$ | Training loss | No additional convention is stated. |
| Retrieval and Ranking: $`\tau`$ | Softmax temperature | Sharpens or flattens a score distribution. |
| Indexing and Vector Search: $`N`$ | Number of vectors in the index | Context distinguishes it from model parameter count. |
| Indexing and Vector Search: $`d`$ | Embedding dimensionality | Examples are 768 and $`1{,}536`$. |
| Indexing and Vector Search: $`M`$ | HNSW neighbor/slot parameter | Base slots commonly $`2M`$; $`8M\,\mathrm B`$ is only their four-byte link payload |
| Indexing and Vector Search: $`M_0`$ | HNSW base-layer neighbor capacity | $`M_0=2M`$ is a common allocation, not exact realized degree |
| Indexing and Vector Search: $`\mathrm{ef}`$ | HNSW search-time candidate list size | Recall and latency knob per query. |
| Indexing and Vector Search: $`n_{\mathrm{probe}}`$ | Number of inverted file (IVF) cells searched per query | Recall and latency knob per query. |
| Indexing and Vector Search: $`m`$ | Number of product quantization (PQ) sub-quantizers, or subvectors | Each subvector has dimension $`d/m`$. |
| Indexing and Vector Search: $`k`$ | PQ centroids per subspace | With $`k=2^{b_{\mathrm{sub}}}`$, total code bits are $`m b_{\mathrm{sub}}`$ and fp32 codebooks use $`4kd\,\mathrm B`$ |
| Indexing and Vector Search: $`b`$ | Bits per sub-code only when explicitly defined | $`k=2^{b_{\mathrm{sub}}}`$; distinguish total $`b_{\mathrm{code}}=m b_{\mathrm{sub}}`$ |
| Indexing and Vector Search: $`R@k`$ | Recall at k | The index’s quality target. |
| Evaluation: $`\operatorname{MRR}`$ | Mean reciprocal rank of the first relevant item | Multiple relevant items are allowed; specify cutoff and missing-hit rule |
| Evaluation: $`\operatorname{DCG},\operatorname{nDCG}@k`$ | Discounted cumulative gain and normalized discounted cumulative gain at k | Used with graded relevance. |
| Evaluation: $`P@k,R@k`$ | Precision and recall at k | Set metrics. They are order-insensitive. |
| Evaluation: $`P_k,\operatorname{WindowDiff}`$ | Text-segmentation error metrics | Used to evaluate a chunker. |
| Evaluation: $`\operatorname{IoU}`$ | Intersection over union of chunk spans | Used to evaluate a chunker. |
| Evaluation: $`F_1`$ | $`2PR/(P+R)`$ for $`P+R>0`$ | State zero-denominator and macro/micro policies; not equivalent to accuracy |
| Training the RAG System: $`D'`$ | Retrieved subset used for a training step | Normalization is over $`D'`$, not $`D`$. |
| Training the RAG System: $`\lambda(d,x)`$ | Normalized retrieval weight in an output ensemble | REPLUG. |
| Training the RAG System: $`P_R(d\mid x)`$ | Retriever’s distribution over the top k | Defined over the same k as the generator preference. |
| Training the RAG System: $`Q_{\mathrm{LM}}(d\mid x,y)`$ | Generator’s preference distribution over the same k | The supervision signal. |
| Training the RAG System: $`\gamma,\beta`$ | Retriever and generator softmax temperatures | Language model (LM)-supervised retrieval. |
| Training the RAG System: $`\operatorname{KL}(P\Vert Q)`$ | Directed Kullback–Leibler divergence | A fixed teacher target commonly uses $`\operatorname{KL}(Q_{\mathrm{LM}}\Vert P_R)`$; direction matters |
| Training the RAG System: $`r`$ | LoRA rank | $`W+(\alpha/r)BA`$ with compatible dimensions; scale and trainable matrices must be stated |
| Training the RAG System: $`A,B`$ | LoRA low-rank factors | The low-rank factors in $`W+BA`$. |
| Multimodal and Graph Retrieval: $`E_q,E_d`$ | Query and document embeddings in a shared space | Modality-agnostic. |
| Multimodal and Graph Retrieval: $`\theta_m`$ | Example modality-specific threshold | Validate score and modality calibration; no universal ban on a calibrated global gate |
| Multimodal and Graph Retrieval: $`S_{ij}`$ | Contrastive similarity matrix | Contrastive Language-Image Pre-training (CLIP). $`N`$ true pairs lie on the diagonal. |
| Multimodal and Graph Retrieval: $`\mathcal G=(V,E)`$ | Knowledge graph, its entities, and its edges | $`V`$ is the entity set. $`E`$ is the edge set. |
| Multimodal and Graph Retrieval: $`\lvert V\rvert,\lvert E\rvert`$ | Entity and edge counts | Drives construction and storage cost. |
| Multimodal and Graph Retrieval: $`\widehat Q=\omega(Q,\mathcal G)`$ | Query grounded against the graph schema | $`\omega`$ is the grounding operator. |
| Multimodal and Graph Retrieval: $`h`$ | Traversal depth in hops | No additional convention is stated. |
| Latency and Cost: $`t_i`$ | Latency of pipeline stage $`i`$ | Stages are embed, search, rerank, prefill, and decode. |
| Latency and Cost: $`C_q`$ | Cost per query under declared resource attribution | $`\sum_j\rho_j s_j/(3.6\times10^6)`$ for dollars/hour and attributable milliseconds; not raw end-to-end wall time |
| Latency and Cost: $`B`$ | Serving batch size | No additional convention is stated. |
| Latency and Cost: $`s`$ | Number of tokens in a context or an answer | Context disambiguates its use. |
| Latency and Cost: $`L`$ | Number of transformer layers | Context distinguishes it from training loss $`\mathcal L`$. |
| Symbols That Carry More Than One Meaning: $`d`$ | Meaning A is a document or chunk in retrieval | Meaning B is embedding or hidden dimensionality in indexing and systems. |
| Symbols That Carry More Than One Meaning: $`k`$ | Meaning A is the number of retrieved contexts | Meaning B is centroids per sub-quantizer in product quantization. |
| Symbols That Carry More Than One Meaning: $`m`$ | Meaning A is the number of product quantization sub-quantizers | Meaning B is the number of datastore sources or a modality index. |
| Symbols That Carry More Than One Meaning: $`M`$ | HNSW parameter in this reference | Faiss PQ and other literatures may also use uppercase $`M`$; map each local meaning |
| Symbols That Carry More Than One Meaning: $`b`$ | Meaning A is the BM25 length-normalization parameter | Meaning B is code length in bits or bytes per parameter. |
| Symbols That Carry More Than One Meaning: $`N`$ | Meaning A is the number of vectors in the index | Meaning B is model parameter count in FLOPs arithmetic. |
| Symbols That Carry More Than One Meaning: $`P`$ | Meaning A is model parameter count in encoder FLOPs arithmetic | Meaning B is instance price per hour in cost arithmetic. |
| Symbols That Carry More Than One Meaning: $`\mathcal G`$ | Meaning A is the knowledge graph | Meaning B is the graph’s schema alone in source section $`40.3`$. |
| Conventions: $`\log`$ | Logarithm with an explicitly declared base | $`\ln`$ for nats/log loss and displayed BM25; $`\log_2`$ for DCG and code bits |
| Conventions: `GB` | Gigabyte in storage arithmetic | Decimal units. $`1\,\mathrm{GB}=10^9\,\mathrm B`$ unless a quoted source uses binary units, which the text states at the point of use. |
| Conventions: $`t_i`$ | Latency | Measured in milliseconds unless otherwise labeled. |
| Conventions: `throughput` | Query throughput | Measured in queries per second unless otherwise labeled. |
| Conventions: $`C_q`$ | Query cost | Measured in United States (US) dollars per query unless otherwise labeled. |
| Conventions: `vector` | Vector orientation | Vectors are column vectors. |
| Conventions: $`\mathsf T`$ | Transpose marker | $`\mathsf T`$ denotes transpose. |
| Conventions: $`1{,}000`$ | Number formatting pattern | Thousands separators appear in numbers of four digits or more. |
| Conventions: $`d=768,\ N_{\mathrm{param}}=110\times10^6`$ | Recurring default configuration | Dimension 768 and Bidirectional Encoder Representations from Transformers base (BERT-base) parameter count 110M support comparison across chapters. |


## Core mechanics
### The Semi-Parametric Frame
- What: $`\theta`$ names the generator’s parametric knowledge. $`D`$ names test-time retrievable knowledge. The remaining symbols trace input, query transformation, gating, retrieval, context selection, and generation.
- Why: The frame keeps weight-based knowledge separate from datastore knowledge and makes each pipeline decision visible.
- Failure without it: A reader could confuse what the user typed with the retriever query, or confuse pre-reranking depth $`k'`$ with the final top-k cutoff.
- Stated cost: No explicit cost formula appears in this section.
### Retrieval and Ranking
- What: This notation defines documents, relevance scores, rank positions, BM25 statistics, encoders, token lengths, late interaction, training loss, and softmax temperature.
- Why: It gives one vocabulary for lexical ranking, encoded retrieval, training, and score calibration.
- Failure without it: Rank could lose its 1-indexed convention, $`d`$ could be mistaken for dimension, or $`\tau`$ could be mistaken for a score instead of a temperature.
- Stated cost: Query length $`L_q`$ and passage length $`L_p`$ drive encoder floating-point operations.
### Indexing and Vector Search
- $`N_{\mathrm{vec}}`$ counts indexed vectors, not necessarily documents, while $`d`$ is their dimension.
- HNSW $`M`$ controls link allocation; $`M_0=2M`$ is a common base-layer capacity. With four-byte slots, $`8M\,\mathrm B`$ is the base-layer-only link subtotal, not complete memory. Expected upper-layer slots add $`4M/(M-1)\,\mathrm B`$ under the stated geometric model.
- For PQ, $`m`$ equal subspaces have width $`d/m`$. If each sub-code has $`b_{\mathrm{sub}}`$ bits, $`k=2^{b_{\mathrm{sub}}}`$, total code length is $`m b_{\mathrm{sub}}`$ bits and fp32 centroid payload is $`4kd\,\mathrm B`$.
- $`\mathrm{efSearch}`$ and $`n_{\mathrm{probe}}`$ are search-work controls, not exact hop counts or guaranteed recall values.
### Evaluation
- MRR uses the first relevant hit and permits several relevant items; it does not assume exactly one exists.
- Precision and recall at a fixed cutoff ignore order within that prefix, but changing membership changes them. Distinguish ANN-neighbor recall, relevance recall and complete supporting-evidence coverage.
- nDCG needs a gain function, positive IDCG or an explicit zero-IDCG policy, cutoff and judgment universe.
- $`F_1`$ requires declared averaging and zero-denominator policies; macro-$`F_1`$ is not accuracy or an overall error rate. Segmentation metrics and span IoU measure different objectives.
### Training the RAG System
- A retrieved training support $`D'`$ must be explicitly defined. A top-candidate normalized distribution is conditional on that support; full-corpus objectives use a different normalizer.
- If $`Q_{\mathrm{LM}}`$ is the fixed teacher target, a common cross-entropy-equivalent direction is $`\operatorname{KL}(Q_{\mathrm{LM}}\Vert P_R)`$. Reverse KL is different; writing generic $`\operatorname{KL}(P\Vert Q)`$ does not specify which system is the target.
- Require positive softmax temperatures. Define mixture weights, support and whether gradients flow through teacher scores.
- For $`W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}`$, use $`A\in\mathbb R^{r\times d_{\mathrm{in}}}`$ and $`B\in\mathbb R^{d_{\mathrm{out}}\times r}`$; a common LoRA update is $`W+(\alpha/r)BA`$. Other orientations require a consistent factor order.
### Multimodal and Graph Retrieval
- Shared embeddings require compatible training/normalization; “multimodal” does not make arbitrary encoder spaces interchangeable.
- $`\theta_m`$ denotes a chosen modality-specific gate. Calibration may justify another policy, so “never global” is not a universal theorem.
- A CLIP-style batch has diagonal designated pairs under its pairing convention; duplicates and multiple valid matches can require a richer target.
- Distinguish full graph $`\mathcal G=(V,E)`$ from $`\operatorname{schema}(\mathcal G)`$. Graph grounding is not proof that the extracted assertions or generated answer are true.
### Latency and Cost
- Stage latency $`t_i`$ and attributable device service time $`s_j`$ are different quantities. A serial critical path can add stage wall times; parallel fan-out follows its dependency graph, not the sum of all branches.
- With resource price $`\rho_j`$ in dollars/hour and attributable service time $`s_j`$ in milliseconds, $`C_q^{\mathrm{resource}}=\sum_j\rho_j s_j/(3.6\times10^6)`$. This excludes fixed reservations, utilization policy and API billing unless explicitly included.
- A one-resource serial approximation can use $`\rho\sum_i t_i/(3.6\times10^6)`$ only when those times really represent attributable, nonoverlapped reservation/service. Batching, concurrency and different devices invalidate casual wall-time billing.
- Use $`N_{\mathrm{param}}`$ for parameters and $`\rho`$ for price to avoid the overloaded $`P`$ in calculations.
### Symbols That Carry More Than One Meaning
- What: The source explicitly overloads $`d`$, $`k`$, $`m`$, $`M`$, $`b`$, $`N`$, $`P`$, and $`\mathcal G`$.
- Why: These symbols remain consistent with their separate retrieval, representation-learning, and systems literatures.
- Failure without it: A familiar symbol could silently acquire the wrong meaning and corrupt a formula or design explanation.
- Stated cost: No cost is assigned to the convention. Each section states the active meaning at first use.
### Conventions
- State the base explicitly: natural logs for nats, log loss and the displayed BM25 IDF; base $`2`$ for DCG and bit counts.
- Use decimal $`\mathrm{GB}`$ and binary $`\mathrm{GiB}`$ precisely, not interchangeably. The storage convention does not require binary values to be quotations.
- Vectors are columns by default; stated row-stacked representations override that default. Check every product’s dimensions.
- The nominal $`d=768`$, $`110\times10^6`$-parameter encoder is an illustrative recurring setup. Parameter count does not uniquely determine output dimension or achieved throughput.
## Diagrams
The manifest lists zero figures and zero captioned tables for physical PDF pages 28-31. No diagram redraw is required.
The source contains seven unnumbered notation tables and one unnumbered overload table. The exhaustive Decoder table preserves every row from those layouts. The prose conventions block is preserved in both the Decoder table and Core mechanics.
## Whiteboard pack
### What to draw
1. Draw $`x\to q=\Phi(x)`$ and a retrieval gate.
2. Draw $`\operatorname{RET}(q,D)`$, candidate depth $`k'`$, eligibility/reranking and final contexts $`c_{1:k}`$.
3. Condition $`\operatorname{GEN}_\theta`$ on the request and contexts to generate $`y`$.
4. Separate vector counts from model parameters, PQ sub-code bits from total code bits, and configured HNSW slots from realized degree.
5. Add evaluation definitions and explicit teacher/retriever KL direction.
6. Finish with dimension checks, log bases, storage units, resource service time and query cost.
### Spoken script
Start with input $`x`$ and transform it into retriever query $`q=\Phi(x)`$. A gate decides whether retrieval runs. Draw candidate contexts from datastore $`D`$, then eligibility checks, reranking and final contexts for the generator. Under that path, separate score, index, evaluation and training panels. Define every overloaded letter locally. PQ bits per sub-code differ from total code bits, and HNSW capacity parameters differ from realized degree. Finally distinguish stage latency from attributable resource time, state the direction of any KL loss, and write explicit logarithm bases and storage units.
## Interview traps
### 1. Why does the notation keep overloaded symbols instead of inventing new ones?
Each symbol is conventional in its own literature. The source preserves those conventions and requires the active meaning to be stated at first use.
### 2. Are $`k'`$ and $`k`$ interchangeable?
No. Candidate depth $`k'`$ and final depth $`k`$ refer to different stages. In the simple truncation pipeline, $`k\le k'`$, but filters can leave fewer eligible items and other retrieval rounds need their own candidate counts.
### 3. What is the difference between $`D`$ and $`D'`$?
$`D`$ denotes the full store and $`D'`$ a declared retrieved training support. Top-candidate normalized distributions use $`D'`$. That is a modeling choice, not proof that every retrieval-training objective normalizes only over a retrieved subset.
### 4. Can $`M`$ denote product quantization?
This reference reserves $`m`$ for PQ and $`M`$ for HNSW, but external APIs can differ. Faiss PQ explicitly uses uppercase $`M`$ for its number of subquantizers. Map meaning and dimensions rather than assuming a global case-based prohibition.[\[1\]](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
### 5. What does $`P`$ mean in cost arithmetic?
Use an explicit price symbol such as $`\rho`$ in cost arithmetic and $`N_{\mathrm{param}}`$ for model parameters. The millisecond-to-hour divisor is $`3.6\times10^6`$. Multiplying price by summed stage wall time is valid only under a stated attribution model; it is not a universal per-query cost formula.
## Key numbers


| Quantity | Source value | Meaning |
| --- | --- | --- |
| Contributing literatures | $`3`$ | Information retrieval, representation learning, and systems supply the conventional notation. |
| Primary notation categories | $`7`$ | Semi-parametric frame, retrieval and ranking, indexing and vector search, evaluation, training, multimodal and graph retrieval, and latency and cost. |
| Explicitly overloaded symbols | $`8`$ | $`d`$, $`k`$, $`m`$, $`M`$, $`b`$, $`N`$, $`P`$, and $`\mathcal G`$. |
| Retrieval gate outputs | $`0`$ or $`1`$ | Do not retrieve or retrieve. |
| Retrieval gate decision | $`1`$ | The source calls retrieve-or-not decision one. |
| Truncation decision | $`5`$ | The source calls truncating $`k'`$ to $`k`$ decision five. |
| Rank origin | $`1`$ | Ranked positions are 1-indexed. |
| Typical BM25 $`k_1`$ | $`\approx1.2`$ | Saturation parameter. |
| Typical BM25 $`b`$ | $`\approx0.75`$ | Length-normalization parameter. |
| Example embedding dimensions | $`768`$, $`1{,}536`$ | Examples for index dimensionality $`d`$. |
| HNSW storage term | $`8M\,\mathrm B`$ base-layer slot subtotal | Omit neither upper layers nor runtime/container memory |
| Layer 0 degree convention | $`M_0=2M`$ | Layer 0 has twice the stated upper-layer out-degree. |
| Product quantization relation | $`k=2^{b_{\mathrm{sub}}}`$; total bits $`m b_{\mathrm{sub}}`$ | Per-subcode bits and total code length are different |
| Mean reciprocal rank assumption | First relevant hit | Does not require exactly one relevant item |
| Contrastive diagonal | $`N`$ true pairs | CLIP places the true pairs on the diagonal of $`S_{ij}`$. |
| Latency stages | $`5`$ | Embed, search, rerank, prefill, and decode contribute stage latencies. |
| Cost denominator | $`3.6\times10^6\,\mathrm{ms/hour}`$ | Use attributable resource time, prices and a stated batching/reservation model |
| Graph-schema overload pointer | Section `40.3` | $`\mathcal G`$ means the graph schema alone at this source pointer. |
| Default logarithm base | Explicit $`\ln`$ or $`\log_2`$ | Natural logs for nats/log loss; base $`2`$ for DCG and bit counts |
| Decimal gigabyte | $`1\,\mathrm{GB}=10^9\,\mathrm B`$ | Binary units apply only when a quoted source states them at the point of use. |
| Thousands-separator threshold | Use separators for readability | Years, identifiers and dimensional subscripts need not acquire grouping separators |
| Recurring default dimension | $`768`$ | Used for comparison across chapters. |
| Recurring BERT-base parameter count | $`110\times10^6`$ | Used for comparison across chapters. |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_06\_for\_practicing\_engineers](00_06_for_practicing_engineers.md) · [00\_08\_acknowledgments](00_08_acknowledgments.md)
