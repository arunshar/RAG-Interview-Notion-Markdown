# Appendix A: Formula Sheet
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [41\_end\_to\_end\_design\_drills](../part-12-systems-design/41_end_to_end_design_drills.md) · [A2\_question\_bank](A2_question_bank.md)
This appendix is a compact derivation and interview-use guide for eight formula groups used in Retrieval-Augmented Generation (RAG) that preserves the source equations, numeric examples, conditions, and rules of thumb while translating Best Matching 25 (BM25), Reciprocal Rank Fusion (RRF), Mean Reciprocal Rank (MRR), normalized Discounted Cumulative Gain (nDCG), modularity, Product Quantization (PQ), floating-point operations (FLOPs) with latency, and end-to-end index memory into operational decisions.
## TL;DR
- Carry units through every derivation. Most design-round arithmetic failures are unit failures.
- Cross-check each result against a published rule or a known system before defending it.
- BM25 separates inverse document frequency, term-frequency saturation, and document-length normalization.
- RRF rewards agreement without requiring comparable scores, while MRR and nDCG measure different ranking goals.
- Modularity compares connectivity with a degree-preserving expectation, while PQ obtains exponential combinations from linearly many centroids.
- FLOP, key-value cache, and Low-Rank Adaptation arithmetic turn model choices into latency and memory budgets.
- Index capacity must include vectors, identifiers, graph links, rebuild headroom, replication, text payload, and metadata.
## The story
Treat the appendix as a cockpit checklist for a RAG design interview.
A number such as $`3.38680485\times10^{13}`$ becomes a compute quantity only when labeled FLOPs. Dividing it by an assumed achieved FLOP/s rate gives an idealized compute time, not a latency benchmark. BM25 separately shows rarity, saturation and document-length effects.
RRF is the sensor combiner. It merges rank signals without pretending that a lexical score and a cosine score share a scale. MRR is the first-hit warning light. It cares only about the first relevant result. nDCG is the ranked-load gauge. It keeps graded relevance and gives later useful results nonzero credit.
Modularity is the route-map gauge. It asks whether a proposed community has more internal connection than degree alone would predict. PQ is the cargo compressor. It splits one large vector into smaller subspaces, stores a modest codebook for each, and represents combinations it never stores explicitly.
The FLOP and latency panel converts model size, token count, context length, cache shape, and adapter rank into time and memory. The final capacity card counts everything loaded on the aircraft. Vector bytes alone are not the fleet. Identifiers, graph links, replication, rebuild copies, text payload, and metadata all travel too.
A safe pilot can rebuild every reading when a constraint changes. Memorizing the final number without its units, condition, and sanity check is not enough. After two weeks of preparation, continued reliance on the card is a reason to re-derive each formula rather than re-read it.
## Decoder table


| Symbol or term | Meaning in this appendix | Mechanical role |
| --- | --- | --- |
| $`q`$ | Query | Supplies terms to BM25 or identifies one evaluation case |
| $`D`$ | Document | Receives a BM25 score |
| $`t`$ | Query term | Contributes one BM25 summand |
| $`N`$ in A.1 | Collection document count | Sets the inverse document frequency scale |
| $`n_t`$ | Documents containing term $`t`$ | Makes common terms less informative |
| $`f(t,D)`$ | Frequency of term $`t`$ in document $`D`$ | Enters a saturating term-frequency factor |
| $`\lvert D\rvert`$ | Document length | Drives BM25 length normalization |
| $`\mathop{\mathrm{avgdl}}\nolimits`$ | Mean document length | Defines the length baseline |
| $`k_1`$ | BM25 saturation parameter | Controls how quickly repeated term counts flatten |
| $`b`$ in A.1 | BM25 length parameter | Interpolates from no length charge to full charge |
| $`\ell`$ | Document length divided by mean length | Makes length normalization dimensionless |
| $`K`$ in A.1 | Length-adjusted BM25 denominator term | Places the length charge inside saturation |
| IDF | Inverse document frequency | Rewards rarer query terms |
| TF | Term frequency | Counts repeated occurrences before saturation |
| $`L`$ in A.2 | Number of ranked lists | Sets how many votes RRF can combine |
| $`r_\ell(d)`$ | Rank of document $`d`$ in list $`\ell`$ | Determines one list’s reciprocal contribution |
| $`k`$ in A.2 | RRF rank constant | Sets the exchange rate between rank and consensus |
| $`w(r)`$ | RRF weight at rank $`r`$ | Makes the fusion curve explicit |
| $`r^{*}`$ | Consensus crossover rank | Marks when agreement beats one list’s top hit |
| RR | Reciprocal Rank | Scores the first relevant result for one query |
| $`\mathop{\mathrm{rank}}\nolimits_q`$ | First relevant rank for query $`q`$ | Defines RR and becomes failure when absent |
| $`\mathcal{Q}`$ | Evaluation query set | Supplies the denominator in MRR |
| MRR | Mean Reciprocal Rank | Averages one first-hit score per query |
| $`\mathop{\mathrm{rel}}\nolimits_i`$ | Relevance grade at rank $`i`$ | Supplies gain to CG and DCG |
| CG | Cumulative Gain | Adds relevance without position sensitivity |
| DCG | Discounted Cumulative Gain | Discounts relevance by log rank |
| IDCG | Ideal Discounted Cumulative Gain | Scores the best ordering of available judgments |
| nDCG | Normalized Discounted Cumulative Gain | Divides DCG by IDCG for cross-query comparison |
| $`k`$ in A.4 | Evaluation cutoff | Limits the ranked prefix being scored |
| $`Q`$ in A.5 | Modularity | Measures connection beyond a degree-fixed chance model |
| $`m`$ in A.5 | Total graph edge count | Normalizes the modularity sum |
| $`A_{ij}`$ | Binary adjacency indicator | Marks whether nodes $`i`$ and $`j`$ share an edge |
| $`k_i`$ | Degree of node $`i`$ | Sets expected random connectivity |
| $`c_i`$ | Community assigned to node $`i`$ | Selects within-community pairs |
| $`\delta(c_i,c_j)`$ | Same-community indicator | Keeps a pair only when both community labels match |
| Leiden | Community-detection heuristic with connected-community guarantees under its assumptions | Does not certify the global maximum or semantic coherence |
| Louvain | Earlier modularity algorithm | Can return a disconnected community |
| GraphRAG and LightRAG | Graph-based retrieval systems with different indexing/update designs | Compare measured maintenance paths; modularity alone proves no cost ordering |
| $`d`$ in A.6 | Vector dimension | Sets raw storage and subvector width |
| $`m`$ in A.6 | Number of PQ subvectors | Sets code length and subvector count |
| $`k`$ in A.6 | Centroids per subspace | Sets bits per subvector and codebook size |
| $`\frac{d}{m}`$ | Subvector width | Must be integral for even splitting |
| $`k^m`$ | Representable vector combinations | Grows exponentially in subspaces |
| $`k\times m`$ | Stored centroids | Grows linearly in subspaces |
| $`N^{*}`$ | PQ lookup-table break-even candidates | Marks when precomputation amortizes |
| $`b`$ in A.6 | Bits for a hypothetical single-codebook identifier | For PQ distinguish sub-code bits from total $`m\log_2 k`$ bits |
| FAISS and Inverted File Product Quantization (IVF-PQ) | The cited vector-search implementation and inverted-file PQ configuration | They supply the memory sanity rule and the amortization setting |
| fp32 | 32-bit floating-point storage | Uses four bytes per scalar |
| bf16 | 16-bit brain floating-point storage | Uses two bytes per scalar |
| int8 | 8-bit integer storage | Uses one byte per scalar |
| int4 | 4-bit integer storage | Uses one-half byte per scalar |
| $`N`$ in A.7 | Model parameter count | Sets model compute and weight memory |
| $`D_{\mathrm{tokens}}`$ | Training token count | Multiplies training FLOPs |
| $`C`$ in A.7 | Approximate total training compute | Equals about $`6ND_{\mathrm{tokens}}`$ |
| $`s`$ | Prefill context length | Enters linear model work and quadratic attention work |
| $`L_{\mathrm{layers}}`$ | Transformer layer count | Multiplies attention and cache cost |
| $`d_{\mathrm{hidden}}`$ | Hidden dimension | Scales attention work |
| $`b_{\mathrm{bytes}}`$ | Bytes per stored scalar | Converts parameter or cache counts into memory |
| KV cache | Key-value cache | Stores attention keys and values per token and request |
| $`n_{\mathrm{kv}}`$ | Key-value head count | Scales cache bytes per token |
| $`d_{\mathrm{head}}`$ | Attention head dimension | Scales cache bytes per token |
| Prefill and decode | Context processing and cached autoregressive generation | Often different bottleneck regimes; not universally compute- and bandwidth-bound |
| LoRA | Low-Rank Adaptation | Replaces full matrix updates with rank-$`r`$ factors |
| $`r`$ in A.7 | Adapter rank | Sets trainable LoRA parameters |
| $`d_{\mathrm{in}}`$, $`d_{\mathrm{out}}`$ | Adapted matrix dimensions | Determine one adapter’s parameter count |
| $`M_{\mathrm{weights}}`$, $`M_{\mathrm{KV},\mathrm{token}}`$, and $`P_{\mathrm{LoRA}}`$ | Weight memory, cache bytes per token, and adapter parameters | They separate fixed, concurrency-dependent, and trainable costs |
| $`N`$ in A.8 | Indexed vector count | Multiplies per-vector index bytes |
| $`b_{\mathrm{vec}}`$ | Bytes for one vector code | Includes raw fp32 or PQ code bytes |
| $`b_{\mathrm{id}}`$ | Posting identifier bytes | Costs four to eight bytes per vector |
| $`b_{\mathrm{link}}`$ | Expected allocated HNSW neighbor-slot bytes under a stated layer model | Not realized degree or complete graph-index memory |
| HNSW | Hierarchical Navigable Small World index | Stores neighbor identifiers across graph layers |
| IVF | Inverted File index | Uses centroids and posting lists without HNSW links |
| $`M`$ | HNSW neighbor parameter | Sets layer-zero and upper-layer link storage |
| $`M_0`$ | Layer-zero HNSW neighbors | Uses up to twice $`M`$ identifiers |
| $`n_{\mathrm{list}}`$ | IVF centroid count | Adds coarse-centroid memory |
| $`M_{\mathrm{index}}`$ | Total index footprint | Counts vector, identifier, link, IVF, and PQ bytes |
| $`R`$ | Replication factor | Multiplies the live index copies |
| $`h`$ in A.8 | Artifact-specific rebuild reserve or overlap fraction | State $`h_I`$ and $`h_P`$ rather than assuming payload is never copied |
| $`M_{\mathrm{payload}}`$ | Text and metadata bytes | Often dominates after strong vector compression |
| $`M_{\mathrm{fleet}}`$ | Replicated index, payload and runtime memory in a declared capacity domain | Payload replication and build-host memory cannot silently disappear |
| $`\mathop{\mathrm{shards}}\nolimits`$ | Logical index partitions, distinct from physical node count | Aggregate-byte ceilings give a node-capacity lower bound, not the shard count |
| GB, TB, KiB, MB, bytes, bits, and FLOP/s | Gigabytes, terabytes, kibibytes, megabytes, byte and bit counts, and floating-point operations per second | Carrying the unit prevents incompatible quantities from being compared |
| RAM | Random-access memory | Supplies usable capacity per node |


## Core mechanics
### Mathematical conventions
Use decimal units $`1\,\mathrm{GB}=10^9\,\mathrm B`$ and $`1\,\mathrm{TB}=10^{12}\,\mathrm B`$; binary units are explicitly $`1\,\mathrm{KiB}=2^{10}\,\mathrm B`$ and $`1\,\mathrm{MiB}=2^{20}\,\mathrm B`$. Keep full precision until the reported result and use $`\approx`$ for rounding or a compute/storage model. Natural logarithms are written $`\ln`$; DCG uses $`\log_2`$.
Ranked lists have positive integer ranks and a declared tie rule, identity/deduplication policy and depth. Define missing-hit and zero-IDCG behavior. Modularity below is for an undirected unweighted graph with $`m\gt 0`$ edges, including diagonal null-model terms; optimization is heuristic, not a global-optimum certificate.
Compute is a dense-model FLOP proxy, not measured latency. Attention architecture, causal masking, achieved throughput, batching, memory traffic, communication and queueing matter. Storage equations are scoped byte ledgers: count only applicable index components, distinguish expected allocated link slots from realized links, and state which artifacts are replicated or rebuilt. A ceiling of aggregate bytes is a capacity lower bound, not a deployable shard/replica plan.
### A.1 BM25
#### Formula and symbols
Section $`18.3`$ derives BM25, which scores one query-document pair by summing over query terms.

```math
\mathop{\mathrm{BM25}}\nolimits(q,D)=\sum_{t\in\mathcal T(q)}
 \ln\!\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right)
 \frac{f(t,D)(k_1+1)}{f(t,D)+k_1(1-b+b\lvert D\rvert/\mathop{\mathrm{avgdl}}\nolimits)}.
```

Here $`\mathcal T(q)`$ is the set of distinct analyzed query terms; repeated-query weighting is omitted. Require $`N\ge1`$, $`0\le n_t\le N`$, $`\mathop{\mathrm{avgdl}}\nolimits\gt 0`$, $`k_1\gt 0`$ and $`0\le b\le1`$. This shifted natural-log IDF is positive even for a term in every document; it is not the unshifted Robertson IDF.
The source defaults are:

```math
k_1=1.2, \qquad b=0.75
```

Define the dimensionless length ratio and adjusted denominator term as:

```math
\ell=\frac{|D|}{\mathop{\text{avgdl}}}, \qquad K=k_1(1-b+b\ell)
```

At $`b=0`$, document length is ignored. At $`b=1`$, the length ratio is charged in full.
#### Plain-language mechanics
IDF raises rare terms. TF saturation gives the first few occurrences most of the benefit. Length normalization changes the saturation denominator instead of applying a hard score penalty. The two knobs are independent. $`k_1`$ controls saturation speed. $`b`$ controls length sensitivity.
#### Worked saturation grid
At average document length, $`\ell=1`$, and $`k_1=1.2`$, the TF factor becomes:

```math
\mathop{\text{TFfactor}}(f)=\frac{2.2f}{f+1.2}
```


| $`f`$ | $`1`$ | $`2`$ | $`5`$ | $`10`$ | $`20`$ | Limit |
| --- | --- | --- | --- | --- | --- | --- |
| TF factor (rounded) | $`1.000`$ | $`1.375`$ | $`1.774`$ | $`1.964`$ | $`2.075`$ | $`2.200`$ |


The second occurrence adds $`0.375`$. Ten additional occurrences, from $`f=10`$ to $`f=20`$, add approximately $`0.111186`$, about $`29.65\%`$ of that second-occurrence increment. The final column is the limit, not a finite-frequency score.
#### Worked length normalization
For a document twice the mean length with one term occurrence:

```math
f=1, \qquad k_1=1.2, \qquad b=0.75, \qquad \ell=2
```


```math
K=1.2(1-0.75+0.75\times2)=1.2\times1.75=2.1
```


```math
\mathop{\mathrm{TFfactor}}\nolimits=\frac{2.2}{1+2.1}\approx0.709677.
```

The TF factor is approximately $`29.0323\%`$ below its average-length, one-occurrence value of $`1`$. The length adjustment acts inside the denominator; the percentage is the resulting score-factor decrease, not a denominator discount.
#### Conditions, failures, and rules
- Treat $`k_1=1.2`$ and $`b=0.75`$ as example starting values, not universal implementation defaults or optima.
- A lower $`b`$, such as $`0.3`$, can be tested when length reflects additional useful content. Document genre alone does not determine the best value.
- The unshifted IDF $`\ln((N-n_t+0.5)/(n_t+0.5))`$ is negative above half the collection. The displayed shifted form is positive and must not be confused with flooring the unshifted form.
- BM25 and hybrid retrieval remain baselines to evaluate on the actual relevance task, including exact entities and rare terms.
### A.2 Reciprocal Rank Fusion
#### Formula and symbols
Section $`21.6`$ derives RRF, which assigns one reciprocal contribution for every list that contains document $`d`$.

```math
\mathop{\text{RRF}}(d)=\sum_{\ell=1}^{L}\frac{1}{k+r_\ell(d)}
```

A missing document contributes nothing. This is equivalent to sending its rank to infinity. The conventional default is:

```math
k=60
```

Within one list, rank $`r`$ has weight relative to that list’s first result:

```math
\frac{w(r)}{w(1)}=\frac{k+1}{k+r}
```

That ratio is exactly one-half at:

```math
r=k+2
```

For $`L`$ equal-weight lists and $`k\ge0`$, assume a document is present at the same rank $`r`$ in every list. Compare it with a competitor present only at rank $`1`$ in one list:

```math
\frac{L}{k+r}\gt \frac{1}{k+1} \quad\Longleftrightarrow\quad r \lt  L(k+1)-k
```

#### Plain-language mechanics
RRF combines ranks rather than claiming comparable raw scores. Require $`k+r\gt 0`$ and consistent document identity. Raising $`k`$ flattens within-list weights; it does not establish that agreement is independent or correct. Correlated lists can repeat the same error. Unequal list weights or different ranks need their actual sums, not the equal-rank shortcut.
#### Worked crossover cases
With two lists and $`k=60`$:

```math
r^{*}=2(61)-60=62
```

Under that equal-rank comparison, a common document at rank $`61`$ or better in both lists beats the competitor present only at one list’s first position. At rank $`62`$ the scores tie. List depth must be sufficient to observe those ranks.
With two lists and $`k=0`$:

```math
r^{*}=2
```

Against a competitor present only at one list’s first position, strict victory requires a common rank of $`1`$. A common rank-$`2`$ item ties it. This inequality does not imply that all $`k=0`$ RRF rankings reduce to interleaving the list heads.
With three lists and $`k=60`$:

```math
r^{*}=3(61)-60=123
```

#### Conditions, failures, and rules
- Treat $`k=60`$ as a common evaluated convention, not a derived optimum.
- Use RRF when rank aggregation suits the task; normalized score fusion is an alternative to validate, not something justified by normalization alone.
- Fusion depth $`100`$ followed by a final depth $`10`$ is an example. Tune candidate coverage, latency and judged quality together.
- Shallow lists may hide useful consensus. An added weak or correlated list may also promote errors; extra agreement is not an automatic quality gain.
### A.3 Mean Reciprocal Rank
#### Formula and symbols
Section $`32.2`$ derives Reciprocal Rank, which depends only on the first relevant result for query $`q`$.

```math
\mathop{\mathrm{RR}}\nolimits_q@K=\begin{cases}
1/r_q,&r_q\le K,\\
0,&\text{no relevant hit in the evaluated prefix},
\end{cases}
```

Here $`r_q`$ is the first relevant rank, $`K\ge1`$ is the evaluation cutoff, and the query set is nonempty. MRR averages those per-query values; multiple relevant items are permitted but only the first contributes:

```math
\mathop{\mathrm{MRR}}\nolimits@K=\frac{1}{\lvert\mathcal Q\rvert}\sum_{q\in\mathcal Q}\mathop{\mathrm{RR}}\nolimits_q@K.
```

A missing relevant result in the evaluated prefix receives zero. Unbounded MRR uses a declared complete ranking or another explicit missing-hit convention.
#### Plain-language mechanics
MRR asks one question: how far down is the first relevant document? It ignores every relevant result after the first.
#### Worked examples
A ten-document list with relevant results at ranks one and ten has:

```math
\mathop{\text{RR}}=1/1=1.0
```

A list with only one relevant result at rank one receives the same score. The following calculation is not MRR:

```math
\frac{1/1+1/10}{10}=0.11
```

It sums every relevant hit and divides by list length. A metric that changes when a second relevant document appears cannot be RR for that query.


| First relevant rank | $`1`$ | $`2`$ | $`3`$ | $`5`$ | $`10`$ | Absent |
| --- | --- | --- | --- | --- | --- | --- |
| RR | $`1.000`$ | $`0.500`$ | $`0.333`$ | $`0.200`$ | $`0.100`$ | $`0`$ |


Rank $`2`$ earns half the rank-$`1`$ reciprocal credit. Rank $`5`$ still earns $`0.2`$ and rank $`10`$ earns $`0.1`$; calling these “failure” is a task judgment, not a property of the formula.
#### Conditions, failures, and rules
- MRR suits first-success objectives even when several relevant items exist. It is not a complete multi-evidence retrieval metric.
- Report cutoff, relevance criterion, query denominator and tie handling.
- Pair rank metrics with recall and, for multi-hop RAG, coverage of the full sufficient evidence set. “At least one gold hit” and “all required evidence retrieved” are different events.
- The value of a rank improvement versus a recall improvement depends on the downstream task and costs.
### A.4 nDCG
#### Formula and symbols
Section $`32.2`$ derives the ranking metrics, beginning with Cumulative Gain, which ignores order.

```math
\mathop{\text{CG}}@k=\sum_{i=1}^{k}\mathop{\text{rel}}_i
```

Discounted Cumulative Gain divides each grade by a log-rank term.

```math
\mathop{\text{DCG}}@k=
\sum_{i=1}^{k}
\frac{\mathop{\text{rel}}_i}{\log_2(i+1)}
```

Normalize by IDCG computed from the same nonnegative gain function, cutoff and judgment universe. For positive IDCG:

```math
\mathop{\mathrm{nDCG}}\nolimits@k=\frac{\mathop{\mathrm{DCG}}\nolimits@k}{\mathop{\mathrm{IDCG}}\nolimits@k},\qquad \mathop{\mathrm{IDCG}}\nolimits@k\gt 0.
```

Under those conditions, and with the ideal ordering formed from the same candidate/judgment universe:

```math
0\leq\mathop{\text{nDCG}}@k\leq1
```

#### Plain-language mechanics
CG ignores position and DCG discounts later gains. The $`+1`$ makes rank $`1`$ undiscounted under $`\log_2`$. IDCG uses a declared judgment universe, ideally all available relevant judgments rather than just the returned items. A score of $`1`$ means an ideal ranking relative to that universe; it does not prove exhaustive labeling of the corpus. For zero IDCG, explicitly exclude the query or assign a stated convention such as zero.
#### Worked discount grid


| Rank $`i`$ | $`1`$ | $`2`$ | $`3`$ | $`5`$ | $`10`$ |
| --- | --- | --- | --- | --- | --- |
| $`\log_2(i+1)`$ | $`1.000`$ | $`1.585`$ | $`2.000`$ | $`2.585`$ | $`3.459`$ |
| DCG weight | $`1.000`$ | $`0.631`$ | $`0.500`$ | $`0.387`$ | $`0.289`$ |
| RR weight | $`1.000`$ | $`0.500`$ | $`0.333`$ | $`0.200`$ | $`0.100`$ |


At rank 10, DCG still assigns $`0.289`$ while RR assigns $`0.100`$. A useful document at rank four or five retains material credit.
#### Gain choice and judgment conditions
The displayed formula uses direct relevance grades. An alternative uses exponential gain:

```math
\mathop{\text{gain}}_i=2^{\mathop{\text{rel}}_i}-1
```

Exponential gain emphasizes higher grades more strongly, but does not make ordinal labels cardinal ground truth. State gain, cutoff and averaging. Treating unjudged items as zero gain is a common evaluation policy, not evidence of non-relevance; shallow pools can bias comparisons.
#### Conditions, failures, and rules
- Report cutoff, gain function, relevance judgments, zero-IDCG policy and macro/micro averaging.
- Binary nDCG and graded nDCG are different metrics. Choose exponential gain only when that weighting matches the objective.
- Pair reranker nDCG with retrieval recall at a declared candidate cutoff, and evaluate end-to-end evidence sufficiency.
- nDCG can penalize missing judged relevant documents as well as poor ordering. Recomputing IDCG only from returned hits can conceal retrieval failure.
### A.5 Modularity
#### Formula and symbols
For an undirected unweighted graph with $`m\gt 0`$, the following is standard modularity at resolution $`1`$. Louvain and Leiden are optimization heuristics; neither name alone guarantees a global maximum.

```math
Q=\frac{1}{2m}
\sum_{i,j}
\left[A_{ij}-\frac{k_i k_j}{2m}\right]
\delta(c_i,c_j)
```

Here $`m`$ counts undirected edges once, the sum over $`i,j`$ is ordered and includes the diagonal, $`A_{ij}`$ is binary adjacency, and $`k_i`$ is node degree. The delta selects pairs assigned to the same community. The configuration-model expectation is:

```math
\frac{k_i k_j}{2m}
```

Modularity therefore measures connectivity beyond chance at fixed degree. It does not measure absolute connectivity.
#### Worked four-node partition
Nodes one, two, and three form a triangle. Node four has one edge to node three.

```math
m=4, \qquad 2m=8
```

For partition A, the triangle and singleton contributions are:

```math
6-\frac{49}{8}=-0.125
```


```math
0-\frac{1}{8}=-0.125
```

Thus:

```math
Q_A=\frac{-0.25}{8}=-0.03125
```

For partition B, all four nodes are together.

```math
Q_B=\frac{8-64/8}{8}=0
```

The modularity gain is:

```math
Q_B-Q_A=0.03125
```

For these two partitions, adding the dangling edge changes a tie into a $`0.03125`$ preference for merging. This does not claim that the selected partition is uniquely optimal over all partitions.
#### Conditions, failures, and rules
- Under the standard assumptions in the Leiden analysis, completed iterations provide connected-community guarantees that Louvain lacks. They do not guarantee semantic coherence or a globally optimal partition.[\[1\]](https://arxiv.org/abs/1810.08473)
- The modularity null model uses whole-graph degrees and edge count. Updates can change optimal partitions outside the edited neighborhood, but do not force every community to change or every implementation to rebuild globally.
- Compare GraphRAG and LightRAG maintenance policies and measured workloads; global modularity alone proves no universal cost ordering.
- Generic relations can distort communities, while capping them can remove useful evidence. Audit the effect rather than assuming semantic value always collapses.
- Modularity has a resolution limit, but no universal internal-edge cutoff guarantees absorption at every hierarchy level. A more precise pairwise merge condition is:

```math
\Delta Q_{A,B}=\frac{e_{AB}}m-\frac{K_AK_B}{2m^2};\qquad
\Delta Q_{A,B}\gt 0\ \Longleftrightarrow\ 2me_{AB}\gt K_AK_B.
```

Here $`e_{AB}`$ counts edges between two communities and $`K_A,K_B`$ are their degree sums. For a single bridge and equal degree sums, merging is favorable when $`K_A=K_B\lt \sqrt{2m}`$. Translating that into internal-edge counts depends on external degrees; the classic symmetric examples have a $`\sqrt{m/2}`$ scale. Resolution and hierarchy choices can change the outcome.[\[2\]](https://arxiv.org/abs/physics/0607100)
### A.6 Product Quantization sizing
#### Formula and symbols
Section $`16.1`$ derives Product Quantization. Split a $`d`$-dimensional vector into $`m`$ subvectors, each with width:

```math
\frac{d}{m}
```

For equal subspaces, require positive integer $`m`$ dividing $`d`$. With $`k=2^{b_{\mathrm{sub}}}`$ centroids, each sub-code uses $`b_{\mathrm{sub}}`$ bits and total packed code length is:

```math
b_{\mathrm{code}}=m\log_2k=m b_{\mathrm{sub}}\ \text{bits},\qquad b_{\mathrm{packed}}=\left\lceil\frac{m b_{\mathrm{sub}}}{8}\right\rceil\,\mathrm B.
```

The number of representable vectors is:

```math
k^m
```

The number of stored centroids is:

```math
k\times m
```

Each centroid has dimension $`\frac{d}{m}`$, so fp32 codebook memory is:

```math
k\times m\times\frac{d}{m}\times4
=k\times d\times4\text{ bytes}
```

For fixed $`k`$, $`d`$ and centroid precision, this shared codebook payload is independent of $`m`$. That is not true if changing $`m`$ changes $`k`$, adds transforms or adds per-subquantizer metadata.
In a deliberately simplified operation count, building the query lookup table costs $`2kd`$ and each candidate lookup costs $`m`$, versus $`2d`$ for a direct distance. The equality break-even is:

```math
N^{*}=\frac{2kd}{2d-m}
```

candidate distances, provided $`2d\gt m`$. Strict savings require more candidates than $`N^*`$. Cache behavior, SIMD, residual/coarse-distance tables and actual kernels can change the measured break-even.
#### Why the mechanism compresses
Representable combinations grow as $`k^m`$. Stored centroids grow only as $`k\times m`$. This asymmetry is the compression mechanism.
#### Worked 768-dimensional design
Start with:

```math
d=768, \qquad m=96, \qquad d/m=8, \qquad k=256=2^8
```

One fp32 vector uses:

```math
768\times4=3072\text{ bytes}
```

The PQ code uses:

```math
96\times8=768\text{ bits}=96\text{ bytes}
```

Compression is:

```math
3072/96=32
```

Representable combinations are:

```math
256^{96}=2^{768}\approx1.55251809\times10^{231}.
```

Stored centroids are:

```math
256\times96=24576
```

The codebook contains $`196{,}608`$ fp32 values and occupies:

```math
256\times768\times4=786432\text{ bytes}=768\text{ KiB}
```

This is $`786.432\,\mathrm{kB}=768\,\mathrm{KiB}`$. The source’s “$`768\,\mathrm{kB}`$” label must not be treated as a decimal-unit equality.
The break-even candidate count is:

```math
N^*=\frac{2\times256\times768}{1{,}536-96}=\frac{393{,}216}{1{,}440}\approx273.066667.
```

Equality occurs at a noninteger count, so the first whole candidate count with strict savings is $`274`$ in this proxy. A list shorter than that may not amortize the table on its own; aggregating probes and reusing tables depends on the implementation. The rounded $`273`$ is not the strict threshold.
#### Why one codebook loses
A single $`b`$-bit codebook needs:

```math
k=2^b
```

For $`b=32`$ in 768 dimensions:

```math
2^{32}\times3{,}072\,\mathrm B=13{,}194{,}139{,}533{,}312\,\mathrm B\approx13.19414\,\mathrm{TB}.
```

This is the full-centroid payload for a hypothetical single $`32`$-bit codebook in $`768`$ dimensions, not $`32`$ bytes per vector. A claim of “400 times the corpus” needs a specified corpus size and representation; it does not follow from codebook size alone.
#### Conditions, failures, and rules
- Default to $`k=256`$, which gives one byte per subvector.
- Choose $`m`$ to divide $`d`$ evenly and meet the byte budget.
- A common starting point is:

```math
m=d/8
```

- Keep $`k`$ and centroid precision fixed when claiming codebook independence from $`m`$.
- At $`k=256`$, code storage is $`m\,\mathrm B`$ per vector, before identifiers and other structures. A Faiss IVF-PQ posting normally adds an eight-byte identifier.[\[3\]](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- The Faiss warning default is $`39`$ training points per centroid; at $`256`$ centroids this is $`9{,}984`$ original training vectors, projected into each subspace. It is a warning heuristic, not a statistical guarantee or $`9{,}984m`$ distinct required source vectors.[\[4\]](https://github.com/facebookresearch/faiss/wiki/FAQ)
### A.7 FLOPs and latency
#### Compute formulas
For a dense Transformer, count a multiply-add as two FLOPs and use $`N`$ as an approximate active parameter count. The usual parameter-matmul proxies are:

```math
\begin{aligned}
W_{\mathrm{forward/token}}&\approx2N,&W_{\mathrm{backward/token}}&\approx4N,\\
W_{\mathrm{train/token}}&\approx6N,&C_{\mathrm{train}}&\approx6ND_{\mathrm{tokens}}.
\end{aligned}
```

Generation has no backward pass, but its $`2N`$ parameter-matmul proxy omits context-dependent attention, memory bandwidth, batching effects and other overhead. MoE active parameters, tied embeddings and architecture details need their own ledger.
For an idealized causal prefill over $`s`$ tokens, counting both attention matrix products over all causal pairs, a compatible proxy is:

```math
W_{\mathrm{prefill}}\approx2Ns+2L_{\mathrm{layers}}s(s+1)d_{\mathrm{hidden}}.
```

The second term is quadratic and approaches $`2L_{\mathrm{layers}}s^2d_{\mathrm{hidden}}`$ for large $`s`$. Full bidirectional attention instead uses approximately $`4L_{\mathrm{layers}}s^2d_{\mathrm{hidden}}`$. Real kernels may exploit masking and attention variants differently. Doubling context adds a superlinear attention component but does not double every serving cost. Correction: prefill is a linear parameter term plus a quadratic attention term, and in the 8B example below the attention term is about $`3\%`$ of prefill at $`2{,}048`$ tokens. Doubling context to $`4{,}096`$ tokens therefore multiplies that prefill proxy by about $`2.06`$. Section $`6.2`$ shows that, under its $`12L_{\mathrm{layers}}d^2`$ body estimate, the attention term matches the parameter term only near $`s=12d_{\mathrm{hidden}}`$, which is $`49{,}152`$ tokens for $`d_{\mathrm{hidden}}=4{,}096`$. The quadratic term therefore becomes comparable to the linear term only at tens of thousands of tokens.
#### Memory formulas
Weight memory is:

```math
M_{weights}=N\times b_{bytes}
```

Nominal payload bytes per scalar are $`4`$ for fp32, $`2`$ for bf16, $`1`$ for int8 and $`0.5`$ for packed int4. Quantization scales, zero points, alignment, unquantized layers and runtime state are additional.
KV cache per token is:

```math
M_{KV,token}=2L_{layers}n_{kv}d_{head}b_{bytes}
```

Multiply by cached tokens and independent request sequences when no prefix sharing applies. Count allocated pages, variable lengths, sliding windows, tensor-parallel placement and cache precision separately.
LoRA trainable parameters for one adapted matrix are:

```math
P_{LoRA}=r(d_{in}+d_{out})
```

#### Worked weight and cache examples
Treating the nominal “27B” parameter count as exactly $`27\times10^9`$ for illustration gives this fp32 weight payload:

```math
27\times10^9\times4=108\text{ GB at fp32}
```

An $`80\,\mathrm{GB}`$ usable device budget cannot hold that fp32 payload. The same rounded-count payload is $`54\,\mathrm{GB}`$ at bf16 and $`27\,\mathrm{GB}`$ at int8 before quantization metadata, KV cache and runtime overhead; it is not an all-in fit guarantee for a named checkpoint.
For Llama $`3.1`$ 8B with 32 layers, eight key-value heads, head dimension 128, and bf16:

```math
2\times32\times8\times128\times2
=131072\text{ bytes}
=128\text{ KiB per token}
```

A 2,048-token context uses:

```math
2048\times128\text{ KiB}=256\text{ MiB}
```

per concurrent request. The source labels this $`256\,\mathrm{MB}`$ even though its KiB-based arithmetic yields $`256\,\mathrm{MiB}`$.
#### Worked latency examples
For an 8B model with 32 layers, hidden size $`4{,}096`$, and context $`2{,}048`$:

```math
(2\times8\times10^9\times2{,}048)
+(2\times32\times2{,}048\times2{,}049\times4{,}096)
=33{,}868{,}048{,}498{,}688\ \mathrm{FLOP}.
```

At the stipulated achieved throughput $`3.4\times10^{14}\,\mathrm{FLOP\,s^{-1}}`$, the compute proxy is:

```math
\frac{33{,}868{,}048{,}498{,}688}{3.4\times10^{14}}\,\mathrm s\approx99.611907\,\mathrm{ms}.
```

For a nominal $`110`$-million-parameter cross-encoder over $`50`$ total query-document pairs of $`256`$ tokens each, parameter-matmul work alone is:

```math
2\times110\times10^6\times256\times50=2.816\times10^{12}\ \mathrm{FLOP}.
```

At the same stipulated throughput, before attention and all serving overhead:

```math
\begin{aligned}
t_{\mathrm{linear}}&\approx\frac{2.816\times10^{12}}{3.4\times10^{14}}\,\mathrm s\approx8.282353\,\mathrm{ms},\\
W_{\mathrm{with\ attention}}&\approx2.816\times10^{12}+4(12)(256^2)(768)(50)\\
&=2.9367959552\times10^{12}\ \mathrm{FLOP},\\
t_{\mathrm{with\ attention}}&\approx8.637635\,\mathrm{ms}.
\end{aligned}
```

#### Worked LoRA example
Use rank 16 adapters on query and output projections of size $`4{,}096`$ by $`4{,}096`$ across 32 layers:

```math
16(4{,}096+4{,}096)\times2\times32=8{,}388{,}608.
```

The corresponding two full square projections per layer contain $`4{,}096^2\times2\times32=1{,}073{,}741{,}824`$ parameters. Comparing only those adapted matrices:

```math
\frac{8{,}388{,}608}{1{,}073{,}741{,}824}=0.0078125=0.78125\%,\qquad \frac{4{,}096}{2\times16}=128.
```

#### Conditions, failures, and rules
- Prefill is often compute-limited at useful batch sizes; low-batch decode is often memory-bandwidth-limited. These are regimes to measure, not universal identities.
- Extra context affects prefill, KV capacity and decode attention/traffic. It need not leave decode tokens/s unchanged.
- The illustrative $`99.611907\,\mathrm{ms}`$ prefill and $`8.282353\,\mathrm{ms}`$ linear-only rerank use one stipulated rate. They are not observed wall times.
- LoRA reduces trainable adapted-matrix parameters by $`128`$ here, not total-model training FLOPs or total GPU memory by that factor.
- Size weights, cache and runtime for target concurrency and utilization. There is no universal two-to-three-times conversion from peak to achieved throughput.
### A.8 Index memory, end to end
#### Per-vector and total formulas
Sections $`16.5`$ and $`15.2`$ derive total index memory and the HNSW link term. Correction: the ten-million-vector table below uses the per-vector layout of Section $`41.1`$, which sets $`M=16`$ with $`2M=32`$ base-layer slots and counts neither coarse centroids nor the PQ codebook. Section $`16.5`$ sets $`M=32`$, counts the expected upper-layer slots, and adds coarse centroids and the PQ codebook to the same eight-byte ID, which gives $`3{,}332\,\mathrm B`$ and $`33.32\,\mathrm{GB}`$ for HNSW-Flat and $`104\,\mathrm B`$, $`1.09\,\mathrm{GB}`$ and a $`30.5`$ times ratio for IVF-PQ96. Per-vector data bytes are:

```math
b_{vec}=
\begin{cases}
4d & \text{for fp32}\\
m & \text{for PQ with }k=256
\end{cases}
```

Posting identifiers use:

```math
b_{id}=4\text{ to }8\text{ bytes}
```

Under independent geometric HNSW layer allocation $`\Pr(\ell\ge j)=M^{-j}`$, four-byte neighbor slots, $`2M`$ base slots and $`M`$ slots per allocated upper layer, the expected slot payload is:

```math
\mathbb E[b_{\mathrm{link}}]=4\left(2M+\frac{M}{M-1}\right)\,\mathrm B,\qquad M\gt 1.
```

This graph term is zero for a plain IVF index, but not necessarily for an IVF coarse quantizer that itself uses HNSW. Layer counters, offsets, actual containers and unused slot capacity are not covered by the expression.
Include only the structures present in the chosen implementation; a separate posting ID can be absent when positions supply identity. For IVF/PQ indicators and a remainder covering transforms, precomputed tables, maps and metadata:

```math
\begin{aligned}
\mathbb E[M_{\mathrm{index}}]={}&N\bigl(b_{\mathrm{vec}}+b_{\mathrm{id}}+\mathbb E[b_{\mathrm{link}}]\bigr)\\
&+\mathbf1_{\mathrm{IVF}}\,4n_{\mathrm{list}}d\,\mathrm B
 +\mathbf1_{\mathrm{PQ}}\,4kd\,\mathrm B+M_{\mathrm{other}}.
\end{aligned}
```

For $`R`$ live index copies and index-only rebuild overlap fraction $`h_I`$, payload replication $`R_P`$ and payload rebuild fraction $`h_P`$, one explicit fleet ledger is:

```math
M_{\mathrm{fleet}}=R(1+h_I)M_{\mathrm{index}}+R_P(1+h_P)M_{\mathrm{payload}}+M_{\mathrm{runtime}}.
```

#### HNSW link derivation
Layer zero stores up to:

```math
M_0=2M
```

The expected number of allocated upper layers is a geometric tail sum, not an exact layer count for every node:

```math
\mathbb E[b_{\mathrm{link}}]=4\left(2M+M\sum_{j\ge1}M^{-j}\right)\,\mathrm B=4\left(2M+\frac{M}{M-1}\right)\,\mathrm B.
```

At $`M=32`$:

```math
\mathbb E[b_{\mathrm{link}}]=4\left(64+\frac{32}{31}\right)\,\mathrm B\approx260.129032\,\mathrm B.
```

The base-layer-only slot payload is $`256\,\mathrm B`$ at $`M=32`$. The expected upper-layer addition is approximately $`4.129032\,\mathrm B`$, or $`1.612903\%`$ of that base payload. This is not a complete all-in Faiss memory rule. The next table deliberately compares only vector/code and base-layer slots; IVF adds its stated posting IDs.[\[3\]](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
#### Ten-million-vector comparison
Use 10 million chunks and $`d=768`$.


| Configuration | Scoped bytes/vector | Corpus subtotal | Ratio to raw subtotal |
| --- | --- | --- | --- |
| HNSW-Flat, $`M=16`$ | $`3{,}072+128=3{,}200\,\mathrm B`$ | $`32.0\,\mathrm{GB}`$ | $`1`$ |
| HNSW over $`64`$-byte PQ | $`64+128=192\,\mathrm B`$ | $`1.92\,\mathrm{GB}`$ | $`16.666667`$ times smaller |
| Faiss-style IVF-PQ, $`96`$-byte code | $`96+8=104\,\mathrm B`$ | $`1.04\,\mathrm{GB}`$ | $`30.769231`$ times smaller |


The compressed-HNSW subtotal has $`128/192=2/3`$ of its bytes in base-layer links. With this link allocation fixed, sending vector payload to zero can reduce that subtotal by at most $`192/128=1.5`$ further. Link widths and topology can be redesigned, so “incompressible” is not an information-theoretic claim. A four-byte-ID toy IVF-PQ posting would be $`100\,\mathrm B`$ and $`1.00\,\mathrm{GB}`$, unlike the eight-byte Faiss posting.
#### Rebuild, replication, and payload
If a full old and new index coexist in the declared capacity domain, index-only overlap can require:

```math
h_I=1.
```

A separate build host moves that memory elsewhere in the fleet. A target reserve such as the following is a planning assumption, not a consequence of hot swapping:

```math
h_I\approx0.2\quad\text{(example reserve, not a bound).}
```

At $`400`$ tokens/chunk, stipulate four UTF-8 bytes/token, not merely four Unicode characters/token. Uncompressed text then uses $`1{,}600\,\mathrm B=1.6\,\mathrm{kB}`$ per chunk. At $`10^7`$ chunks:

```math
10^7\times1{,}600\,\mathrm B=16\,\mathrm{GB}.
```

Metadata at $`200\,\mathrm B`$ per chunk adds $`2\,\mathrm{GB}`$, giving an $`18\,\mathrm{GB}`$ payload before compression, object overhead and additional copies.
Using only the $`1.04\,\mathrm{GB}`$ Faiss posting subtotal, two live copies, full index-only rebuild overlap, two payload copies and no payload rebuild gives:

```math
2\times2\times1.04+2\times18=40.16\,\mathrm{GB}.
```

If the $`18\,\mathrm{GB}`$ payload is instead one shared resident copy, the corresponding subtotal is $`2\times2\times1.04+18=22.16\,\mathrm{GB}`$. The old $`22\,\mathrm{GB}`$ figure additionally used the four-byte-ID toy $`1\,\mathrm{GB}`$ index. All versions omit remaining index/runtime memory; neither proves fit or availability.
#### Conditions, failures, and rules
- Separate capacity, per-query service demand, traffic, replication and failover. Chapter 41’s ceiling estimates are lower bounds rather than a claim that a fixed two-node cluster suffices.
- Aggregate-memory ceiling division gives only:

```math
n_{\mathrm{nodes}}\ge\left\lceil\frac{M_{\mathrm{resident,total}}}{M_{\mathrm{usable/node}}}\right\rceil.
```

- Shards are logical partitions; replicas are copies. Aggregate bytes alone do not assign partitions, traffic or failure domains.
- Text can dominate after compression, but the crossover depends on chunk size, representation, compression and replication; “30 times” is not universal.
- State identifiers, graph slots, codebooks, coarse centroids, rebuild overlap, payload copies and runtime. A $`32`$-fold vector-code reduction is not a $`32`$-fold whole-fleet reduction.
## Diagrams
Source visual accounting is 0 figures and 0 tables in the manifest. The source contains display equations and unnumbered numeric grids, not captioned source visuals. This appendix preserves those items as equations and Markdown grids without inventing figure or table numbers.
## Whiteboard pack
### What to draw
1. Draw a ranking lane with BM25, RRF, MRR, and nDCG in source order.
2. Under BM25, split IDF from saturating and length-normalized TF.
3. Under RRF, draw two ranked lists meeting at the crossover inequality.
4. Under MRR and nDCG, contrast first-hit credit with graded log-discounted credit.
5. Draw modularity as observed adjacency minus degree-based expected adjacency.
6. Draw one vector split into $`m`$ PQ subspaces and label $`k^m`$ versus $`k\times m`$.
7. Draw compute, weight, KV cache, and LoRA formulas as a model-serving block.
8. Finish with the memory stack: vector, id, links, centroids, replicas, rebuild, payload.
9. Add units to every intermediate value and circle each independent sanity check.
### Spoken script
Start with lexical scoring and rank aggregation, then first-hit and graded ranking metrics. State gain, cutoff and missing-judgment policy. Draw modularity as observed connectivity minus a degree-based expectation, with no semantic or global-optimum guarantee. Next show PQ code bits, shared codebooks and scoped memory. Convert model FLOPs into an idealized compute time only under a declared achieved rate. Add attention, weight precision, KV cache and adapter parameters. Finish with a ledger for index structures, payload replication, rebuild copies and runtime; byte ceilings are capacity lower bounds, not deployment plans.
## Interview traps
### 1. Why is a numerically correct answer still wrong?
A value without units, scope, or achieved hardware rate is not defensible. Track bytes versus bits, decimal GB versus KiB, per-token versus per-request cache, per-vector versus fleet memory, and FLOPs versus FLOPs per second. State which local meaning of $`N`$, $`k`$, $`m`$, $`b`$, or $`Q`$ applies before substituting.
### 2. Which ranking formula is being misused?
BM25 length normalization belongs inside the saturation denominator, while RRF uses ranks because source scores are incomparable. MRR reads only the first relevant result, whereas nDCG needs a cutoff, gain scale, and judgment-pool caveat. Diagnose the objective before tuning a constant.
### 3. Which sanity ablation exposes a false metric or graph claim?
Add a second relevant hit after rank $`1`$: RR remains $`1`$. Permute unequal grades: CG stays fixed while DCG can change. Remove the dangling edge from the four-node graph: both displayed partitions then have modularity $`0`$, a tie rather than a forced reversal. Generic relations may still yield communities, but semantic quality requires its own audit.
### 4. Why did compression or latency miss the estimate?
For PQ, check whether codebook memory was confused with per-vector codes and whether $`m`$ divides $`d`$. For latency, use achieved rather than peak FLOPs per second and include quadratic attention. For index memory, add identifiers, HNSW links, rebuild overlap, replicas, payload, and metadata because a 32 times vector compression does not imply 32 times fleet compression.
### 5. What claim limits must be said aloud?
The RRF constant is a convention, not an optimum. nDCG needs positive IDCG or an explicit zero-IDCG rule. Modularity has graph- and resolution-dependent limits. The toy PQ operation model gives $`N^*\approx273.066667`$ and first strict integer saving at $`274`$. The compute times use unrounded arithmetic and one assumed rate; metadata, payload copies, model architecture and batching can change the answer.
## Key numbers


| Topic | Key values | Use and limit |
| --- | --- | --- |
| Formula-sheet reliance | Two weeks | If the sheet is still needed, re-derive each formula instead of re-reading it |
| BM25 defaults and limits | $`k_1=1.2`$, $`b=0.75`$; $`b\in[0,1]`$ | Illustrative parameters, not universal defaults |
| BM25 TF limit | $`1.000`$, $`1.375`$, $`1.774`$, $`1.964`$, $`2.075`$, $`2.200`$ | Frequencies 1, 2, 5, 10, 20, and infinity |
| BM25 length case | $`K=2.1`$, factor $`\approx0.709677`$ | About $`29.0323\%`$ factor decrease versus average length |
| BM25 lower length setting | Example $`b=0.3`$; unshifted IDF negative above half | Tune by evaluation; the displayed shifted IDF stays positive |
| RRF default | $`k=60`$ | Conventional original evaluation value |
| RRF crossovers | Equal-rank tie points $`62`$, $`123`$, and $`2`$ at $`k=0`$ | Strict victories are below the tie point; no general interleaving conclusion |
| RRF deep fusion | Example depth $`100`$, final depth $`10`$ | Tune jointly with candidate coverage and costs |
| RR grid and false formula | $`1,\ 0.5,\ 0.333,\ 0.2,\ 0.1,\ 0`$, and non-MRR $`0.11`$ | First hit at 1, 2, 3, 5, 10, absent, plus the rejected two-hit calculation |
| nDCG discount grid | Log values $`1.000`$, $`1.585`$, $`2.000`$, $`2.585`$, $`3.459`$ and weights $`1.000`$, $`0.631`$, $`0.500`$, $`0.387`$, $`0.289`$ | Ranks 1, 2, 3, 5, and 10 |
| nDCG range and grades | $`[0,1]`$ for nonnegative gains and positive IDCG | State gain, judgment universe, cutoff and zero-IDCG policy |
| Modularity toy | $`m=4`$, $`2m=8`$, contributions $`-0.125`$ and $`-0.125`$, $`Q_A=-0.03125`$, $`Q_B=0`$ | Merge gain is $`0.03125`$ |
| Modularity resolution | $`2me_{AB}\gt K_AK_B`$ favors merging | No universal internal-edge cutoff or every-level absorption guarantee |
| PQ design | $`d=768`$, $`m=96`$, $`k=256`$ | Eight dimensions and one byte per subvector |
| PQ compression | $`3{,}072`$ bytes to 96 bytes, 32 times | Vector-only compression |
| PQ combinations | $`2^{768}\approx1.55251809\times10^{231}`$ | $`24{,}576`$ stored centroids; not materialized combinations |
| PQ codebook | $`196{,}608`$ fp32 scalars; $`786{,}432\,\mathrm B=768\,\mathrm{KiB}`$ | Also $`786.432\,\mathrm{kB}`$; fixed $`k,d`$ and precision |
| PQ break-even | $`N^*\approx273.066667`$; strict whole count $`274`$ | Simplified operation model, not measured latency |
| PQ single codebook | $`2^{32}`$ centroids; $`\approx13.19414\,\mathrm{TB}`$ | A corpus multiplier needs an explicit corpus size |
| PQ training rule | $`39\times256=9{,}984`$ source training vectors | Faiss warning heuristic; same vectors project into each subspace |
| Compute and precision rules | Forward $`2N`$, backward $`4N`$, training $`6N`$, fp32 4 bytes, bf16 2, int8 1, int4 $`0.5`$ | Per-token FLOPs and per-scalar bytes |
| Gemma 2 27B weights | $`108\,\mathrm{GB}`$ fp32, $`54\,\mathrm{GB}`$ bf16, $`27\,\mathrm{GB}`$ int8, $`80\,\mathrm{GB}`$ card | Excludes context cache and fp32 does not fit |
| Llama $`3.1`$ 8B KV | 128 KiB per token, $`256\,\mathrm{MiB}`$ at $`2{,}048`$ tokens | The source labels the binary result $`256\,\mathrm{MB}`$ per concurrent request |
| Prefill | $`33{,}868{,}048{,}498{,}688\,\mathrm{FLOP}`$; $`\approx99.611907\,\mathrm{ms}`$ | Causal-pair attention proxy at the stipulated achieved rate |
| Reranking | $`2.816\times10^{12}\,\mathrm{FLOP}`$, $`8.282353\,\mathrm{ms}`$ linear only | With stated bidirectional attention: $`2.9367959552\times10^{12}\,\mathrm{FLOP}`$, $`8.637635\,\mathrm{ms}`$ |
| LoRA | $`8{,}388{,}608`$ versus $`1{,}073{,}741{,}824`$; $`0.78125\%`$ | A $`128`$-fold reduction only in the adapted-matrix trainable parameters |
| HNSW links | Expected $`260.129032\,\mathrm B`$ versus base-only $`256\,\mathrm B`$ | $`M=32`$; $`1.612903\%`$ extra slot payload, not exact node memory |
| HNSW-Flat | $`3{,}200\,\mathrm B`$/vector; $`32.0\,\mathrm{GB}`$ | Vector/base-slot subtotal only |
| HNSW-PQ | $`192\,\mathrm B`$/vector; $`1.92\,\mathrm{GB}`$ | $`16.666667`$ times smaller subtotal; $`2/3`$ base links |
| IVF-PQ | Faiss $`104\,\mathrm B`$/posting, $`1.04\,\mathrm{GB}`$ | $`96+8`$ bytes; toy $`96+4=100`$ is a different ID assumption |
| Rebuild headroom | $`h_I=1`$ for a full overlapping copy; $`0.2`$ is an example reserve | Separate build hosts still consume fleet resources |
| Payload | $`16\,\mathrm{GB}`$ text plus $`2\,\mathrm{GB}`$ metadata | Assumes four bytes/token and one declared payload copy |
| Fleet example | $`40.16\,\mathrm{GB}`$ with two payload copies; $`22.16\,\mathrm{GB}`$ shared | Scoped posting subtotal, two live copies, full index rebuild overlap; runtime omitted |
| Payload crossover | Workload-dependent, not a universal $`30`$-fold threshold | Count text compression, code size, links and copy policies |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [41\_end\_to\_end\_design\_drills](../part-12-systems-design/41_end_to_end_design_drills.md) · [A2\_question\_bank](A2_question_bank.md)
