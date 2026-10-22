# For Interview Candidates
[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_04\_how\_to\_use\_this\_book](00_04_how_to_use_this_book.md) · [00\_06\_for\_practicing\_engineers](00_06_for_practicing_engineers.md)
This guide turns Retrieval-Augmented Generation (RAG) interview preparation into a decision-first plan for design, derivation, debugging, pacing, and practice.
## TL;DR
- Organize answers around decisions, constraints and alternatives rather than only component names.
- Derive scoped storage and query-work estimates; corpus size and request rate jointly affect multiple resources.
- Distinguish reranker candidate depth $`k^{\prime}`$ from final evidence count $`k`$ and from actual prompt-token count.
- The lost-in-the-middle U-curve concerns evidence position at a given context length, not a universal quality curve as $`k`$ grows.
- Precision of retrieved documents is not calibrated answer correctness and cannot be compared directly with an abstention threshold.
- The suggested study schedules and timed drills are planning heuristics, not validated readiness cutoffs.
## The story
Imagine that you are auditioning to run a restaurant kitchen. A user query is a customer’s order. The retriever is a pantry runner who fetches ingredients. The generator is the cook who turns those ingredients into a finished dish.
The audition has four possible rounds. A kitchen manager first checks whether you know the tools and stations. The design round asks you to plan the whole service. The deep dive keeps you at one station until your reasoning runs out. The debugging round gives you a bad dish and asks where the service failed. Behavioral and project-history conversations may sit beside these rounds, but these source pages do not cover them.
The judges watch for four signals. First, they want decisions, not a recitation of stations. Saying “receive order, prepare ingredients, cook, and plate” names work that every kitchen performs. It does not explain a choice. Second, they want quantities you can derive. A pantry size or serving-time estimate must come from visible arithmetic. Third, they change the party size, pantry capacity, or time budget and watch you recalculate. Fourth, they expect you to name each technique’s cost.
The five decisions are a useful planning lens: whether to retrieve, where to look, how to reformulate, how to search, and how to present evidence. Measure outcomes and test plausible causes of regressions across this path. Multiple decisions or their interactions can be responsible; the framework does not guarantee one owning stage.[\[1\]](../part-01-landscape/03_a_repeatable_framework_for_any_rag_design_question.md)[\[2\]](../part-10-evaluation/34_evaluating_the_system.md)
The top-$`k`$ question asks how much evidence to present. Increasing depth in a nested candidate list cannot remove an already included answer, but it need not improve generated-answer quality. A senior answer prices actual input tokens and tests evidence placement. The lost-in-the-middle U-curve is quality versus evidence position at fixed context, not quality first rising and then falling with $`k`$. A staff answer separates reranker depth $`k^{\prime}`$ from generator depth $`k`$ and tests whether further search has positive marginal value.[\[3\]](../part-06-retrieval-ranking/22_reranking.md)[\[4\]](../part-09-context-assembly/30_lost_in_the_middle.md)
Good pacing starts before the five decisions. You ask about pantry size, order rate, serving-time limit, ingredient freshness, and the required quality. You then move through the five choices, measurement, changed limits, and a short wrap-up. Skipping sizing can confuse pantry capacity with kitchen labor. In system terms, that mistake confuses memory, which stores the search structure, with compute, which processes requests.
Use short routes for targeted practice, not as readiness certifications. For a longer route, build an owned test system, write judgments and inject controlled faults only in an isolated recoverable copy. Never remove or poison production/shared data as a learning exercise.
For each arithmetic section, read the recipe once, cover it, rewrite it from memory, and change one constant. Then answer the interview prompts aloud in a short form and a longer form. Finish by checking that every takeaway can be explained without the page.
The common failures now look concrete. Do not merely recite the kitchen line. Do not send the pantry runner on every order. Do not treat counter space as free. Do not quote a quality score without saying what it counts. Do not confuse consistency with a bad ingredient for truth about the world. Do not replace the runner before locating the broken station. Do not assume a label on the plate proves which ingredient actually shaped the dish.
## Decoder table


| Technical term | Plain-English meaning | Why it matters |
| --- | --- | --- |
| Retrieval-Augmented Generation (RAG) | A system that retrieves external material and gives it to a generator | It is the system family discussed in these interview pages |
| Retrieval | Fetching material for a query | It can help or hurt, so the design must decide when to use it |
| Component | One standard stage or tool in a system | Listing components does not reveal the choices between them |
| Retriever | The component that searches for candidate material | Its query, scorer, index, and depth all affect the result |
| Generator | The component that produces the final answer from the query and presented material | Presentation choices shape what the generator can use |
| Generation | Producing the final answer | It is the last stage in the component walk |
| Ingest | Loading source material into the retrieval system | It begins the seven-stage component walk but is not a decision by itself |
| Corpus | The full document collection available for search | Its size drives index memory and affects system sizing |
| Datastore | The set of sources from which retrieval can draw | Conflicts between sources can create silent contradictions |
| Query | The request given to the retrieval system | The retriever may need a transformed version of the user’s words |
| Query transformation | Changing the user’s request before retrieval | It can make the request more useful to the retriever |
| Query compression | Removing parts of a query before retrieval | It is one possible transformation for selected queries |
| Query expansion | Adding material to a query before retrieval | It is one possible transformation for selected queries |
| Query rewriting | Restating a query before retrieval | It is one possible transformation for selected queries |
| Query decomposition | Splitting a query into smaller requests | It is one possible transformation for selected queries |
| Chunk | A document segment used as a retrieval unit | Chunk size affects how much evidence each retrieved item carries |
| Fixed-size chunking | Splitting text into segments of a set size | The source warns that semantic chunking does not always beat it |
| Semantic chunking | Splitting text by meaning rather than only by size | Its assumed advantage is false in identifiable conditions |
| Embedding | A numeric representation used for retrieval | Its dimension enters index-memory arithmetic |
| Embedder | The component that creates embeddings | Fine-tuning it before locating a regression may target the wrong stage |
| Vector | One numeric embedding stored in the index | Per-vector bytes determine total index memory |
| Vector database | A store used to search embeddings | Naming it alone does not constitute a design decision |
| Bi-encoder | Encodes query and document separately for a comparison score | Separate encoding does not require shared weights in every architecture |
| Cross-encoder | Jointly processes query and candidate to score their interaction | Joint scoring has different caching and per-pair work from separate encoding |
| Best Matching 25 (BM25) | A lexical scorer whose saturation and length normalization matter | A candidate should explain both effects without notes |
| Saturation | A scoring effect where additional evidence stops adding the same gain | It is one BM25 behavior the screen can probe |
| Length normalization | Adjusting a lexical score for document length | It is another BM25 behavior the candidate should explain |
| Scorer | The rule that assigns relevance scores to candidates | It belongs in the how decision |
| Ranking | Ordering candidates by their relevance scores | It is a common deep-dive topic |
| Index | A search structure over the stored material | Its memory and recall at fixed latency require explicit trade-offs |
| Approximate index | An index that gives something up to search efficiently | The screen may ask what that sacrifice is |
| HNSW | A layered proximity-graph index | State level convention, precision, ID widths and storage scope |
| HNSW height | Random maximum sampled level | A logarithmic-order estimate is not an exact level count for any realized index |
| Per-vector storage subtotal | Payload and explicitly counted structural terms | Multiply by vector count, then budget omitted fixed/runtime terms separately |
| $`N_v`$ | Number of stored vectors in this guide | Not model-parameter count or necessarily number of documents |
| $`d`$ | Embedding dimension | Doubling it doubles an uncompressed fixed-precision vector payload |
| $`M`$ | HNSW graph parameter | Common base allocation uses $`2M`$ slots, not exactly that many live neighbors |
| Gate $`g(x)`$ | Retrieval action policy for request $`x`$ | Respect evidence and safety requirements, not confidence alone |
| Closed-book answer | An answer produced without retrieved material | Retrieval-augmented accuracy can fall below it in stated conditions |
| Retrieval-augmented accuracy | Accuracy when retrieved material is supplied | It can be worse than closed-book accuracy on high-popularity questions |
| Accuracy | How often answers meet the correctness target | A drop in it starts the debugging example |
| Source selection | Choosing which datastore source to trust or use | A bad choice can create a contradiction that the generator resolves silently |
| Candidate depth $`k^{\prime}`$ | Candidates passed to reranking | Tune candidate coverage and per-pair work |
| Evidence count $`k`$ | Items presented to generation | Actual tokens, support and placement matter, not count alone |
| Reranker | A component that reorders retrieved candidates | Its candidate depth can differ from the generator’s evidence depth |
| Rerank depth | The number of candidates considered by the reranker | It affects retrieval work and should be priced |
| Serialization | The format used to turn retrieved items into generator input | It belongs in the present decision |
| Prompt position | Where retrieved evidence is placed in the generator input | Evidence position can change whether the generator uses it well |
| Context | The retrieved material placed in the generator input | More context can cost tokens and make evidence harder to use |
| Context window | The available input space for the generator | A longer window is not an automatic fix for a recall problem |
| Prefill tokens | All input tokens processed before generation | Include instructions, query, evidence, labels and delimiters; caching changes billed/physical work |
| Attention dilution | The spread of model attention across more context | Extra retrieved items can make useful evidence harder to use |
| Lost-in-the-middle U-curve | Quality versus evidence position under a specified context | Not a universal U-shaped curve versus evidence count or context length |
| $`\mathop{\mathrm{P}}\nolimits@3`$ | Fraction of the first three results judged relevant | Not calibrated answer correctness or an abstention probability |
| Abstention threshold | The confidence boundary below which the system should decline to answer | It appears in the condition for preferring a second retrieval pass |
| Second retrieval pass | Another search step after an initial result | It can be a better use of budget than enlarging k |
| QPS | Incoming requests per second | Interacts with corpus size, per-query work, bandwidth, memory and concurrency |
| Latency | The time needed to answer a query | Index recall and reranking must fit a latency budget |
| Constraint | A limit or requirement that the design must satisfy | Changed constraints test whether the candidate can re-derive |
| Binding constraint | The requirement that most strongly shapes the design | The sizing phase should identify it before components are chosen |
| Freshness | How current the searchable material must be | It is a required sizing question in the design round |
| Accuracy bar | The required answer-quality target | It helps identify the binding system constraint |
| Memory | Resident data and runtime allocations | Depends on representation, replicas, caching, concurrency and implementation |
| Compute | Processing work/capacity for requests and maintenance | Depends on request rate and work per request, including corpus size |
| Node | One machine in a capacity plan | Ten versus two is an illustration, not a derived feasibility result |
| Hybrid retrieval | Combining retrieval approaches | The four-week plan asks candidates to say what it fixes |
| Reciprocal rank fusion | A method named for combining ranked retrieval results | The four-week plan asks candidates to explain what it fixes |
| Control flow | The rules that decide which retrieval steps run | Each method should be tied to a failure mode and a trade-off |
| Macro, micro, mezzo | Aggregate, stage and targeted diagnostic views | Organizes hypotheses; not guaranteed unique causal attribution |
| Evaluation | Measuring whether a system worked | It closes the design loop and supports regression localization |
| Regression | A decline from an earlier system result | The debugging round asks the candidate to find its stage |
| Failure attribution | Testing which changes cause the observed regression | Allow multiple causes, interactions and uncertainty |
| Derivation | Rebuilding an answer from its reasoning or arithmetic | It lets the candidate adapt when a constraint changes |
| Metric | A numeric measure of system behavior | It needs a denominator and evaluation definition |
| Denominator | The set of cases counted under a reported rate | Without it, a recall value cannot be interpreted |
| $`\mathop{\mathrm{Recall}}\nolimits@k`$ | Recall at a declared result depth | Specify task judgments or exact neighbors, eligibility, aggregation and zero-relevant policy |
| Judgment set | The examples used to decide what retrieval results are relevant | It supplies the denominator behind a recall claim |
| Relevance grading | The rule used to label degrees of relevance | A metric cannot be interpreted without it |
| Faithfulness | Agreement between an answer and its retrieved document | An answer can be faithful to a document that is wrong |
| Factuality | Agreement between an answer and what is actually true | It is a different failure from unfaithfulness |
| Grounding | Actual support of an answer by retrieved material | A citation does not by itself prove grounding |
| Post-hoc citation | A citation attached after the answer is produced | It may bear little relation to which source shaped the answer |
| Synthetic poisoned test document | Adversarial item in an isolated authorized copy | Never insert it into a production/shared datastore |
| Fine-tuning | Updating a model on additional training data | Doing it before attribution may spend effort on the wrong component |
| Component walk | Reciting a possible processing chain | Some architectures omit stages; explain choices and alternatives |


## Core mechanics
The source’s interview rubric favors explicit decisions over unexamined component lists. This is practice advice, not a measured causal estimate of which answer feature explains most hiring outcomes.
### What the Loop Actually Looks Like
The source describes four possible round types. Real loops can reorder, combine or omit them; no industry-wide frequency distribution is established here.
#### The screen
**What it is:** An illustrative $`30`$- to $`45`$-minute screening conversation. The source mentions hiring managers; interviewer roles and duration vary by employer.
**Why it exists:** It tests whether the candidate’s vocabulary is real. Example probes include the difference between a bi-encoder and cross-encoder, BM25 saturation, and the sacrifice made by an approximate index.
**What fails without the knowledge:** A candidate can name a term but cannot explain its behavior or trade-off. The book maps its core questions to this round.
#### The design round
**What it is:** An illustrative $`45`$- to $`60`$-minute design prompt, such as a support assistant or an assumed $`10^8`$-document workload at $`500\,\mathrm{queries\,s^{-1}}`$. These are constraints to analyze, not a demonstrated deployment or capacity guarantee.
**Why it exists:** It tests whether the candidate can turn constraints into explicit choices. The five-decision skeleton supports this round. Chapter 41 drills it seven times.
**What fails without the structure:** The answer becomes a component list. It does not expose which choices respond to the binding constraint.
#### The deep dive
**What it is:** A specialist pushes one component until the candidate runs out of derivation. Common targets are indexing from Chapters 15 to 17, ranking from Chapters 18 to 22, and evaluation from Chapters 32 to 34.
**Why it exists:** It tests depth rather than broad vocabulary. The book calibrates its senior questions to this round.
**What fails without derivation:** Memorized conclusions stop working when the specialist changes a number or asks where a term came from.
#### The debugging round
**What it is:** A debugging exercise in which a system regressed and the candidate chooses evidence and interventions. The source’s claims about increasing frequency and low preparation are not quantified here.
**Why it exists:** It tests end-to-end failure attribution. Chapter 34 supplies the macro, micro, mezzo method. Section $`41.6`$ supplies the worked instance.
**What fails without attribution:** The candidate jumps to a favorite fix before finding the broken stage.
Behavioral and project-history rounds can sit beside these four. The source pages do not cover them.
### What the Interview Actually Tests
The source proposes three general evaluation signals plus one retrieval-specific signal. Treat these as its practice rubric rather than a rule used by every interview loop.
#### Signal 1: Decisions instead of components
**What it is:** The candidate chooses among alternatives. The candidate does not merely say, “chunk, embed, store, and retrieve the top 5.”
**Why it matters:** Naming common components is not enough to expose choices. Not every architecture embeds, chunks or reranks: lexical-only, whole-document and other designs differ. State why each included stage is needed and when retrieval should be bypassed or required by policy.
**What fails without it:** The answer cannot express the system’s highest-leverage choice. Section $`3.1`$ develops a complete decision-based alternative.
#### Signal 2: Numbers with derivations
**What it is:** Define the scope before giving a number. In the worked HNSW example, $`d=768`$, $`M=16`$, fp32 vectors use $`4\,\mathrm{B/dimension}`$, and four-byte link IDs with allocated base-layer capacity $`2M`$ contribute $`8M`$ bytes per vector.[\[5\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[6\]](../part-05-indexing/16_compression_and_index_economics.md)
The vector-plus-base-link subtotal is $`b_{\mathrm{sub}}=(4d+8M)\,\mathrm B=3{,}200\,\mathrm B`$ per vector. At $`N_v=10^7`$, it is $`32\,\mathrm{GB}`$ in decimal units. The $`128\,\mathrm B`$ base-link allocation stays fixed only when quantizing vectors without changing links. It is not inherently incompressible. Upper layers, labels/IDs, codebooks, metadata, source text, allocation, replicas and runtime are excluded; this is not total index/process memory.
**Why it matters:** Appendix A supplies formulas as vocabulary. The chapters explain where the terms come from.
**What fails without it:** A number without definitions, units, precision and omitted terms cannot be checked or safely transferred. Be ready to derive it rather than relying on an unattributed remembered result.
#### Signal 3: Re-derivation under a changed constraint
**What it is:** Re-evaluate when constraints change: for example, $`d=1{,}536`$, a $`50`$-fold corpus increase, or half the latency budget. A single-machine requirement may become infeasible; do not promise it survives simply by recomputing one payload term.
**Why it matters:** Re-derivation tests transfer to changed assumptions. Roughly $`30\,\mathrm s`$ is the source’s practice target, not a validated hiring signal or guaranteed time for a complete capacity plan.
**What fails without it:** A candidate who memorized an architecture must start over while the interviewer watches.
#### Signal 4: Technique cost
**What it is:** The candidate says when a method harms the system or spends a resource.
**Why it matters:** This retrieval-specific signal is the one the source leans on hardest. RAG folklore assumes retrieval helps, more context is better, citations imply grounding, and semantic chunking beats fixed-size chunking. The source says each assumption is measurably false in identifiable conditions.
**What fails without it:** The answer repeats optimistic defaults. It does not identify the conditions that reverse the recommendation. A candidate who knows those conditions can give an answer nobody else in the loop gave.
### The Answer Skeleton
The schematic $`y=\mathop{\mathrm{GEN}}\nolimits_\theta(x,\mathop{\mathrm{RET}}\nolimits(x,D))`$ connects query $`x`$, datastore $`D`$, retrieval and generation. When, where, what, how and present organize useful design questions. They are not a theorem enumerating all independent free parameters: eligibility, ingestion, training, decoding, security, randomness, budgets and implementation choices are also hidden by this abstraction.[\[1\]](../part-01-landscape/03_a_repeatable_framework_for_any_rag_design_question.md)
Use the five decisions in order. Say the order aloud.
#### 1. When
**What it is:** State the retrieval gate $`g(x)`$ and its evidence policy. It decides whether request $`x`$ triggers retrieval; apparent model confidence alone does not establish truth or permit bypassing required sources.
**Why it exists:** In the cited Mallen et al. setting, retrieval flipped roughly $`10\%`$ of the otherwise-correct closed-book subset to wrong. High-popularity results are model/task-specific, not a universal rate or a rule to bypass retrieval for every familiar fact.[\[7\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
**What fails without it:** Always retrieve becomes an unexamined default. The design cannot express the option that retrieval should not run.
**Stated cost:** Retrieval can reverse an otherwise-correct answer and consumes work. Price matched retrieval/no-retrieval actions under support and safety requirements; naming a gate alone does not demonstrate production experience.
#### 2. Where
**What it is:** Name the datastore sources. State what happens when two sources disagree.
**Why it exists:** A datastore is not a single undifferentiated thing. Source choice controls which evidence reaches the generator. Section $`25.3`$ develops source selection.
**What fails without it:** The generator receives a silent contradiction and resolves it on its own.
**Stated cost:** Wrong source selection can cause unsupported answers, contradictions, security problems, delay or unnecessary expense. Which consequence dominates depends on the application; contradiction is not always more important than latency.
#### 3. What
**What it is:** Choose the query that the retriever sees. Options named in the source are compression, expansion, rewriting, and decomposition.
**Why it exists:** The useful retrieval request may differ from the string typed by the user. Chapter 24 covers these transformations.
**What fails without it:** The design assumes one transformation fits every query. It also fails to say which queries receive the change.
#### 4. How
**What it is:** Choose a scorer, index and candidate depth $`k^{\prime}`$. Define the candidate-recall target, eligible corpus, reference judgments or exact neighbors, and latency/work budget.
**Why it exists:** This decision ties implementation choices to arithmetic. The candidate should discuss index memory, recall at a fixed latency, and rerank depth.
**What fails without it:** The answer names an index but does not price its memory, recall, or latency behavior. Chapters 15 to 22 supply the relevant stack.
#### 5. Present
**What it is:** Choose evidence from the $`k^{\prime}`$ candidates for a final count $`k`$, then define ordering, truncation, serialization and placement. Price actual tokens after deduplication, compression and formatting, not $`k`$ alone.
**Why it exists:** Retrieved material must be shaped before the generator sees it. Chapters 30 and 31 cover these choices.
**What fails without it:** The generator receives too much, badly ordered, badly formatted, or poorly positioned material.
**Stated cost:** Assembly can change support, latency and output quality. The source does not establish that presentation is universally the least-discussed or most-expensive decision.
#### Close the loop
**What it is:** Define success metrics, uncertainty, versioned traces and diagnostic interventions. The five-decision map helps organize hypotheses; it cannot guarantee one unique cause for every regression.[\[2\]](../part-10-evaluation/34_evaluating_the_system.md)
**Why it exists:** A design needs both evaluation and diagnosis. Chapter 34 provides the localization approach.
**What fails without it:** The candidate lacks evidence that the system meets its objectives or a way to test competing causes of a later regression. Several stages and interactions may be involved.
Chapter 41 applies the full skeleton seven times against seven binding constraints.
### What Separates Core, Senior, and Staff Answers
The source illustrates different answer shapes with the top-$`k`$ question. The level labels are practice rubrics, not proof that knowledge amount never matters or that one answer shape guarantees a hiring level.


| Level | What the answer does |
| --- | --- |
| Core | Name candidate inclusion, actual input-token cost and answer-use risks separately. In a nested list, larger $`k`$ preserves existing candidates, but generated-answer quality need not increase. |
| Senior | With twenty full $`512`$-token chunks, evidence contributes $`20\times512=10{,}240`$ tokens. This excludes instructions, query, source labels and delimiters. Measure prefill and placement; the positional U-curve is not a universal quality-versus-$`k`$ law. |
| Staff | Separate reranker depth $`k^{\prime}`$ and evidence count $`k`$. Do not compare $`\mathop{\mathrm{P}}\nolimits@3`$ with an answer-abstention threshold. Use calibrated answer correctness and support requirements with decision costs; buy another pass only if its marginal value exceeds its cost. |


**Why the distinction matters:** Separate objectives, quantify a constraint and name evidence that would reverse the decision. Extra length is not a substitute for those tests.
**What fails without it:** More detail can still leave the premise untested. The answer never exposes different objectives or reversal conditions.
### Timing a 45-Minute Design Round
The source’s independent pacing ranges sum to $`35`$ through $`48`$ minutes, so they do not themselves define one $`45`$-minute plan. The following illustrative allocation sums exactly to $`45\,\mathrm{min}`$; adapt it to the interviewer rather than treating every phase as fixed.


| Phase | Time | What you are doing |
| --- | --- | --- |
| Requirements and sizing | $`5\,\mathrm{min}`$ | Ask about corpus size, QPS, latency, freshness, and the accuracy bar. Ask which constraint binds. |
| Decision 1: when | $`3\,\mathrm{min}`$ | State the gate. Price retrieval rather than assuming it. |
| Decisions 2 to 4 | $`14\,\mathrm{min}`$ | Cover sources, query transformation, scorer, and index. Derive memory or latency. |
| Decision 5: present | $`7\,\mathrm{min}`$ | Cover ordering, truncation, format, and position. |
| Evaluation | $`6\,\mathrm{min}`$ | Name metrics and uncertainty; test stages and interactions rather than promise one cause. |
| Constraint changes | $`8\,\mathrm{min}`$ | Re-derive answers after interviewer follow-ups. |
| Wrap-up | $`2\,\mathrm{min}`$ | Summarize the five decisions and invite feedback. |


#### Requirements and sizing
**What it is:** Gather corpus size, query rate, latency, freshness, and accuracy requirements before selecting components.
**Why it exists:** Storage scales with vector count $`N_v`$, representation, graph/ID overhead, replication and headroom. At request rate $`Q`$, compute, bandwidth, I/O and concurrency also depend on per-query work, corpus size, dimensions, filtering, caching and candidate depth. Size both axes using measured service demand, not a strict memory-versus-compute separation.
**What fails without it:** A node count based on one aggregate number can over- or underprovision. The source’s ten-versus-two-node example is illustrative; two-node feasibility is not derived here. Capacity and throughput bounds still need hardware measurements, placement, availability and tail-latency headroom.
### Preparation Plans
#### One Week
**What to do:** Use the deliberately narrow one-week route: Chapters 3 and 34 first, followed by 1, 15, 18, 22, 25, 30 and 32. Fill role-critical gaps; this schedule is not a claim that a week suffices for every candidate.
Use the last two days of that week for timed practice on Sections $`41.1`$ and $`41.6`$. These drills are part of the week, not two additional days, and completing them does not certify deployment readiness.
**Why this plan exists:** It prioritizes a partial map that supports reasoning.
**What fails without it:** Skimming a complete map does not create the same ability to derive an answer.
#### Four Weeks
At roughly $`2\,\mathrm{h/day}`$ over $`28`$ study days, the four-week plan budgets $`56\,\mathrm h`$. This assumes daily study including weekends. It is an overview/practice plan, not a guarantee of deep mastery of every chapter.
#### Week 1: Foundation
**Coverage:** Parts I to III, Chapters 1 to 11, and Appendix A.
**Practice:** Close the book after each chapter and re-derive the key formulas on paper.
**Targets:** Explain the five-decision lens, conditional retrieval harm, and scoped memory arithmetic for specified $`N_v`$, $`d`$ and $`M`$. Do not treat one HNSW subtotal as a formula for every index family.
#### Week 2: The retrieval stack
**Coverage:** Parts IV to VI, Chapters 12 to 23.
**Practice:** Work every example with a calculator.
**Targets:** Derive the level distribution and scoped storage under a stated HNSW convention. For independent sampled levels with $`M>1`$, $`\Pr(\ell\ge j)=M^{-j}`$ for nonnegative integer $`j`$; maximum height is random, typically of logarithmic order, not an exact number fixed by $`N_v`$. Explain BM25 saturation, length normalization, hybrid retrieval and RRF with their assumptions.[\[5\]](../part-05-indexing/15_approximate_nearest_neighbor_search.md)[\[8\]](../part-06-retrieval-ranking/18_classical_ir_you_are_expected_to_know.md)[\[9\]](../part-06-retrieval-ranking/21_learned_sparse_and_multi_vector_retrieval.md)
#### Week 3: Control flow and evaluation
**Coverage:** Parts VII to X, Chapters 24 to 34.
**Practice:** For every control-flow method, write one sentence naming the failure it addresses and what it trades away.
**Targets:** Run macro, then micro, then mezzo on a described regression. Produce a ranked list of suspects.
#### Week 4: Trust, scale, and drills
**Coverage:** Use Week 4 for Chapters 35 through 41, one overview chapter per day. Within the $`2\,\mathrm{h/day}`$ budget, allocate $`120`$ minutes to reading/practice on the first four days and $`75`$ minutes to chapter work on each of the last three days.
**Practice:** On each of those last three days, use the remaining $`45\,\mathrm{min}`$ for one mock, covering Sections $`41.1`$, $`41.3`$ and $`41.6`$. These overlap the reading week rather than adding three days. Week 4 then has $`11.75\,\mathrm h`$ of chapter work plus $`2.25\,\mathrm h`$ of mocks, totaling $`14\,\mathrm h`$.
**Targets:** Compare each attempt against the chapter. Record which constraints appeared late and which numbers could not be derived. Those two lists become the remaining study plan.
#### Three Months
**What to do:** Read linearly at up to three numbered subsections per day while building an owned test system. The source contents list $`247`$ main-chapter subsections, requiring $`\lceil247/3\rceil=83`$ reading days at that rate. A declared $`90`$-day planning horizon leaves seven other days; three calendar months need not be exactly ninety days. Write relevance judgments and measure $`\mathop{\mathrm{Recall}}\nolimits@k`$ with explicit eligibility, reference set, aggregation and zero-relevant policy.
Break only an isolated, authorized test copy: remove one third of its indexed material, shuffle context order, and insert one synthetic adversarial document. Keep real sensitive data out, preserve a recoverable baseline, and never poison or destructively test a production/shared datastore. Check which tests detect each intervention and its side effects.
**Why this plan exists:** Candidates who have watched their own system fail answer debugging questions differently from candidates who have only read about failure.
**What fails without it:** Passive knowledge does not supply observed failure behavior or tested evaluation.
### How to Drill a Section
Passive reading does not build a derivation habit. Use this loop for every section that contains arithmetic.
1. Follow the derivation once, tracking definitions and units.
2. Cover the page and reproduce the key relation from memory, including assumptions.
3. Change a constraint: double $`d`$, halve the latency budget, or multiply $`N_v`$ by $`50`$. Recompute and check feasibility rather than forcing the old architecture to fit.
4. Answer the three practice questions aloud, first targeting $`90\,\mathrm s`$ and then a five-minute explanation. These are drill targets, not competence cutoffs.
5. Check the takeaways against the derivation and state what evidence would change the conclusion.
**Why it exists:** Changing a constraint rehearses a useful interview and engineering skill; no particular interviewer follow-up is guaranteed.
**What fails without it:** Familiarity with a derivation can be mistaken for the ability to reproduce and adapt it.
Use Appendix B when you want to test retention. Its questions are held apart from their answers. The chapters deliberately do not hold them apart.
### Common Failure Modes
#### The component walk
**Failure:** The candidate recites ingest, chunk, embed, index, retrieve, rerank, and generate.
**Why it fails:** The listed stages are common, not universal. A lexical retriever need not embed, a system need not rerank, and whole-document retrieval need not chunk. State the actual architecture, its alternatives and its evidence policy.
#### Never questioning whether to retrieve
**Failure:** The candidate treats always retrieve as the neutral default.
**Why it fails:** Always retrieving is a policy with task-dependent benefits and harms, not a neutral default. Conversely, bypassing retrieval may violate currentness or support requirements. Evaluate the policy; no interview-frequency claim is established here.
#### Treating context as free
**Failure:** Automatically increasing final evidence count $`k`$ or the context window to fix any recall problem confuses candidate inclusion with answer use.
**Why it fails:** More candidates in a nested set can improve answer inclusion without improving generated answers. Lost-in-the-middle results concern position under specified models and tasks; longer windows need not fix them, but no universal U-shaped quality-versus-$`k`$ law follows.[\[4\]](../part-09-context-assembly/30_lost_in_the_middle.md)
#### Quoting a metric without its denominator
**Failure:** Saying $`\mathop{\mathrm{Recall}}\nolimits@k=0.9`$ without defining $`k`$, judged relevance versus exact-neighbor reference, eligible corpus, query aggregation, grading and the policy for no relevant items. Approximate-neighbor recall and task relevance recall are not interchangeable.[\[10\]](../part-10-evaluation/32_evaluating_retrieval.md)
**Why it fails:** The number cannot be interpreted. Chapter 32 derives the metrics so the candidate can state their denominators precisely.
#### Confusing faithfulness with factuality
**Failure:** The candidate treats agreement with retrieved text as proof that the answer is true.
**Why it fails:** An answer can be perfectly faithful to a retrieved document that is wrong. Chapters 4 and 33 keep the two failures and their fixes separate.
#### Proposing a fix before attributing the failure
**Failure:** The candidate hears that accuracy dropped and immediately proposes fine-tuning the embedder.
**Why it fails:** A familiar fix is not a diagnosis. Use matched tests and targeted interventions to separate workload, corpus, labeling, retrieval, assembly and generation changes. Several stages or interactions may contribute; the diagnostic method is not a proof of one unique cause.[\[2\]](../part-10-evaluation/34_evaluating_the_system.md)
#### Citations as an attribution story
**Failure:** The candidate promises source citations and assumes that they prove grounding.
**Why it fails:** A post-hoc citation may genuinely support a claim, but attaching it does not by itself prove the model actually used that source when generating. Check evidential entailment separately from historical use, world truth and citation presence.[\[11\]](../part-09-context-assembly/31_attribution_and_citation.md)
## Diagrams
The manifest records zero figures and zero tables for physical source pages 19 to 23. The source contains no numbered figure or table captions to normalize.
The two unnumbered tabular layouts are preserved under Core mechanics. They compare answer levels and give the 45-minute pacing plan.
## Whiteboard pack
### Numbered drawing order
1. Put the design prompt beside explicit corpus size, request rate, latency, freshness, evidence and safety requirements.
2. Draw When, Where, What, How and Present as an organizing lens, not an exhaustive parameter theorem.
3. Under When, write $`g(x)`$, evidence requirements and the cost of each action.
4. Under Where, record eligible sources, permissions, conflicts and provenance.
5. Under What, record justified compression, expansion, rewriting or decomposition.
6. Under How, write scorer, index, $`k^{\prime}`$, scoped memory, recall reference and latency.
7. Under Present, write $`k`$, actual tokens, order, truncation, format and position.
8. Add evaluation and debugging, with arrows for controlled tests of multiple hypotheses and interactions.
9. End with separate objectives, a changed constraint, costs and a reversal condition.
### 90-100 word script
A strong RAG answer starts with constraints, not a component list. I first size the corpus, query rate, latency, freshness, and accuracy bar. Then I make five decisions in order. I decide when retrieval should run, where evidence comes from, what query reaches the retriever, how scoring and indexing work, and how evidence reaches the generator. I attach a number and a cost to each choice. Then I close the loop with evaluation and stage-level debugging. If a constraint changes, I re-derive the design. That decision-first structure shows judgment, arithmetic, trade-offs, and failure attribution.
## Interview traps
### 1. Why is a component walk a weak design answer?
Many designs include some of ingest, chunk, embed, index, retrieve, rerank and generate, but not every design contains every stage. Reciting a possible chain does not expose constraints, alternatives or when evidence retrieval is required or unnecessary.
### 2. When would you not use retrieval?
In the cited study, roughly $`10\%`$ of otherwise-correct closed-book answers were flipped to wrong by retrieval. Preserve that denominator, task and model; it is not ten percent of every request or a general instruction to bypass evidence for popular facts.[\[7\]](../part-01-landscape/01_what_a_rag_interview_actually_tests.md)
### 3. What should follow any number you state?
State the source or derivation, symbol definitions, units, scope and omitted terms. When a constraint changes, recompute and check feasibility. Recall of a number and ability to derive it are different skills.
### 4. How should you answer “How do you choose top-k?”
Separate candidate depth $`k^{\prime}`$ and evidence count $`k`$. Retrieval precision such as $`\mathop{\mathrm{P}}\nolimits@3`$ measures document relevance, not calibrated probability that the answer is correct. Do not compare it directly with an answer-abstention threshold. Test coverage/support and calibrated answer correctness against decision costs; a second pass is justified only when its expected marginal value exceeds its cost.[\[12\]](../part-03-prompting/11_abstention_and_calibration.md)[\[3\]](../part-06-retrieval-ranking/22_reranking.md)[\[13\]](../part-07-query-control/25_routing_and_adaptive_retrieval.md)
### 5. Why do citations not prove grounding?
A post-hoc citation can be attached without showing that the cited source shaped the answer. Faithfulness to retrieved text, factual correctness, citation presence, and actual grounding must remain separate claims.
## Key numbers


| Number or range | Context | Meaning or trade-off |
| --- | --- | --- |
| Four possible round types | Source practice model | Order, inclusion, interviewer and frequency vary; not an industry survey |
| $`30`$ to $`45\,\mathrm{min}`$ | Illustrative screen | Employer-specific, not a measured universal duration |
| $`45`$ to $`60\,\mathrm{min}`$ | Illustrative design round | Adapt pacing to the actual round |
| $`10^8`$ documents; $`500\,\mathrm{queries\,s^{-1}}`$ | Assumed design prompt | Size capacity and per-query service work, not a demonstrated deployment |
| 7 drills | Chapter 41 | Applies the five-decision skeleton seven times |
| Three general plus one retrieval-specific signal | Source practice rubric | Not a measured hiring rule |
| Top 5 | Weak component-list example | A retrieval depth stated without a choice or derivation has little discriminative value |
| $`d=768`$, $`M=16`$ | HNSW example | Assumes fp32 vectors and four-byte allocated base-link IDs |
| $`(4d+8M)\,\mathrm B=3{,}200\,\mathrm B`$ | Per-vector subtotal | Vector plus allocated base links only |
| $`N_v=10^7`$; $`32\,\mathrm{GB}`$ | Scoped storage subtotal | Decimal units; excludes other structure, metadata and runtime |
| $`128\,\mathrm B`$ | Base links per vector | Unchanged in vector-only compression; not inherently incompressible |
| $`d=1{,}536`$ | Changed dimension | Recompute vector payload and other terms separately |
| $`50`$-fold corpus increase | Changed constraint | A single-machine target can become infeasible |
| Half the latency budget | Changed constraint | Recompute feasibility with measured service demand |
| About $`30\,\mathrm s`$ | Arithmetic practice target | Not a guaranteed time for complete sizing or a validated hiring signal |
| Five decisions | Organizing lens | Not an exhaustive free-parameter theorem |
| Decisions versus components | Interview answer shape | Decisions expose judgment, while a standard component list does not |
| Retrieve versus do not retrieve | When decision | Retrieval can help, but it is neither free nor neutral |
| Roughly $`10\%`$ | Cited otherwise-correct subset | Conditional retrieval harm, not all production queries |
| Source-selection errors | Where decision | Contradiction, security, expense and latency trade-offs depend on application |
| $`k^{\prime}`$ to $`k`$ | Candidate versus evidence depth | Different objectives; actual context tokens are separate |
| Larger $`k`$ | Top-depth trade-off | Nested-set inclusion can improve without improved generated answers |
| $`20\times512=10{,}240`$ tokens | Evidence-only contribution | Not the entire prompt or a measured prefill duration |
| $`\mathop{\mathrm{P}}\nolimits@3`$ | Document relevance metric | Not directly comparable with an answer-abstention threshold |
| Two distinct depths | Reranking versus generation | Tune coverage, answer support, calibrated correctness and work |
| More versus useful context | Assembly trade-off | Extra evidence can help or distract; measure position and token budget |
| $`45\,\mathrm{min}`$ | Illustrative exact pacing plan | The earlier independent ranges span thirty-five through forty-eight minutes |
| $`5\,\mathrm{min}`$ | Requirements and sizing | Gather corpus size, QPS, latency, freshness, and accuracy bar |
| $`3\,\mathrm{min}`$ | Decision 1 | State and price the retrieval gate |
| $`14\,\mathrm{min}`$ | Decisions 2 to 4 | Cover sources, query transformation, scorer, index, and arithmetic |
| $`7\,\mathrm{min}`$ | Decision 5 | Cover ordering, truncation, format, and position |
| $`6\,\mathrm{min}`$ | Evaluation | Name measurements, uncertainty and diagnostic interventions, allowing interactions |
| $`8\,\mathrm{min}`$ | Constraint changes | Re-derive after interviewer follow-ups |
| $`2\,\mathrm{min}`$ | Wrap-up | Summarize the five decisions and invite feedback |
| Two workload axes | Capacity planning | Both corpus and query work affect multiple resources |
| Ten nodes versus two | Sizing illustration | No two-node feasibility or universal savings factor is derived |
| One week | Narrow preparation route | Nine selected chapters plus two drill days; not a readiness certificate |
| Last two days within the week | Timed design practice | Not two additional days |
| $`28\times2=56\,\mathrm h`$ | Four-week plan | Daily study including weekends; an overview/practice budget |
| Week 1, Parts I to III, Chapters 1 to 11 | Four-week foundation | Re-derive formulas and learn the five decisions, retrieval harm, and index sizing |
| Week 2, Chapters 12 to 23 | Retrieval stack | Derive random HNSW level laws and scoped bytes, not a deterministic height |
| Week 3, Chapters 24 to 34 | Control and evaluation | Diagnostic views guide tests; causes can interact |
| Week 4, Chapters 35 to 41 | One overview chapter daily | Last three days share their two-hour budget with mocks |
| $`3\times45=135\,\mathrm{min}`$ | Mocks within Week 4 | Two and a quarter hours, leaving eleven and three-quarter hours of chapter work |
| $`\lceil247/3\rceil=83`$ reading days | At up to three subsections daily | A declared ninety-day horizon leaves seven other days; calendar months vary |
| Remove one third of a test index | Authorized isolated fault injection | Never delete shared/production data as a practice exercise |
| 5 drill steps | Arithmetic practice | Read, reproduce, change, speak, and verify |
| Double $`d`$, halve latency, or scale $`N_v`$ by $`50`$ | Practice perturbations | Re-derive scope, cost and feasibility |
| $`90\,\mathrm s`$, then five minutes | Spoken practice targets | Not validated competence thresholds |
| Seven listed possible stages | Component-list example | Not every design contains all seven |
| $`\mathop{\mathrm{Recall}}\nolimits@k=0.9`$ | Incomplete metric claim | Needs depth, reference, eligibility, aggregation and zero-relevant policy |


[00-START-HERE](../00-START-HERE.md) · [01-READING-ROUTES](../01-READING-ROUTES.md) · [00\_04\_how\_to\_use\_this\_book](00_04_how_to_use_this_book.md) · [00\_06\_for\_practicing\_engineers](00_06_for_practicing_engineers.md)
