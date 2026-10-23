# The RAG Interview
Unofficial study notes on *The RAG Interview* by Hao Hoang. No affiliation with or endorsement by the author is claimed, and rights in the source work remain with the author. See [NOTICE.md](../NOTICE.md).
## Book index
This Notion edition links forty-one chapters, nine front-matter units and seven appendices: $`41+9+7=57`$ content units. This start page, Reading Routes and Source Index are three additional navigation pages, making sixty indexed pages. Links open Notion pages; the titles and source page labels do not imply downloadable Markdown files or current Notion page numbers. Use the route guide for selected study and the source index for original-edition spans and declared prose-only reading estimates.
[01-READING-ROUTES](01-READING-ROUTES.md) · [98-SOURCE-INDEX](98-SOURCE-INDEX.md)
## Roadmap


| Part | Chapters | Focus |
| --- | --- | --- |
| Front matter | $`00.01`$–$`00.09`$ | You learn the scope, study method, notation, and intended use of the edition. |
| I. The RAG Interview Landscape | 1–3 | You frame the problem, compare alternatives, and structure a design answer. |
| II. The Generator Side | 4–8 | You examine generator capabilities, possible failure causes and model-capacity economics without assuming one unique owner for every error. |
| III. Prompting and Context Construction | 9–11 | You control instructions and assess prompt sensitivity, abstention, and calibration. |
| IV. Representing What You Retrieve | 12–14 | You choose representations, retrieval units, and ways to preserve document structure. |
| V. Indexing and Vector Search | 15–17 | You choose and operate an index within recall, latency, memory, and update budgets. |
| VI. Retrieval and Ranking | 18–23 | You compare retrieval families and ranking stages by quality and serving cost. |
| VII. Query Understanding and Control Flow | 24–26 | You reformulate queries, route requests, and control repeated retrieval. |
| VIII. Training the RAG System | 27–29 | You decide what to train and how to obtain useful supervision. |
| IX. Generation and Context Assembly | 30–31 | You manage context position and check the evidence behind generated claims. |
| X. Evaluation | 32–34 | You measure retrieval and generation separately and together, then test regression hypotheses and interactions. |
| XI. Trust, Credibility, and Adversarial Robustness | 35–36 | You evaluate credibility signals, provenance and access boundaries without treating authenticity as proof of truth or a security guarantee. |
| XII. Scaling, Advanced Variants, and Design Drills | 37–41 | You apply the earlier decisions to scale, federation, multimodality, graphs, and design drills. |
| Appendices | A1–A7 | You look up formulas, test yourself, review designs, and prepare system documentation. |


## Contents
### Front matter
- [00\_01\_title\_page](front-matter/00_01_title_page.md)
- [00\_02\_contents](front-matter/00_02_contents.md)
- [00\_03\_preface](front-matter/00_03_preface.md)
- [00\_04\_how\_to\_use\_this\_book](front-matter/00_04_how_to_use_this_book.md)
- [00\_05\_for\_interview\_candidates](front-matter/00_05_for_interview_candidates.md)
- [00\_06\_for\_practicing\_engineers](front-matter/00_06_for_practicing_engineers.md)
- [00\_07\_notation\_and\_symbols](front-matter/00_07_notation_and_symbols.md)
- [00\_08\_acknowledgments](front-matter/00_08_acknowledgments.md)
- [00\_09\_about\_the\_author](front-matter/00_09_about_the_author.md)
### Part I. The RAG Interview Landscape
- [01\_what\_a\_rag\_interview\_actually\_tests](part-01-landscape/01_what_a_rag_interview_actually_tests.md)
- [02\_the\_competing\_answers\_to\_the\_same\_problem](part-01-landscape/02_the_competing_answers_to_the_same_problem.md)
- [03\_a\_repeatable\_framework\_for\_any\_rag\_design\_question](part-01-landscape/03_a_repeatable_framework_for_any_rag_design_question.md)
### Part II. The Generator Side
- [04\_what\_the\_llm\_brings\_and\_what\_it\_breaks](part-02-generator/04_what_the_llm_brings_and_what_it_breaks.md)
- [05\_data\_privacy\_and\_the\_legal\_surface](part-02-generator/05_data_privacy_and_the_legal_surface.md)
- [06\_scaling\_laws\_and\_the\_economics\_of\_retrieval\_vs\_parameters](part-02-generator/06_scaling_laws_and_the_economics_of_retrieval_vs_parameters.md)
- [07\_in\_context\_learning\_the\_mechanism\_rag\_rides\_on](part-02-generator/07_in_context_learning_the_mechanism_rag_rides_on.md)
- [08\_reading\_the\_machine\_circuits\_induction\_heads\_and\_attribution](part-02-generator/08_reading_the_machine_circuits_induction_heads_and_attribution.md)
### Part III. Prompting and Context Construction
- [09\_prompting\_for\_retrieval\_augmented\_systems](part-03-prompting/09_prompting_for_retrieval_augmented_systems.md)
- [10\_prompt\_sensitivity](part-03-prompting/10_prompt_sensitivity.md)
- [11\_abstention\_and\_calibration](part-03-prompting/11_abstention_and_calibration.md)
### Part IV. Representing What You Retrieve
- [12\_text\_representation](part-04-representations/12_text_representation.md)
- [13\_chunking\_and\_granularity](part-04-representations/13_chunking_and_granularity.md)
- [14\_beyond\_plain\_text\_tables\_layout\_documents](part-04-representations/14_beyond_plain_text_tables_layout_documents.md)
### Part V. Indexing and Vector Search
- [15\_approximate\_nearest\_neighbor\_search](part-05-indexing/15_approximate_nearest_neighbor_search.md)
- [16\_compression\_and\_index\_economics](part-05-indexing/16_compression_and_index_economics.md)
- [17\_operating\_a\_vector\_store](part-05-indexing/17_operating_a_vector_store.md)
### Part VI. Retrieval and Ranking
- [18\_classical\_ir\_you\_are\_expected\_to\_know](part-06-retrieval-ranking/18_classical_ir_you_are_expected_to_know.md)
- [19\_learning\_to\_rank](part-06-retrieval-ranking/19_learning_to_rank.md)
- [20\_dense\_retrieval](part-06-retrieval-ranking/20_dense_retrieval.md)
- [21\_learned\_sparse\_and\_multi\_vector\_retrieval](part-06-retrieval-ranking/21_learned_sparse_and_multi_vector_retrieval.md)
- [22\_reranking](part-06-retrieval-ranking/22_reranking.md)
- [23\_generative\_retrieval](part-06-retrieval-ranking/23_generative_retrieval.md)
### Part VII. Query Understanding and Control Flow
- [24\_query\_reformulation](part-07-query-control/24_query_reformulation.md)
- [25\_routing\_and\_adaptive\_retrieval](part-07-query-control/25_routing_and_adaptive_retrieval.md)
- [26\_iterative\_recursive\_and\_agentic\_retrieval](part-07-query-control/26_iterative_recursive_and_agentic_retrieval.md)
### Part VIII. Training the RAG System
- [27\_fine\_tuning\_the\_generator](part-08-training/27_fine_tuning_the_generator.md)
- [28\_training\_the\_retriever](part-08-training/28_training_the_retriever.md)
- [29\_bootstrapping\_training\_data](part-08-training/29_bootstrapping_training_data.md)
### Part IX. Generation and Context Assembly
- [30\_lost\_in\_the\_middle](part-09-context-assembly/30_lost_in_the_middle.md)
- [31\_attribution\_and\_citation](part-09-context-assembly/31_attribution_and_citation.md)
### Part X. Evaluation
- [32\_evaluating\_retrieval](part-10-evaluation/32_evaluating_retrieval.md)
- [33\_evaluating\_generation](part-10-evaluation/33_evaluating_generation.md)
- [34\_evaluating\_the\_system](part-10-evaluation/34_evaluating_the_system.md)
### Part XI. Trust, Credibility, and Adversarial Robustness
- [35\_source\_credibility](part-11-trust-robustness/35_source_credibility.md)
- [36\_provenance\_and\_adversarial\_robustness](part-11-trust-robustness/36_provenance_and_adversarial_robustness.md)
### Part XII. Scaling, Advanced Variants, and Design Drills
- [37\_latency\_cost\_and\_systems](part-12-systems-design/37_latency_cost_and_systems.md)
- [38\_distributed\_and\_federated\_rag](part-12-systems-design/38_distributed_and_federated_rag.md)
- [39\_multimodal\_rag](part-12-systems-design/39_multimodal_rag.md)
- [40\_graph\_rag](part-12-systems-design/40_graph_rag.md)
- [41\_end\_to\_end\_design\_drills](part-12-systems-design/41_end_to_end_design_drills.md)
### Appendices
- [A1\_formula\_sheet](appendices/A1_formula_sheet.md)
- [A2\_question\_bank](appendices/A2_question_bank.md)
- [A3\_design\_checklists](appendices/A3_design_checklists.md)
- [A4\_the\_rag\_card\_and\_index\_datasheet](appendices/A4_the_rag_card_and_index_datasheet.md)
- [A5\_annotated\_reading\_list](appendices/A5_annotated_reading_list.md)
- [A6\_notation\_quick\_reference](appendices/A6_notation_quick_reference.md)
- [A7\_glossary](appendices/A7_glossary.md)
