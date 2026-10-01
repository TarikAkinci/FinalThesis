# Final Thesis: Cross-Cultural Disagreement in Offensive Language Judgments

# Paper Summaries

- **Quantifying the Persona Effect in LLM Simulations** ✔️
    
    Year: 2024
    
    Author(s): Tiancheng Hu, Nigel Collier
    
    Main Message: 
    
    - General sociodemographic variables explain less than 10% of human rating differences in subjective NLP datasets, so prompting LLMs with specific human personas slightly improves the model but is highly limited.
    
    Experiment:
    
    Linear, mixed-effect regression to analyse 10 subjective NLP datasets (with unaggregated annotations and annotator persona variables) and the ANES 2012 public opinion survey.
    Prompted 6 different 70-billion LLMs like GPT-4, Llama-2-70b, etc. with and without detailed persona profiles to see if models could accurately simulate human ratings
    
    Questions:
    
    1. How much variance in human annotation could persona variables explain?
    2. Can incorporating persona variables via prompting improve LLMs' predictions?
    3. For what types of samples is persona prompting most useful?
    4. How effectively can LLMs simulate personas when the importance of persona variables varies?
    
    Key Findings:
    
    **Demographic Limitation**: Sociodemographic variables (like race, gender, age…) only explain 1.4% to 10.6% of human annotation variance, while the text sample itself up to 70%
    
    **The Monolith Trap**: LLMs tend to simulate the average response of a demographic group, treating them as a single monolith, failing to capture wide differences within the same demographic group
    
    **Disagreement Sweet Spot:** Persona prompting is most effective on samples with high entropy and low standard deviation (many human annotators disagree, but their disagreements are small)
    
    **Predictability Treshold**: linear relationship between human predictability and LLM simulation accuracy. If a demographic variable has a low correlation with humman annotations (R^2 < 0,1), the LLM's ability to use that persona to predict annotations drops to zero
    

- **Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement ✔️**
    
    Year: 2025
    
    Author(s): Junyu Lu, Kai Ma…
    
    Main Message: 
    
    LMMs excel at predicting offensive language when human annotators completely agree, but they struggle on subjective, ambiguous matters. In these low-agreement cases, LLMs act as "overconfident judges" and produce highly certain, rigit predictions that completely ignore human uncertainty -> high false-positives
    
    Experiment:
    
    MD-Agreement dataset, which separates offensive and non-offensive text based on human consensus levels (unanimous A++, mild A+, and weak/disagreed A0).
    Evaluated 12 open-source and closed-source LLMs (including GPT-4o, Claude-3.5, Llama-3-70B…) on model confidence alignment. Tested whether training on disagreement samples improves performance and analysed which linguistic features cause the most model errors.
    
    Key Findings:
    
    **Performance Crash on Ambiguity**: high accuracy on clear-cute A++ samples (~87%) but drops sharply to below 65% on low-agreement A0 samples
    
    **Severe Overconfidence**: LLMs maintain incredibly high internal self-consistency (Kamma > 0.75) even on A0 samples, they don't show doubt (incapable of it) even when most humans do
    
    **Bias Towards Censorship**: LLMs highly biased toward classifying non-offensive A0 content (also called N0) as offensive which dropped the accuracy on N0 to 45% -> risk of over-moderation in the real world
    
    **Sweet Spot**: Fine-tuning LLMs using QLoRA on A+ yields the best generalisation and improves out-of-distribution accuracy. Key is to teach the model nuanced boundary lines withotu introducing too much noise
    
    **Disagreement Triggers**: LLMs fail most frequently at identifying sarcasm (F1 score of 54,6%) and rhetorical questions (F1 of 61,6%), while easily classifying explicit triggers like swearing (83.8%)
    
    ---
    

- **Exploring Cross-Cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis** ✔️
    
    Year: 2024
    
    Author(s): Nayeon Lee, Chani Jung…
    
    Main Message: 
    
    Traditional hate speech datasets and LLMs are overwhelmingly biased toward North American cultural norms. Nationality-based cultural background is a major driver of systematic label disagreements and standard LLMs cannot adapt to cultural nuances through simple prompt instructions (like the inclusion of country name in the prompt).
    
    Experiment:
    
    CreHate Dataset with 1580 social media posts annotated by people from 5 diverse English speaking countries. Dataset balanced with North American-centric posts and culturally unique posts from other countries using localised keywords (like slang).
    
    Analysed the rate and causes of cross-country disagreement and evaluated whether LLMs (like GPT-4) and BERT-based classifiers could learn culturally tailored perspectives
    
    Key Findings:
    
    Culture Drives Disagreement: Only 56% of posts achieved unanimous agreement across all 5 countries. Pairwise agreement highest between culturally close nations (AU and GB 83.7%) and lowest between culturally distant onces (SG and ZA 74%). Pairwise annotation agreement is highly negatively correlated (r = -0.658) with Kogut-Singh cultural distance index scores
    
    Reasons for Disagreements: main drivers of cross-cultural label differences were sarcasm (31.7%), personal bias on divisive local topics (27.3%), the perceived offensiveness of swear words (10.0%), and incomplete cultural context (8.0%)
    
    LLMs lack of cultural Adaptability: Standard LLMs show a heavy bias toward US labels and simply adding a countr context to a prompt (e.g. "is this post offensive in Singapore?") does not improve performance. LLMs CANNOT shift their cultural criteria on the fly
    
    Effectiveness of Corss-Cultural Training: Methods like Multi-labeling, multi-task learning, and culture tagging outperform monocultural training and effectively teach models to predict country-specific labels
    
    ---
    

- **The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum** ✔️
    
    Year: 2025
    
    Author(s): Olufunke O. Sarumi, Charles Welch…
    
    Main Message: 
    
    LLMs fail to replicate the true diversity of individual human views (Strong Perspectivism), tending instead toward producing highly aggregated, consensus-driven outputs (Weak Perspectivism) based on their pre-training data
    
    Experiment:
    
    Generated synthetic offensive/abusive annotations for two datasets: HS-Brexit (contrasting views of Muslim/immigrant targets vs. Western researchers) and ConvAbuse (annotated by gender studies experts)
    
    Prompted Llama-2-13B with demographic personas representing these annotators under strong and weak pespectivism across various temperatures and compared standard annotator modeling algorithms' performance on LLM-generated vs human data
    
    Key Findings:
    
    **Artificial Consensus**: LLM-generated annotations have much higher agreement levels than real human annotations (Krippendorff's alpha metric 0,91 vs 0,35) -> persona prompting still tends to default to standard corpus consensus that the model is trained on
    
    **Failure of Personalised Models**:  Algorithms using specific annotator IDs (personalised models) perform best when modeling real humans while when trained on LLM data, simple text-only models without annotator IDs (like SBERT) perform better -> LLM personas fail to provide genuinely unique individual perspectives
    
    **Prototypical Persona Alignment**: LLMs fail to match individual human annotators 1-to-1, but they do form prototypical personas that partially align with broad human groups like multiple human annotators being mapped closely to a single "white British female" LLM persona profile
    
    **Context-Dependent Perspectivism**: Strong perspectivism (predicting individual labels) works better when personas have personalised, overlapping features (like in ConvAbuse) and Weak perspectivism (group-level disagreement) works better when datasets are characterised by sharp, constrasting demographic divisions (like in HS-Brexit)
    

- **Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets** ✔️
    
    Year: 2025
    
    Author(s): Tommaso Giorgi
    
    Main Message: 
    
    Human bias is highly complex and intersectional. Persona-prompted LLMs completely fail to replicate human annotator-target bias distributions
    
    Experiment:
    
    Massive "Measuring Hate Speech" dataset with 135.000+ annotaitons by 8400+ humans which records detailed demographic attributes for both the annotators and the targets of online hate.
    
    Two key metrics defined: Bias Intensity (measuring the direction and strength of over- or underestimating hate) and Bias Prevalence (how frequently the disagreement occurs
    
    Prompted 4 open-source LLMs (like SOLAR, Llama-3-8B...) to role-play the exact same human demographics across 136.000 trials to see if their biases aligned
    
    Key Findings:
    
    **Human In-Group Sensitivity:** Humans show a mild but statistically significant tendency to overestimate hate directed at groups sharing their own traits (which is to be expected), particularly the transgender and seniour annotators
    
    → **The Intersectional Reality of Human Bias**: 46% of all possible human annotator-target combinations showed significant labeling bias (like teenagers underestimating hate across all targets due to desensitisation or Transgender women being highly sensitive to hate directed at LGBTQ individuals)
    
    **LLM Personalisation Failure**: Prompt demographics barely change the output of LLMs, as persona-prompted LLMs showed significant bias in 3.3% of annotator-target combinations (compared to 46% for humans). The actual intensity of biases between humans and LLMs was completely uncorrelated (r=-0.105) -> role-playing LLMs do NOT capture human sociological biases
    

- **D3CODE: Disentangling Disagreements in Data across Cultures on Offensiveness Detection and Evaluation** ✔️
    
    Year: 2024
    
    Author(s): Aida Davani…
    
    Main Message: 
    
    Subjectivity studies in NLP often focus on Western contexts and simple demographic attributes (e.g. gender, age, race…), overlooking that individuals within groups hold diverse values that influence perceptions beyond group norms.
    
    Dataset curation must move beyond traditional demographics to effectively capture subjective and culturally sensitive perspective of offensiveness → incorporate annotator’s core moral values
    
    Experiment:
    
    **D3CODE** dataset, collecting parallel offensiveness ratings on over 4.5K English social media commpents (from Jigsaw) with 4309 participants balanced across gender and age from 21 countries and eight geo-cultural regions.
    
    Participants rated offensiveness on a 5-point Likert scale (from strongly disagree to strongly agree) and filled out the Morael Foundations Questionnaire (MFQ-2) to measure their priorities across six moral dimensions: Care, Equality, Proportionality, Authority, Loyalty, and Purity
    
    Key Findings:
    
    **significant demographic and regional variations in annotation certainty**: e.g. younger annotators, men and annotators from non-Western regions are less likely to select the “I don’t understand” option
    
    **Highest level of inter-annotator disagreement** on comments containing explicit mentions of **social group identities** (e.g. transgender, LGB, Jewish, Muslim, etc.)
    
    **Shared regional cultures not enough to explain moral preferences**, none of the data-driven moral clusters perfectly aligned with geographic regions → individual moral variations cross geographic borders
    
    → **Modeling moral values is necessary to build culturally aware, pluralistic moderation pipelines**
    

- **Annotation alignment: Comparing LLM and human annotations of conversational safety** ✔️
    
    Year: 2024 (EMNLP 2024)
    
    Author(s): Rajiv Movva…
    
    Main Message: 
    
    LLMs highly capable of emulating the average consensus of human safety evaluations, but blind to demographic variations. They also completely fail at predicting when different demographic groups will disagree.
    
    Experiment:
    
    Evaluated 5 leading LLMs (GPT-3, 4, 4o, Gemini 1.5, Llama 3.1 405B) on 350 ser-chatbot conversations from the DICES-350 dataset. Each convo already rated for safety by 112 human annotators beforehand, representing 10 race-and-gender demographic groups. 
    
    Tested different prompt styles (rating-only vs. chain-of-thought “analyse-rate”), examined average alignment (RQ1), group-specific alignment (RQ2), and the model’s ability to predict demographic disagreements (RQ3)
    
    Key Findings:
    
    GPT-4 (with analyse-rate prompts) and Llama 3.1 correlate with the average human consensus **better** **than the average human**
    
    LLMs much more permissive of chatbots giving sensitive advice (e.g. relationship, medical, legal, etc.) than humans, but far more stringent (censor-prone) regarding implicit bias, positive, stereotyping or failure to actively denounce hateful statements (e.g. swear words even in positive context) → systematic value patterns in disagreements between LLM and human
    
    Human-LLM Alignment **varies as much within demographic groups as across them** → demographics alone insufficient to capture alignment
    
    LLMs completely failed at predicting which demographic group would find a chatbot response more unsafe → zero correlation with the actual human group disagreements
    

- **GRASP: A Disagreement Analysis Framework to Assess Group Associations in Perspectives** ✔️
    
    Year: 2024 (NAACL)
    
    Author(s): Vinodkumar Prabhakaran…
    
    Main Message: 
    
    GRASP framework provides statistical tools to measure choesion and strength of group-level perspectives, identifying which axes are most relevant for a given subjective dataset without enforcing majoritarian gold standards → previously missing as a systematic approach
    
    Experiment:
    
    Formulated **Group Association Index (GAI)**, community-detection-inspired metric that calculates the ratio of In-group cohesion (agreement within a subgroup, measured via Krippendorff’s alpha index) to Cross-Group Cohesion (agreement with those outside the group, measured via Cross-Replication Reliability). 
    
    Applied GAI alongside permutation-based signifance testing to two datasets: DICES-350 (conversational safety) and D3CODE (offensiveness)
    
    Key Findings:
    
    No significant GAI scores for individual demographic dimensions (gender, age…) in conversational safety (DICES), but **highly significant group associations** in **intersectional subgroups** (e.g. Latino women, Black women, and white men).
    
    Highest GAI: white men, driven by a strong, systematic tendency to favor “safe” ratings on borderline content
    
    Both age and region primary drivers of systematic disagreement in offensiveness detection (D3) → 18-30 yo Western Europeans showed high in-group agreement, whereas 50+ yo annotators and non-binary gender annotators had the highest overall group associations
    
    → This paper provides a good **mathematical and statistical toolkit for the project**. Consider GAI score for the thesis!
    

- **Exploring Cross-Cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis** ✔️
    
    Year: 2024 (NAACL)
    
    Author(s): Nayeon Lee
    
    Main Message: 
    
    Traditional English hate speech datasets skew heavily toward North Americal perspectives and assume a single language represents a single culture whereas in reality, English-speaking counties differ significantly in their hate speech interpretations based on nationality
    
    → heavy US-centric cultural bias in LLMs
    
    Experiment:
    
    Constructed the CREHate dataset, 1580 posts with 980 sampled from the North Americal biased SBIC dataset and 600 collected using localised keywords from Reddit and YouTube comments in Australia, UK, Singapore and South Africa.
    
    Collected parallel annotations (5 per country) across AU, GB, US, SG and ZA and evaluated 5 LLMs on zero-shot country-specific hate speech classification.
    
    Trained BERT-based classifiers using cross-cultural methods (multi-labeling, multi-task learning, and culture tagging)
    
    Key Findings:
    
    - Only **56.2% of the posts achieved unanimous label agreement** across all five countries.
    - Pairwise country agreement is highly negatively correlated (r = -0.658) with Kogut-Singh cultural distance, with close countries agreeing most (UK and Australia at 83.7%) and distant ones agreeing least (Singapore and South Africa at 74%).
    - Disagreements are primarily driven by differing understandings of **sarcasm (31.7%)**, **personal bias on divisive local topics (27.3%)**, and the perceived offensiveness of **swearing (10.0%)**.
    - GPT-4 exhibits a strong bias toward US-centric labels. Specifying the target country in the prompt (e.g. "Answer if this is hate in Australia") fails to change its predictions → lack of cultural adaptability.
    - Cross-cultural modeling techniques, specifically **multi-task learning** (separate classifier layers for each country) and **culture tagging** (pre-pending country tokens like [SG] to the text), significantly outperform monocultural baselines.

- **Whose Ground Truth? Accounting for Individual and Collective Identities Underlying Dataset Annotation** ✔️
    
    Year: 2021
    
    Author(s): Remi Denton…
    
    Main Message: 
    
    Crowdworkers are not interchangable units who enforce a single “one truth” gold standard and treating them like this is an unethical design decision that silences subjective variations, flatten minority viewpoints and embeds systemic demographic biases in downstream models
    
    Experiment:
    
    This paper did not conduct a new experiment, they just developed a conceptual framework that synthesises extensive interdisciplinary literature on crowdsourced data curation
    
    Key Findings:
    
    - Disagreements in highly subjective tasks (like toxic language and sentiment) represent valuable signal, not noise
    - lived experiences of annotators = critical domain expertise (e.g. women identify online abuse more readily)
    - platforms are designed to treat crowdworkers as interchangeable which causes a power asymmetry → forces the workers to anticipate what the “requester” or the platform wants to see → leads to the “portability trap” where models are applied to social contexts they do not represent
    
    **Strategic reccomendation:** ML practitioners must decide up front if a task is subjective and explicitly release unaggregated individual labels and annotator demographic/group contexts 
    

- **Jury Learning: Integrating Dissenting Voices into Machine Learning Models** ✔️
    
    Year: 2022
    
    Author(s): Mitchell L. Gordon
    
    Main Message: 
    
    Proposes a supervised ML architecture that resolves annotator disagreements explicitly through the metaphor of a jury: allowing practitioners to define which demographic groups, in what proportion, determine a classifier’s prediction for any given item
    
    Experiment:
    
    Developed a DL architecture that combines (BERTweet) for content embedding with a Deep & Cross Network recommender system which together as one model learns individual-level annotator embeddings and group embeddings.
    
    Trained model on a balanced toxicity dataset of 107.620 comments labeled by 17.280 annotators with demographic metadata. Also 18 online moderators to design custom juries to evaluate safety outcomes
    
    Key Findings:
    
    Jointly modeling individual annotators (treats "predicting Annotator A's label" and "predicting Annotator B's label" as distinct, but related, tasks) (MAE 0.61) is significantly more accurate at predicting individual human labels than traditional aggregate classifiers (assuming one ground-truth) (MAE 0.90) and group-only classifiers (assumes people within a specific demographic or behavioral group think the same way) (MAE 0.81)
    
    Predicting aggregate/jury verdicts using a **median-of-means estimator** (over 100 resampled juries) achieves a highly accurate MAE of 0.27 (vs. 0.41 for traditional classifiers)
    
    when moderators designed highly diverse juries (averaging 5.7 unique race values and 3.1 unique gender values), increasing representation of historically marginalised groups, this customised composition **flipped the model’s toxicity decisions on 13.6% of all items**, primarily on the most divisive comments
    
    The architecture successfully supports **conditional juries** (adapting composition based on the topic, e.g. allocating more seats to LGBTQ+ individuals on comments about LGBTQ+ rights) and **counterfactual juries** (identifying the minimal change in jury composition needed to flip a decision)
    

- **Who and What? Using Linguistic Features and Annotator Characteristics to Analyze Annotation Variation** ✔️
    
    Year: 2026
    
    Author(s): Maximilian Maurer…
    
    Main Message: 
    
    **Annotation is an interactive process**: annotator demographics and attitudes alone explain very little rating variance unless they are analysed in **interaction with specific linguistic characteristics of the text**
    
    Experiment:
    
    Conducted largest annotation variation analysis to date, spanning four unaggregated harmful language datasets (CTDP, MHS, POPQUORN, D3CODE), containing a total of >25k items, >8k annotators, and >205k annotations. Extracted 327 general linguistic features (e.g. syntactic complexity and psycholinguistic norms) alongside domain-specific lexical signals (slurs and vulgarity).
    
    Utilised Bayesian multilevel regression with Horseshoe priors to model up to 5264 fixed effects while including random intercepts and annotators to capture cross-classified dependencies
    
    Key Findings:
    
    Main demographic characteristics often useless without **interaction effects**: e.g. in POPQUORN, younger annotators desentisised to explicit vulgarity while older annotators significantly more sensitive to hateful words (age.L:n_hateful)
    
    In MHS, **powerful intersectional effect between ideology and age** → towards political extremes the rating gap between age groups become highly pronounced: older liberals rating items much higher (more offensive) and older conservatives rating them lower
    
    In D3CODE: **moral foundations (**specifically care which was most significant across all) **are strong individual predictors of offensiveness ratings**, especially women show systematic sensitivity to gendered slurs (genderwoman:n_pr)
    
    (once again) item-specific attitudes (e.g. whether an annotator believes a comment should be removed or is fine to see) explain more variance than broad sociodemographic profiles)
    
    effect patterns vary drastically across datasets: annotation findings **are highly context-specific** and do not generalise easily
    
    → **to successfully predict *when and where* disagreement arises, the model must analyse the interaction between the specific linguistic structure of text and the individual profile of the reader** for the most accurate prediction
    

- **Majority Vote Silences Minority Values: Annotator Disagreement at the Hate/Offensive Boundary in HateXplain** ✔️
    
    Year: 2026 (ICML Pluralistic Alignment Workshop)
    
    Author(s): Joshua Muhumuza…
    
    Main Message: 
    
    Majority-vote aggregation in hate speech moderation pipelines structurally flawed: annotator disagreement systematically concentrates at the **fuzzy boundary between “hate speech” and “offensive speech”**. 
    
    Standard downstream modeling adjustment **cannot recover what aggregation discards** (Once you force a majority vote)
    
    Experiment: 
    
    HateXplain dataset (20.148 posts annotated as hatespeech, normal or offensive). Trained three BERT-base models: Model A (Hard labels/majority vote), Model B (Soft labels/annotator distribution), and Model C (Per-annotator heads ensembled at inference)
    
    Analysed token-level rationale annotations (calculating Jaccard similarity between the text tokens highlighted by different annotators as their justification) → verified if disagreement was driven by random annotation noise/error vs. genuine severity thresholds
    
    Key Findings:
    
    48.7% of all training posts show annotator disagreement → **42.6% of this disagreement concentrates specifically at the hate/offensive boundary**
    
    Offensive posts highly over-represented in disagreement, with a disagreement rate of 67.9%
    
    All 3 models suffer a massive 22-28% accuracy drop on disagreement posts compared to agreed posts
    
    Model C: highest agreed accuracy 0.826 but worst disagreement accuracy 0.545 and accuracy on offensive disagreement posts 0.245 → per-annotator heads fail to generalise to contested boundary inputs
    
    Jaccard similarity analysis of rationales (selected justifications words) shows that **73.1% of different-label annotators highlight the exact same tokens** (Jaccard ≥ 0.3) → **disagreement is a structural threshold conflict on identical evidence**, not random reader carelessness
    
    Model A highly confident on boundary errors (0.710), Model C more honest but less accurate (0.495)
    
    → standard downstream models cannot solve the primary, subjective boundary of “offensive vs hateful” of human disagreement. The solution must be **upstream**, using graduated harm scales and representing affected communities’ values explicitely (again with the moral classification)
    

- **Disentangling Perceptions of Offensiveness: Cultural and Moral Correlates** ✔️
    
    Year: 2024 (FAccT)
    
    Author(s): Aida Mostafazadeh Davani…
    
    Main Message: Substantial geo-cultural differences in how people perceive offensiveness in language, heavily mediated by the moral foundations of **Care and Purity**
    
    Individual-level moral values far more significant for shaping perceptions of offensiveness than country-level demographic categories
    
    Experiment:
    
    large-scale cross-cultural study of 4309 participants from 21 countries across 8 cultural regions rating the offensiveness of 4554 comments on a 5-point Likert scale → measured their moral reasoning along six moral foundations using the MFQ-2 and conducted statistical mediation and decomposition analyses to evaluate how commercial moderation engines (Perspective API) align with these regional and moral perspectives 
    
    Key Findings:
    
    Arab and Latin America rated comments as significantly more offensive, Oceania and Sinosphere rated as least offensive (regardless of age, gender, socio-economic status and whether a formal definition of offensiveness was provided)
    
    Under Hofstede's cultural dimensions, countries that score high on Uncertainty Avoidance, Individualism, Femininity, and Short-term orientation are significantly more sensitive to offensive language
    
    Geocultural variations significantly **mediated by moral priorities of Care (avoiding harm) and Purity (avoiding degradation)**
    
    Decomposition analysis proves that **individual-level moral deviation** has a much more significant impact on ratings than country-level moral averages → A 1-point increase in individual Care score relative to their country average is associated with a 0.14 increase in assigned offensiveness
    
    Perspective API and Jigsaw labels align preferentially with individuals scoring high in Care, and show extremely weak correlation with Sinosphere and Oceania → commercial moderation engines prioritize Western, WEIRD value systems
    

- **Who’s asking? User personas and the mechanics of
latent misalignment** ✔️
    
    Year: 2024
    
    Author(s): Asma Ghandeharioun (Google Research / Brown Uni)…
    
    Main Message: 
    
    Safety training (like RLHF) implements localised, layer-specific safeguards in the later layers of the model instead of globally erasing harmful or misaligned capabalities → whether a model refuses a dangerous query depends heavily on the inferred **"user persona"**
    
    Manipulating user personas via activation steering highly effective at bypassing safety filters
    
    Experiment:
    
    created **SneakyAdvBench** (100 sneaky adversarial queries rewritten to be less obviously harmful, e.g., asking how to steal an identity in a subtle way), evaluated Llama-2-13B-chat and Gemma-7B
    
    manipulated user personas (pro-social: *curious*, *altruistic*; anti-social: *selfish*, *unlawful*) using two methods: Prompt Prefixes (PP) and Contrastive Activation Addition (CAA) steering vectors in mid-layers. Analysed internal mechanics using early-layer decoding (early exiting) and open-ended Patchscopes
    
    Key Findings:
    
    Aligned LLMs biased by who they think they are talking to. Pro-social user personas (e.g., curious, altruistic) and negated anti-social personas (e.g., selfish with a negative multiplier) significantly increase the model's willingness to answer sneaky, harmful queries
    
    ntervening directly on hidden activations in early-to-mid layers (specifically layer 13) using CAA far more effective at changing refusal behavior than natural language prompt prefixes
    
    Even when an LLM successfully refuses a harmful query at the final output layer, the harmful information remains fully encoded in its early-to-mid layer representations and can be extracted by directly decoding from earlier layers (e.g., layer 13 to 25)
    
    Applying certain steering vectors (like altruistic+ or selfish-) physically changes how the model interprets a query → model begins to ascribe innocent, "good" motives to the user, reinterpreting a query on copying brands as "avoiding plagiarism and ensuring originality”
    
    The cosine similarity between a persona's steering vector and the model's baseline refusal vector in layer 13 is highly predictive of its downstream effect on safety filters → allows researchers to predict safety bypasses before running inference
    
    **Offensiveness and safety are not absolute, static properties of a text but rather highly dynamic interpretations that shift based on the rater's perspective →** user persona directly influences the model’s internal semantic representation of the text itself. 
    
    This paper completely invalidates single-label classification and provides the internal, layerwise framework to study where disagreement and perspective-taking are represented inside LLMs
    

- **Robust Persona-Aware Toxicity Detection with Prompt Optimization and
Learned Ensembling** ✔️
    
    Year: 2025
    
    Author(s): Berk Atil (Pennsylvania State Uni / Resolution)…
    
    Main Message: 
    
    No single prompting strategy that consistently performs best across different rater personas and base models for toxicity detection
    
    Most effective way to represent pluralistic viewpoints: **learned meta-ensembling** → training an SVM classifier over a binary prediction vector of diverse prompts
    
    Experiment:
    
    Evaluated different persona-conditioned prompting methods (such as direct persona prompting, Value Profiles, and TextGrad-optimized persona prompts) against a default, non-persona baseline across 8 diverse rater personas on the Social Bias Frames dataset (44k social media posts annotated with demogs)
    
    Used LLama 3.1 and Qwen 2.5 base and Deepseek R1-distilled models for this
    
    Key Findings:
    
    No single prompting method uniformly dominant, one might work brilliantly for one demog persona but fail miserably on another
    
    Using TextGrad to automatically optimise prompts through “textual differentiation” has persona prompt performance returns comparable to human-designed Value Profiles but both produce complementary errors → different edge-case errors for both, so neither one clearly superior
    
    Distilled reasoning models help smaller models but degreade performance of larger models → assumption: optimising models for logical and mathematical tasks may be damaging their social and contextual reasoning (would be very interesting if actually the case)
    
    prompted with a persona, models tend to predict non-offensive much more often → boosts accuracy on some personas but completely breaks down for personas that naturally experience higher rates of offensive content in their env (e.g. Hispanic Woman) → systemic bias
    
    non-linear SVM classifier trained on the 4-bit binary vector of prompt predictions (default, persona, value profile, optimized persona) consistently outperformed every single prompting method, unweighted majority-voting, and theoretically weighted majority-voting across all models and personas
    
    Technical architecture for the thesis: consider using a learned meta-ensembling (like SVM classifiers) instead of trying to find the “one perfect prompt”
    

- **Unveiling and Manipulating Prompt Influence in Large Language Models** ✔️
    
    Year: 2024
    
    Author(s): Zijian Feng (Nanyang Technological University, Singapore / Singapore-ETH Centre)…
    
    Main Message: 
    
    Traditional methods to explain how prompts influence LLM outputs (like attention weights or gradients) suffer from linearity assumptions → don’t align with token-generation objectives
    
    Projecting hidden representations into the covab space using the Language Model head (LM head) is more effective. This framework, **Token Distribution Dynamics (TDD)** successfully pinpoints and neutralises offensive triggers in a zero-shot setting
    
    Experiment:
    
    Proposed 3 versions of TDD: TDD-forward, TDD-backward, and TDD-bidirectional across 11 datasets from the **Benchmark of Linguistic Minimal Pairs (BLiMP)** using **five LLMs** (GPT2, GPTJ, BLOOM, Pythia, LLaMA2)
    
    Assessed explanation faithfulness using AOPC and Sufficiency metrics via token perturbation and applied TDD to 2 text-control tasks: zero-shot toxic language suppression (on 1,225 prompts from RealToxicityPrompts) and sentiment steering (on 5,000 prompts from OpenWebText)
    
    Key Findings:
    
    The hidden states of *all* input tokens (not just the last token) can be projected via the LM head as interpretable token distributions over the vocabulary space, which converge monotonically towards the final layer
    
    TDD variants (specifically TDD-backward and TDD-bidirectional) significantly outperform state-of-the-art gradient-based, vector-based (attention rollout), and perturbation-based saliency methods in identifying which prompt tokens causally drive the LLM's output
    
    TDD-backward more effective than TDD-forward because of prompt tokens in reverse order → mitigates the confounding influence of forward linguistic conventions (grammar/syntax), allowing the model to focus purely on semantic dependencies
    
    TDD successfully suppresses toxic output generations in a zero-shot setting by identifying toxic trigger words in prompts (top 15% saliency) and replacing them with space tokens, significantly reducing toxicity while maintaining text fluency
    
    TDD enables zero-shot sentiment steering by identifying the most influential prompt token and substituting it with either "positive" or "negative" keys, outperforming other explanation methods like Con-GI and Con-GN
    
    This paper basically provides a mathematical tool to **reveal exactly which words in a user's text triggered the offensive prediction**. Also gives the TDD’s zero-shot trigger neutralisation to surgically replace only the specific “problematic” words instea dof flat-out censoring entire posts
    

- **Demographic Prompting at Scale: When More Attributes Hurt
LLM–Human Agreement** ✔️
    
    Year: 2025
    
    Author(s): Mahammed Kamruzzaman…
    
    Main Message: 
    
    Prompting Large Language Models (LLMs) with specific rater demographics to simulate subjective human judgments is not a straightforward "more is better" fix
    
    Providing a small, selected group of high-signal, directionally coherent demographic attributes (1 to 3) improves the model's alignment with humans. Adding more than that (where it crosses the line of “too many”) triggers an “**over-specification threshold”** that consistently degrades performance
    
    Experiment:
    
    evaluated the entire combinatorial space of demographic attributes (from single variables to full combinations) across **5 subjective language tasks** (Toxicity, Sentiment, Politeness, Offensiveness, and Emotion) using 5 open-source LLMs (Llama-3.2-3B, Mistral-7B, Gemma-3-12B, Qwen2.5-7B, and DeepSeek-R1-7B)
    
    measured model-human agreement using quadratic-weighted Cohen’s κ and mapped the structural quality of human annotations using SHAP, Linear Support Vector Classifiers (LinearSVC), and internal neuron probing
    
    Key Findings:
    
    Model-human alignment consistently peaks with only 1 to 3 demographic attributes in the prompt and degrades under the full set → The “all-together” demographic config never the best performing
    
    **Knowing which demographic attributes most strongly predict variation in actual human annotations** (using SHAP) **provides zero predictive power on whether prompting an LLM with those variables will improve its alignment**
    
    Attributes with directionally coherent signals (supgroups agree on which words signal the label) yield alignment gains, subgroups with directionally opposed signals degrade alignment → no single persona prompt can resolve conflicting subgroup signals
    
    An attribute can be easily learned by a classifier but still be unsusable by a prompt → age = most lexically learnable attribute but prompting with it causes some of the largest alignment degredations because its subgroups are directionally conflicted
    
    activating more specialised neurons doesn't make the model better at following instructions → Internal neurons **only** help a model align with humans if the human annotations point in a **clear, consistent direction**. Deepseek Model  **highest number of specialized neurons**, yet the **worst at being steered** by persona prompts
    
    Connection to Thesis: 
    
    **Simply prompting models with rater demographics is not a silver bullet** to capture disagreement. To predict *when and where* disagreement arises, the computational models must analyse the **structural coherence of the rater groups' annotation signals** and text features, rather than treating demographic labels as a standard, modular plug-and-play input