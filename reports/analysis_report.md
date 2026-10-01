# D3 Dataset Analysis Report (Phase 1 & 2)

## Phase 1 — Get your hands dirty first

### Structural Basics
- **Dataset Shapes**: Items (4590, 4), Raters (4309, 12), Ratings (150702, 5).
- **Join Keys**: The joins are perfectly clean. There are **0 orphan ratings** (no missing items or raters).
- **Annotator-density**: The distribution of ratings per item is plotted in `plots/ratings_per_item.png`. 
- **Rater Contribution**: The distribution of items per rater is plotted in `plots/items_per_rater.png`.
- **-1 ("didn't understand") Values**: Approximately **3.95% (6047/153251)** of the ratings are `-1`.
  - This absolutely clusters by country! The UK (12.6%), Canada (12.5%), and New Zealand (10.7%) have the highest proportions of `-1` ratings, suggesting potential language/comprehension signals or cultural differences in interpreting specific slangs.

### Basic Descriptive Exploration
- **Overall Ratings**: The overall distribution of ratings (excluding `-1`) heavily skews towards `0` (not offensive), as seen in `plots/rating_dist_overall.png`.
- **Ratings by Category**: The moral category tends to have lower offensiveness ratings, while social-group items have a higher spread. See `plots/rating_dist_category.png`.
- **Ratings by Region**: Regional variations are present (e.g., Arab Culture tends to rate higher on offensiveness in certain contexts). See `plots/rating_dist_region.png`.
- **Rater Demographics**: The paper's claim of demographic balance is largely true for Gender (Man: 49.9%, Woman: 49.2%). Age is skewed younger (18-30: 46.9%, 30-50: 34.7%). Regions are perfectly balanced (~12-13% each). See `plots/rater_demographics.png`.
- **MFQ-2 Scores**: Regional variation in moral values is clearly visible in the raw numbers. For example, "care" and "fairness" scores are universally high, while "loyalty" and "purity" show stronger regional variance. See `plots/mfq_by_region.png`.

## Phase 2 — Deeper Analysis Questions

### Disagreement-Specific
- **Highest Disagreement Items**: The items with the highest variance (Std. Dev ~1.90) belong exclusively to the `social-group` category. Examples include:
  1. *"When a Christian performs a terrorist act, do you label it a Christian act of terrorism?..."*
  2. *"Imam calling for Jews to be killed in sermon at Montreal mosque draws police complaint..."*
  3. *"2 million? I'm going to vomit. Muslims have not built this country..."*
  - **Qualitative check**: Yes, these perfectly match the pattern of divisive topics (religion, terrorism) and explicit identity mentions that trigger polarization and high variance in offensiveness ratings.
- **Krippendorff's α**: The overall inter-rater agreement is low: **α = 0.1576**. 
  - However, within regions, agreement tends to be slightly higher, supporting the cultural distance hypothesis. North America (0.2072) and Western Europe (0.2063) show the highest intra-region agreement, while the Sinosphere shows the lowest (0.1048).
- *(Note: We approximated cultural grouping through MFQ dimension averages per country, visualized in `plots/country_mfq_heatmap.png`, in place of a direct Hofstede external join).*

### Moral Values Angle
- **Regression Analysis**: We regressed the individual rating against the deviation of an individual's MFQ score from their country's average. 
  - **Result**: Individual deviations from cultural norms *do* predict rating deviations. Specifically, an individual's deviation in the **care** foundation is a strong positive predictor (coeff = 0.1444, p < 0.0001) of rating something as more offensive. **Loyalty** deviation also positively predicts ratings (coeff = 0.0269, p < 0.0001), whereas **equality** deviation shows a slight negative effect.
- **Heatmap**: `plots/country_mfq_heatmap.png` clearly shows how baseline moral foundations cluster and vary across countries in the dataset.

### Item-Level
- **Disagreement by Category**: `social-group` is indeed the highest-disagreement bucket, confirming the hypothesis. Mean standard deviations by category:
  - `social-group`: **1.389**
  - `random`: **1.141**
  - `moral`: **0.874**
  - *(Visualized in `plots/disagreement_by_category.png`)*
- *(Note: The Jigsaw 2018 vs 2019 breakdown was omitted as the source mapping wasn't available in `d3-items.csv`)*

### Modeling Warm-ups
We used `all-MiniLM-L6-v2` sentence embeddings to predict offensiveness.
- **Predicting Mean Item Rating**: A Ridge regressor trained on text embeddings to predict the mean item rating achieves an **$R^2$ of 0.3327**. This indicates that a solid portion of the variance in the *average* offensiveness of an item can be predicted from the text alone without rater demographics.
- **Leakage Check (Predicting Individual Ratings)**:
  - **Random Split (leaks items across sets)**: $R^2$ = 0.0695
  - **Item-Disjoint Split (strict)**: $R^2$ = 0.0554
  - **Observation**: Performance drops when doing a proper disjoint split. This confirms the leakage you flagged—if the model sees the same item in train and test (even from different raters), it memorizes the item itself, artificially inflating the score.
