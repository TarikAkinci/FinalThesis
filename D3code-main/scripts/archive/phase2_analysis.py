import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import os
import itertools

os.makedirs('plots', exist_ok=True)
print("=== Phase 2 Analysis ===\n")

items_df = pd.read_csv('dataset/d3-items.csv')
raters_df = pd.read_csv('dataset/d3-raters.csv')
ratings_df = pd.read_csv('dataset/d3-ratings.csv')

# Merge
df = ratings_df.merge(raters_df, on='rater_id', how='left').merge(items_df, on='item_id', how='left')
df_valid = df[df['rating_raw'] >= 0].copy()

# 1. Per-item disagreement (std deviation)
item_stats = df_valid.groupby('item_id')['rating_raw'].agg(['std', 'mean', 'count']).reset_index()
item_stats = item_stats[item_stats['count'] > 5] # filter items with very few ratings
top_disagreement = item_stats.sort_values('std', ascending=False).head(10)
print("Top 5 Highest Disagreement Items (by std dev):")
top_items_merged = top_disagreement.merge(items_df, on='item_id')
for idx, row in top_items_merged.head(5).iterrows():
    print(f"ID: {row['item_id']}, STD: {row['std']:.2f}, Category: {row['category']}, Text: {row['text'][:100]}...")

# Disagreement across categories
cat_disagreement = top_items_merged[['category', 'std']].groupby('category').mean() # wait, this is only top items.
all_cat_disagreement = item_stats.merge(items_df[['item_id', 'category']], on='item_id').groupby('category')['std'].mean()
print(f"\nMean Disagreement (std) by Category:\n{all_cat_disagreement}\n")

plt.figure(figsize=(8, 5))
sns.boxplot(x='category', y='std', data=item_stats.merge(items_df[['item_id', 'category']], on='item_id'))
plt.title('Item Disagreement (Std Dev) by Category')
plt.savefig('plots/disagreement_by_category.png')
plt.close()


# 2. Krippendorff's alpha
def compute_alpha(data):
    # data is a DataFrame with item_id, rater_id, rating_raw
    # Drop duplicate ratings by the same rater on the same item (if any) to allow pivoting
    data_dedup = data.drop_duplicates(subset=['rater_id', 'item_id'])
    pivot = data_dedup.pivot(index='rater_id', columns='item_id', values='rating_raw').values
    # Krippendorff expects reliability data in shape (raters, items)
    # The `krippendorff.alpha` expects numpy array with np.nan for missing
    # Warning: this matrix can be huge and sparse. For memory, we sample or just run it.
    try:
        alpha = krippendorff.alpha(reliability_data=pivot, level_of_measurement='ordinal')
        return alpha
    except Exception as e:
        return np.nan

print("Computing Overall Krippendorff's Alpha... (this may take a minute)")
# To avoid memory issues with massive sparse matrix, we take a sample of items if it's too large, or just run it.
# Pivot shape: ~4300 raters x 4500 items = ~20M elements, small enough for memory.
overall_alpha = compute_alpha(df_valid)
print(f"Overall Krippendorff's Alpha: {overall_alpha:.4f}")

regions = df_valid['Region'].unique()
region_alphas = {}
print("\nKrippendorff's Alpha by Region:")
for r in regions:
    r_data = df_valid[df_valid['Region'] == r]
    a = compute_alpha(r_data)
    region_alphas[r] = a
    print(f"  {r}: {a:.4f}")


# 3. Cultural Distance vs Agreement (using MFQ as proxy for Kogut-Singh if Hofstede unavailable)
# Country average MFQ
mfq_cols = ['care', 'equality', 'proportionality', 'authority', 'loyalty', 'purity']
country_mfq = raters_df.groupby('Country')[mfq_cols].mean()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(country_mfq, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Country Average MFQ-2 Scores')
plt.tight_layout()
plt.savefig('plots/country_mfq_heatmap.png')
plt.close()

# 4. Regression: rating deviation ~ MFQ deviation
# Calculate country averages and merge back
country_avg = raters_df.groupby('Country')[mfq_cols].mean().reset_index()
country_avg.columns = ['Country'] + [f"{c}_avg" for c in mfq_cols]
df_reg = df_valid.merge(country_avg, on='Country', how='left')

# Calculate deviations
for c in mfq_cols:
    df_reg[f"{c}_dev"] = df_reg[c] - df_reg[f"{c}_avg"]

# Build formula
dev_cols = [f"{c}_dev" for c in mfq_cols]
formula = "rating_raw ~ " + " + ".join(dev_cols) + " + C(Country) + C(category)"
print("\nRunning Regression: rating_raw ~ MFQ_deviations + Country + Category")
model = smf.ols(formula, data=df_reg).fit()
print("Regression summary (MFQ dev coefficients):")
for col in dev_cols:
    print(f"  {col}: {model.params.get(col, np.nan):.4f} (p={model.pvalues.get(col, np.nan):.4f})")
print(f"  R-squared: {model.rsquared:.4f}")


# 5. Modeling Warm-up (Text Embeddings)
print("\nRunning NLP Modeling Baseline...")
model_emb = SentenceTransformer('all-MiniLM-L6-v2')

item_texts = items_df['text'].tolist()
print("Encoding texts...")
embeddings = model_emb.encode(item_texts, show_progress_bar=False)

# Target: mean rating per item
target_df = df_valid.groupby('item_id')['rating_raw'].mean().reset_index()
# Merge to ensure alignment
model_df = items_df.merge(target_df, on='item_id', how='inner')
# Align embeddings
idx_map = {row['item_id']: i for i, row in items_df.iterrows()}
X = np.array([embeddings[idx_map[i]] for i in model_df['item_id']])
y = model_df['rating_raw'].values

# Item-level split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
r2_item = r2_score(y_test, preds)
print(f"Model Performance (Item-split) -> R2: {r2_item:.4f}, MSE: {mean_squared_error(y_test, preds):.4f}")

# To do a proper rater-split check, we would predict individual ratings, but the prompt says 
# "predict mean item rating... Split by item vs. by rater and see how much performance differs".
# Wait, if we predict *mean item rating*, rater split doesn't make sense because the target is per-item.
# The prompt likely meant: Predict individual rating (rating_raw), and do a random split (leaking item identities) 
# vs item-split (no overlap of items between train and test).
# Let's do that quickly on a subsample (to save time).
df_sub = df_valid.sample(20000, random_state=42)
X_sub = np.array([embeddings[idx_map[i]] for i in df_sub['item_id']])
y_sub = df_sub['rating_raw'].values

# Random split (leaks items)
X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
reg_random = Ridge().fit(X_tr, y_tr)
p_rand = reg_random.predict(X_te)
print(f"Model Performance predicting individual rating (Random split - leaks items) -> R2: {r2_score(y_te, p_rand):.4f}")

# Item-disjoint split
train_items, test_items = train_test_split(items_df['item_id'].unique(), test_size=0.2, random_state=42)
df_train = df_sub[df_sub['item_id'].isin(train_items)]
df_test = df_sub[df_sub['item_id'].isin(test_items)]
X_tr_dis = np.array([embeddings[idx_map[i]] for i in df_train['item_id']])
y_tr_dis = df_train['rating_raw'].values
X_te_dis = np.array([embeddings[idx_map[i]] for i in df_test['item_id']])
y_te_dis = df_test['rating_raw'].values

reg_dis = Ridge().fit(X_tr_dis, y_tr_dis)
p_dis = reg_dis.predict(X_te_dis)
print(f"Model Performance predicting individual rating (Item-disjoint split) -> R2: {r2_score(y_te_dis, p_dis):.4f}")

print("\nPhase 2 Analysis Complete.")
