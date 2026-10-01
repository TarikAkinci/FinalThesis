import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

print("=== Phase 1 Data Exploration ===\n")


items_df = pd.read_csv('/Users/tarikakinci/Desktop/FinalThesis/D3code-main/dataset/d3-items.csv')
raters_df = pd.read_csv('/Users/tarikakinci/Desktop/FinalThesis/D3code-main/dataset/d3-raters.csv')
ratings_df = pd.read_csv('/Users/tarikakinci/Desktop/FinalThesis/D3code-main/dataset/d3-ratings.csv')

print(f"Items shape: {items_df.shape}")
print(f"Raters shape: {raters_df.shape}")
print(f"Ratings shape: {ratings_df.shape}\n")


missing_items = ratings_df[~ratings_df['item_id'].isin(items_df['item_id'])]
missing_raters = ratings_df[~ratings_df['rater_id'].isin(raters_df['rater_id'])]
print(f"Orphan ratings (missing items): {len(missing_items)}")
print(f"Orphan ratings (missing raters): {len(missing_raters)}\n")


ratings_per_item = ratings_df.groupby('item_id').size()
plt.figure(figsize=(10, 5))
sns.histplot(ratings_per_item, bins=30, kde=True)
plt.title('Distribution of Ratings per Item')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Items')
plt.savefig('plots/ratings_per_item.png')
plt.close()
print("Saved plots/ratings_per_item.png")


items_per_rater = ratings_df.groupby('rater_id').size()
plt.figure(figsize=(10, 5))
sns.histplot(items_per_rater, bins=30, kde=True)
plt.title('Distribution of Items per Rater')
plt.xlabel('Number of Items Rated')
plt.ylabel('Count of Raters')
plt.savefig('plots/items_per_rater.png')
plt.close()
print("Saved plots/items_per_rater.png")


merged_df = ratings_df.merge(raters_df, on='rater_id', how='left').merge(items_df, on='item_id', how='left')
minus_one_count = (merged_df['rating_raw'] == -1).sum()
total_ratings = len(merged_df)
print(f"Fraction of -1 ('didn't understand') ratings: {minus_one_count / total_ratings:.4f} ({minus_one_count}/{total_ratings})")

if minus_one_count > 0:
    minus_one_by_country = merged_df[merged_df['rating_raw'] == -1]['Country'].value_counts(normalize=True) * 100
    print("\n-1 Ratings by Country (% of all -1 ratings):")
    print(minus_one_by_country)
else:
    print("\nNo -1 ratings found in `rating_raw`.")


valid_ratings = merged_df[merged_df['rating_raw'] >= 0]

plt.figure(figsize=(8, 5))
sns.countplot(x='rating_raw', data=valid_ratings, palette='viridis')
plt.title('Overall Distribution of Raw Ratings (0-4)')
plt.savefig('plots/rating_dist_overall.png')
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(x='rating_raw', hue='category', data=valid_ratings, palette='Set2')
plt.title('Ratings by Item Category')
plt.savefig('plots/rating_dist_category.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.countplot(x='rating_raw', hue='Region', data=valid_ratings, palette='Set3')
plt.title('Ratings by Region')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('plots/rating_dist_region.png')
plt.close()

print("Saved rating distribution plots.\n")


print("--- Rater Demographics ---")
for col in ['Age', 'Gender', 'Region', 'SES']:
    print(f"\n{col} Distribution (%):")
    print(raters_df[col].value_counts(normalize=True).sort_index() * 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(x='Age', data=raters_df, ax=axes[0,0], palette='pastel')
sns.countplot(x='Gender', data=raters_df, ax=axes[0,1], palette='pastel')
sns.countplot(y='Region', data=raters_df, ax=axes[1,0], palette='pastel')
sns.countplot(x='SES', data=raters_df, ax=axes[1,1], palette='pastel')
plt.tight_layout()
plt.savefig('plots/rater_demographics.png')
plt.close()
print("Saved plots/rater_demographics.png")


mfq_cols = ['care', 'equality', 'proportionality', 'authority', 'loyalty', 'purity']
mfq_melted = raters_df.melt(id_vars=['Region'], value_vars=mfq_cols, var_name='MFQ_Foundation', value_name='Score')

plt.figure(figsize=(14, 8))
sns.boxplot(x='MFQ_Foundation', y='Score', hue='Region', data=mfq_melted)
plt.title('MFQ-2 Moral Foundation Scores by Region')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('plots/mfq_by_region.png')
plt.close()
print("Saved plots/mfq_by_region.png")
print("\nPhase 1 Exploration Complete.")
