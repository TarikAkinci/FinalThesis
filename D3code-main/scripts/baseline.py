import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from evaluate import ratings_to_distribution, entropy, evaluate_central_tendency

def compute_stats(ratings):
    valid = [r for r in ratings if r != -1]
    mean_rating = np.mean(valid) if valid else np.nan
    dist = ratings_to_distribution(ratings)
    ent = entropy(dist)
    return pd.Series({'mean_rating': mean_rating, 'entropy': ent})

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')

    with open(os.path.join(project_root, 'splits.json')) as f:
        splits_dict = json.load(f)
    
    item_to_split = {}
    for split_name, item_ids in splits_dict.items():
        for i_id in item_ids:
            item_to_split[i_id] = split_name
            
    items_df = pd.read_csv(os.path.join(dataset_dir, 'd3-items.csv'))
    ratings_df = pd.read_csv(os.path.join(dataset_dir, 'd3-ratings.csv'))

    stats_df = ratings_df.groupby('item_id')['rating_raw'].apply(compute_stats).unstack()
    
    df = items_df.merge(stats_df, on='item_id', how='inner')
    df['split'] = df['item_id'].map(item_to_split)
    
    missing_splits = df['split'].isna().sum()
    if missing_splits > 0:
        print(f"Dropping {missing_splits} items not found in splits.json")
    df = df.dropna(subset=['split'])    
    print("\nSplit counts:")
    print(df['split'].value_counts())
    
    print("\nMean entropy by category:")
    print(df.groupby('category')['entropy'].mean().sort_values())
    
    df = df.dropna(subset=['mean_rating', 'entropy'])
    
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    

    train_mean_rating = df_train['mean_rating'].mean()
    train_mean_entropy = df_train['entropy'].mean()
    
    naive_pred_mean = np.full(len(df_test), train_mean_rating)
    naive_pred_entropy = np.full(len(df_test), train_mean_entropy)
    

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(df_train['text'])
    X_test = vectorizer.transform(df_test['text'])
    
    model_mean = Ridge()
    model_mean.fit(X_train, df_train['mean_rating'])
    tfidf_pred_mean = model_mean.predict(X_test)
    
    model_entropy = Ridge()
    model_entropy.fit(X_train, df_train['entropy'])
    tfidf_pred_entropy = model_entropy.predict(X_test)
    

    results = []
    
    res_naive_mean = evaluate_central_tendency(df_test['mean_rating'], naive_pred_mean)
    res_naive_mean.update({'experiment_name': 'naive', 'target': 'mean_rating'})
    results.append(res_naive_mean)
    
    res_naive_entropy = evaluate_central_tendency(df_test['entropy'], naive_pred_entropy)
    res_naive_entropy.update({'experiment_name': 'naive', 'target': 'entropy'})
    results.append(res_naive_entropy)
    
    res_tfidf_mean = evaluate_central_tendency(df_test['mean_rating'], tfidf_pred_mean)
    res_tfidf_mean.update({'experiment_name': 'tfidf_ridge', 'target': 'mean_rating'})
    results.append(res_tfidf_mean)
    
    res_tfidf_entropy = evaluate_central_tendency(df_test['entropy'], tfidf_pred_entropy)
    res_tfidf_entropy.update({'experiment_name': 'tfidf_ridge', 'target': 'entropy'})
    results.append(res_tfidf_entropy)
    
    results_df = pd.DataFrame(results)
    cols = ['experiment_name', 'target', 'mae', 'rmse', 'pearson_r', 'spearman_r']
    results_df = results_df[cols]
    
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    results_csv_path = os.path.join(project_root, 'results.csv')
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)
        
    print(f"\nSaved results to {results_csv_path}")
    

    df_test = df_test.copy()
    df_test['abs_error'] = np.abs(df_test['mean_rating'] - tfidf_pred_mean)
    print("Mean absolute error by category (TF-IDF on mean_rating):")
    print(df_test.groupby('category')['abs_error'].mean().sort_values(ascending=False))

if __name__ == '__main__':
    main()
