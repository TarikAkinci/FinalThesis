import pandas as pd
import json
from sklearn.model_selection import train_test_split

SEED = 42

df_items = pd.read_csv('/Users/tarikakinci/Desktop/FinalThesis/D3code-main/dataset/d3-items.csv')

# item_id is not globally unique in d3-items.csv: 34 ids are shared by rows with
# different text/category (e.g. one 'random' item and one 'moral' item collide
# on the same id). d3-ratings.csv keys ratings only by item_id, so for these
# rows there's no way to tell which text a given rating actually belongs to.
# Drop them rather than guess, since a wrong guess would silently corrupt the
# frozen split (this is what caused item ids to leak across train/val/test
# in the previous version of this script, which split on raw undeduped rows).
dupe_ids = df_items['item_id'][df_items['item_id'].duplicated(keep=False)].unique()
n_before = len(df_items)
df_items = df_items[~df_items['item_id'].isin(dupe_ids)].copy()
print(f"Dropped {len(dupe_ids)} ambiguous item_ids ({n_before - len(df_items)} rows) with colliding text/category")

# First, split 70% train and 30% temp
train_items, temp_items = train_test_split(
    df_items,
    test_size=0.30, 
    stratify=df_items['category'],
    random_state=SEED
)

# Then split temp into 50% val (15% overall) and 50% test (15% overall)
val_items, test_items = train_test_split(
    temp_items,
    test_size=0.50,
    stratify=temp_items['category'],
    random_state=SEED
)

splits = {
    'train': train_items['item_id'].tolist(),
    'val': val_items['item_id'].tolist(),
    'test': test_items['item_id'].tolist()
}

print(f"Total items: {len(df_items)}")
print(f"Train size: {len(splits['train'])}")
print(f"Val size: {len(splits['val'])}")
print(f"Test size: {len(splits['test'])}")

train_set, val_set, test_set = set(splits['train']), set(splits['val']), set(splits['test'])
assert not (train_set & val_set), "train/val overlap"
assert not (train_set & test_set), "train/test overlap"
assert not (val_set & test_set), "val/test overlap"
print("No item_id overlap across splits (verified)")

out_path = '/Users/tarikakinci/Desktop/FinalThesis/D3code-main/splits.json'
with open(out_path, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"Splits saved to {out_path}")
