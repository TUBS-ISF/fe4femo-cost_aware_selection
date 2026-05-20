import sys
from sklearn.model_selection import StratifiedKFold
from helper.load_dataset import load_feature_data, get_flat_models

data_path = sys.argv[1] if len(sys.argv) > 1 else "fe4femo/data"

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
df = load_feature_data(data_path)

for train_index, test_index in kf.split(df, get_flat_models(df)):
    print(" ".join(str(x) for x in train_index.tolist()))
    print(" ".join(str(x) for x in test_index.tolist()))
