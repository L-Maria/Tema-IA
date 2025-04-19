import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# 1. Load dataset
vowel = fetch_openml(name='vowel', version=2, as_frame=True)
X = vowel.data
y = vowel.target
X = X.select_dtypes(include=[np.number])

# 2. Split into train/test (optional, just for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Define parameters
in_bag_percentages = [0.25, 0.40, 0.60, 0.75, 0.90]
feature_options = ['sqrt', 0.10, 0.50, 0.80, 0.90, 1.00]  # special case for 'sqrt'

# 4. Function to map feature options
def compute_max_features(option, n_features):
    if option == 'sqrt':
        return 'sqrt'
    else:
        return max(1, int(option * n_features))

n_features = X_train.shape[1]

# 5. Train and evaluate all combinations
results = []

for in_bag in in_bag_percentages:
    sample_size = int(in_bag * len(X_train))
    for feat_option in feature_options:
        max_features = compute_max_features(feat_option, n_features)

        # Bootstrap manually (simulate in-bag size)
        idx = np.random.choice(len(X_train), size=sample_size, replace=True)
        X_bootstrap = X_train.iloc[idx]
        y_bootstrap = y_train.iloc[idx]

        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=10,
            max_features=max_features,
            bootstrap=True,  # default bootstrap
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_bootstrap, y_bootstrap)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            'in_bag_percentage': in_bag,
            'feature_option': feat_option,
            'sample_size': sample_size,
            'max_features': max_features,
            'accuracy': acc
        })

# 6. Save results
results_df = pd.DataFrame(results)
print(results_df)



results_df.to_csv("rezultate_random_forest.csv", index=False)
print("\nRezultatele au fost salvate în fisierul 'rezultate_random_forest.csv'.")



