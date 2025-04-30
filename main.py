import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler



import seaborn as sns
import matplotlib.pyplot as plt

# === Load and Label Data ===
df = pd.read_csv("Data/vowel-context.data", header=None, delim_whitespace=True)
df.columns = ['Set', 'Speaker', 'Sex'] + [f'Feature_{i}' for i in range(10)] + ['Class']

# === Drop Irrelevant Columns (keep 'Set' for splitting) ===
df = df.drop(columns=['Speaker', 'Sex'])

# === Split Data Based on 'Set' Column ===
train_df = df[df['Set'] == 0].drop(columns=['Set'])
test_df = df[df['Set'] == 1].drop(columns=['Set'])

# # === Outlier Removal Function ===
# def remove_outliers(df):
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     for col in numeric_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         df = df[(df[col] >= lower) & (df[col] <= upper)]
#     return df






# # === Visualize Features Before and After Outlier Removal ===
# features = [f'Feature_{i}' for i in range(10)]

# # Create a copy of the original training data before removing outliers
# train_df_before = train_df.copy()
# train_df_after = remove_outliers(train_df.copy())  # For visualization only

# # Plot side-by-side boxplots
# fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# sns.boxplot(data=train_df_before[features], ax=axes[0], palette="Set2")
# axes[0].set_title('Before Outlier Removal')
# axes[0].set_ylabel('Value')

# sns.boxplot(data=train_df_after[features], ax=axes[1], palette="Set2")
# axes[1].set_title('After Outlier Removal')
# axes[1].set_ylabel('Value')
# axes[1].set_xticklabels(features, rotation=45)

# plt.tight_layout()
# plt.show()







# === Remove Outliers Only from Training Data ===
# train_df = remove_outliers(train_df)

# === Define Features and Labels ===
X_train = train_df.drop(columns='Class')
y_train = train_df['Class']

# === Balance the Training Data ===

X_test = test_df.drop(columns='Class')
y_test = test_df['Class']

# === Define Hyperparameter Grid ===
param_grid = {
    'n_estimators': [10],  # fixed at 10 trees
    'max_samples': [0.25, 0.4, 0.6, 0.75, 0.9],
    'max_features': ['sqrt', 0.1, 0.5, 0.8, 0.9, 1.0]
}

# === Grid Search ===
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# === Evaluate Best Model ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")

# === Plot Results ===
scores = grid_search.cv_results_['mean_test_score']
# Reshape the grid search results for correct plotting
scores_matrix = scores.reshape(len(param_grid['max_samples']), len(param_grid['max_features']))

plt.figure(figsize=(10, 6))
plt.imshow(scores_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cross-Val Accuracy')

# Annotate each score on the grid
for i, ms in enumerate(param_grid['max_samples']):
    for j, mf in enumerate(param_grid['max_features']):
        plt.text(j, i, f"{scores_matrix[i, j]:.3f}", ha='center', va='center', color='white')

plt.title("Random Forest Accuracy (10 Trees)")
plt.xlabel("max_features")
plt.ylabel("max_samples")
plt.xticks(range(len(param_grid['max_features'])), param_grid['max_features'], rotation=45)
plt.yticks(range(len(param_grid['max_samples'])), param_grid['max_samples'])
plt.tight_layout()
plt.show()


sns.countplot(x='Class', data=df)
plt.title('Distribuția claselor (Class)')
plt.xticks(rotation=45)
plt.show()
