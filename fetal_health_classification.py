import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fetal_health.csv")

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Range of components for PCA
n_components_range = range(1, X.shape[1] + 1)

# Store results
mean_cv_scores = []
test_accuracies = []
f1_scores = []

# Evaluate for each number of PCA components
for n in n_components_range:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    rf.fit(X_train_pca, y_train)

    # Cross-validation
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_scores = cross_val_score(rf, X_train_pca, y_train, cv=cv, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    mean_cv_scores.append(mean_cv_score)

    # Test accuracy
    y_pred = rf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_accuracy)

    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)

# Results DataFrame
results_df = pd.DataFrame({
    "n_components": list(n_components_range),
    "Mean CV Accuracy": mean_cv_scores,
    "Test Accuracy": test_accuracies,
    "F1 Score": f1_scores
})

print(results_df)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(n_components_range, mean_cv_scores, label='Mean CV Accuracy', marker='o')
plt.plot(n_components_range, test_accuracies, label='Test Accuracy', marker='x')
plt.plot(n_components_range, f1_scores, label='F1 Score', marker='s')
plt.xlabel('Number of Principal Components')
plt.ylabel('Score')
plt.title('Model Performance for Different Number of Principal Components')
plt.legend()
plt.grid(True)
plt.show()
