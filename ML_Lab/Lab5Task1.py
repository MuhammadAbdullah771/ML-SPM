"""Lab Task 1: Wine classification with Gaussian vs Multinomial Naive Bayes."""
import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
wine = load_wine()
X, y = wine.data, wine.target
print("Shape:", X.shape, "| Classes:", list(wine.target_names))
print(pd.DataFrame(X, columns=wine.feature_names).describe().T.head(5), "\n")

# 2. Train / test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape, "\n")

# 3. Train both models (all wine features are non-negative, so MultinomialNB works)
models = {"GaussianNB": GaussianNB(), "MultinomialNB": MultinomialNB()}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(model, X, y, cv=5).mean()
    results[name] = (acc, cv)
    print(f"===== {name} =====")
    print(f"Test accuracy : {acc:.4f}")
    print(f"5-fold CV acc : {cv:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=wine.target_names))

# 4. Compare
best = max(results, key=lambda k: results[k][1])
print("Comparison:")
for k, (a, c) in results.items():
    print(f"  {k:14s} test={a:.4f}  cv={c:.4f}")
print(f"-> {best} performs better.")
print("Why: GaussianNB models continuous features with normal distributions, which suits "
      "wine's chemical measurements. MultinomialNB assumes count-like data, so it fits poorly.\n")

# 5. Sample predictions with the better model
model = models[best]
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
print(f"Sample predictions using {best}:")
out = pd.DataFrame({
    "Actual": [wine.target_names[i] for i in y_test[:10]],
    "Predicted": [wine.target_names[i] for i in pred[:10]],
    "Confidence": proba[:10].max(axis=1).round(3)})
out["Correct"] = out.Actual == out.Predicted
print(out.to_string(index=False))

# Predict a brand-new wine sample (first test row, as an example)
new = X_test[0].reshape(1, -1)
print("\nNew wine ->", wine.target_names[model.predict(new)[0]],
      "| probabilities:", model.predict_proba(new).round(3)[0])