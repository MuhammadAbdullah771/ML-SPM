"""Lab Task 2: Naive Bayes on the loan dataset -> predict customers who have NOT fully paid."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, recall_score, precision_score)

# ---------- 1. Load & explore ----------
df = pd.read_csv("loan_data.csv")
print("Shape:", df.shape, "\n"); print(df.head(), "\n"); df.info()
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().any().any() else "None")
print("\nSummary statistics:\n", df.describe().T.round(2))
print("\nTarget distribution (not.fully.paid):\n", df["not.fully.paid"].value_counts())
print((df["not.fully.paid"].value_counts(normalize=True)*100).round(1).astype(str) + "%")
print("\nDefault rate by purpose:\n",
      df.groupby("purpose")["not.fully.paid"].mean().sort_values(ascending=False).round(3))

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
sns.countplot(data=df, x="not.fully.paid", ax=ax[0, 0]); ax[0, 0].set_title("Target distribution")
sns.countplot(data=df, y="purpose", hue="not.fully.paid", ax=ax[0, 1]); ax[0, 1].set_title("Loan purpose vs outcome")
sns.histplot(data=df, x="fico", hue="not.fully.paid", bins=30, kde=True, ax=ax[1, 0]); ax[1, 0].set_title("FICO score vs outcome")
sns.boxplot(data=df, x="not.fully.paid", y="int.rate", ax=ax[1, 1]); ax[1, 1].set_title("Interest rate vs outcome")
plt.tight_layout(); plt.savefig("loan_eda.png", dpi=120); plt.close()

plt.figure(figsize=(11, 8))
sns.heatmap(df.drop(columns="purpose").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("loan_correlation.png", dpi=120); plt.close()
print("\nSaved: loan_eda.png, loan_correlation.png")

# ---------- 2. Preprocess ----------
data = pd.get_dummies(df, columns=["purpose"], drop_first=True)
X = data.drop(columns="not.fully.paid"); y = data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# ---------- 3. Train Gaussian Naive Bayes ----------
nb = GaussianNB().fit(X_train, y_train)
pred = nb.predict(X_test); proba = nb.predict_proba(X_test)[:, 1]
print("\n===== GaussianNB (default 0.5 threshold) =====")
print("Accuracy:", round(accuracy_score(y_test, pred), 4), "| ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Fully paid", "Not fully paid"], zero_division=0))

# ---------- 4. Handle class imbalance: tune decision threshold ----------
print("Threshold tuning (target = catching defaulters):")
best_t, best_f1 = 0.5, 0
for t in np.arange(0.05, 0.6, 0.05):
    p = (proba >= t).astype(int)
    pr, rc = precision_score(y_test, p, zero_division=0), recall_score(y_test, p)
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
    print(f"  t={t:.2f} precision={pr:.3f} recall={rc:.3f} f1={f1:.3f}")
    if f1 > best_f1: best_t, best_f1 = t, f1
print(f"Best threshold by F1: {best_t:.2f}")
pred_t = (proba >= best_t).astype(int)
print(f"\n===== GaussianNB (threshold {best_t:.2f}) =====")
print("Accuracy:", round(accuracy_score(y_test, pred_t), 4))
print(confusion_matrix(y_test, pred_t))
print(classification_report(y_test, pred_t, target_names=["Fully paid", "Not fully paid"], zero_division=0))

# ---------- 5. Sample predictions ----------
sample = X_test.head(10)
out = pd.DataFrame({"Actual": y_test.head(10).values,
                    "Predicted": (nb.predict_proba(sample)[:, 1] >= best_t).astype(int),
                    "P(not fully paid)": nb.predict_proba(sample)[:, 1].round(3)})
print("Sample predictions (1 = not fully paid):\n", out.to_string(index=False))
