import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_csv("transactions.csv")

# 2. Разделение признаков и целевой переменной
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# 3. Балансировка классов с помощью SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 5. Снижение размерности с помощью PCA (для визуализации)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Визуализация после PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_resampled, alpha=0.3, palette='Set1')
plt.title("PCA после SMOTE")
plt.savefig("pca_plot.png")
plt.close()

# 7. Обучение модели RandomForest
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Оценка качества
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_proba)
print(f"AUC ROC: {auc:.4f}")

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая Random Forest")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
