import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# ====================== 1. VERİ HAZIRLAMA ======================
data = load_breast_cancer()
X, y = data.data, data.target

# Eğitim / test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# ====================== 2. PIPELINE OLUŞTURMA ======================
# StandardScaler + SVC
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42))
])

# GridSearch için hiperparametre aralığı
param_grid = {
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.1, 1, 10],
    "svc__degree": [2, 3]  # 'poly' kernel için dereceler
}

# ====================== 3. MODEL EĞİTİMİ (GridSearchCV) ======================
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print(f"En iyi CV skoru: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ====================== 4. TAHMİN VE DEĞERLENDİRME ======================
y_pred = best_model.predict(X_test)
accuracy = best_model.score(X_test, y_test)

print("\n-- Test Seti Sonuçları --")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["Kanser Yok", "Kanser Var"]))

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cm)

# Tahmin olasılıkları (ROC ve PR eğrileri için)
y_score = best_model.predict_proba(X_test)[:, 1]

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Precision-Recall Eğrisi
precision, recall, _ = precision_recall_curve(y_test, y_score)

# ====================== 5. GELİŞMİŞ PLOTLY GÖRSELLEŞTİRME ======================
# 3 alt grafik (1 satır, 3 sütun): 
#   1) Karışıklık Matrisi, 2) ROC Eğrisi, 3) Precision-Recall Eğrisi
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Karışıklık Matrisi", "ROC Eğrisi", "Precision-Recall Eğrisi"],
    horizontal_spacing=0.08
)

# -----------------------------------------------------------
# (a) Karışıklık Matrisi (Heatmap)
# -----------------------------------------------------------
# Confusion Matrix'i DataFrame'e dönüştürerek annot için metin hazırlama
cm_df = pd.DataFrame(cm, index=["Gerçek: Yok", "Gerçek: Var"], columns=["Tahmin: Yok", "Tahmin: Var"])
annot_text = cm_df.values.astype(str)

heatmap = go.Heatmap(
    z=cm_df.values,
    x=cm_df.columns,
    y=cm_df.index,
    colorscale="Blues",
    showscale=True,
    text=annot_text,
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{z}<extra></extra>"
)

fig.add_trace(heatmap, row=1, col=1)

# -----------------------------------------------------------
# (b) ROC Eğrisi
# -----------------------------------------------------------
roc_curve_trace = go.Scatter(
    x=fpr,
    y=tpr,
    mode="lines",
    line=dict(color="firebrick", width=3),
    name=f"ROC AUC = {roc_auc:.2f}",
    hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"
)

# Rastgele sınıflandırma çizgisi
roc_line = go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    line=dict(color="gray", dash="dash"),
    showlegend=False
)

fig.add_trace(roc_curve_trace, row=1, col=2)
fig.add_trace(roc_line, row=1, col=2)

# -----------------------------------------------------------
# (c) Precision-Recall Eğrisi
# -----------------------------------------------------------
pr_curve_trace = go.Scatter(
    x=recall,
    y=precision,
    mode="lines",
    line=dict(color="green", width=3),
    hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
    name="Precision-Recall"
)

fig.add_trace(pr_curve_trace, row=1, col=3)

# ====================== 6. DÜZENLEMELER ======================
# Genel başlık ve layout ayarları
fig.update_layout(
    title_text="<b>SVM Performans Analizi (Breast Cancer)</b>",
    title_x=0.5,  # Ortaya hizalama
    font=dict(size=14),
    plot_bgcolor="white"
)

# X/Y eksen etiketleri
fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

fig.update_xaxes(title_text="Recall", row=1, col=3)
fig.update_yaxes(title_text="Precision", row=1, col=3)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

# Görseli etkileşimli göstermek için
fig.show()
