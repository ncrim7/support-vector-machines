import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer

# 1. Veri Setini Yükle
data = load_breast_cancer()
X = data.data
feature_names = data.feature_names

# DataFrame'e dönüştürme
df = pd.DataFrame(X, columns=feature_names)

# 2. Alt Grafiklerin (Subplots) Hazırlanması
# - Veri setinde 30 özellik (feature) olduğu için 6 satır x 5 sütun = 30 alt grafik
rows = 6
cols = 5

fig = make_subplots(
    rows=rows, 
    cols=cols, 
    subplot_titles=feature_names,
    horizontal_spacing=0.03,
    vertical_spacing=0.08
)

# 3. Her Özellik için Ayrı Bir Histogram
for i, col in enumerate(df.columns):
    row_index = i // cols + 1  # Alt grafik satırı
    col_index = i % cols + 1   # Alt grafik sütunu
    
    # Histogram izini (trace) ekliyoruz
    fig.add_trace(
        go.Histogram(
            x=df[col],
            nbinsx=30,               # İsteğe bağlı: histogramda kullanılacak bin sayısı
            marker=dict(color='teal'),
            showlegend=False
        ),
        row=row_index,
        col=col_index
    )

# 4. Genel Görünüm Ayarları
fig.update_layout(
    title_text="Breast Cancer Veri Seti: Her Özellik için Histogramlar",
    height=1800,   # Grafiğin yüksekliğini artırarak daha okunabilir hale getiriyoruz
    width=1200,    # Genişlik
    font=dict(size=12),
    showlegend=False
)

# Grid çizgileri ve ek eksen ayarları
for i in range(rows * cols):
    row_index = i // cols + 1
    col_index = i % cols + 1
    
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False, row=row_index, col=col_index)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False, row=row_index, col=col_index)

# 5. Grafiği Göster
fig.show()
