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

# 3. Her Özellik için Ayrı Bir Kutu Grafiği
for i, col in enumerate(df.columns):
    row_index = i // cols + 1  # Alt grafik satırı
    col_index = i % cols + 1   # Alt grafik sütunu
    
    # Kutu grafiği izini (trace) ekliyoruz
    fig.add_trace(
        go.Box(
            y=df[col],
            name=col,
            boxpoints='outliers',      # Sadece aykırı noktaları göster
            marker=dict(size=3),       # Nokta boyutu
            line=dict(width=1),        # Kutu kenar kalınlığı
            showlegend=False           # Her alt grafikte tekrar efsane (legend) göstermeye gerek yok
        ),
        row=row_index,
        col=col_index
    )

# 4. Genel Görünüm Ayarları
fig.update_layout(
    title_text="Breast Cancer Veri Seti: Her Özellik için Kutu Grafikleri",
    height=1800,   # Grafiğin yüksekliğini artırarak daha okunabilir hale getiriyoruz
    width=1200,    # İsteğe bağlı genişlik
    font=dict(size=12),
    showlegend=False
)

# Alt grafiklerin y ekseni başlığı ve x ekseni başlığını gereksiz hale getirebilirsiniz,
# çünkü her kutu grafiği tek boyutlu bir dağılım gösteriyor.
# Dilerseniz "Value" veya "Feature" gibi bir şey ekleyebilirsiniz.
# Burada sade bir görünüm için bırakıyoruz.
for i in range(rows * cols):
    row_index = i // cols + 1
    col_index = i % cols + 1
    
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", row=row_index, col=col_index)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", row=row_index, col=col_index)

# 5. Grafiği Göster
fig.show()
