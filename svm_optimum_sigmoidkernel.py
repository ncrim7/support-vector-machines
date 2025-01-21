import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. Veri Seti: Breast Cancer
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Eğitim-Test Ayrımı (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Pipeline: Ölçeklendirme (StandardScaler) + SVC (kernel='sigmoid')
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel='sigmoid', probability=True, random_state=42))
])

# 4. Hiperparametre Arama Aralığı
# C     : Düzenleme (regularization) parametresi
# gamma : Sigmoid kernelde çarpım katsayısı
# coef0 : Sigmoid çekirdekteki ek sabit terim
param_grid = {
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": [0.01, 0.1, 1, 10, "scale"],  
    "svc__coef0": [0.0, 0.5, 1, 2]  # Sigmoid için bağımsız terim
}

# 5. GridSearchCV ile En İyi Parametre Değerlerini Bulma
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",   # Değerlendirme metriği: doğruluk
    cv=5,                 # 5 katlı çapraz doğrulama
    n_jobs=-1,            # Tüm işlemci çekirdeklerini kullan
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. En İyi Parametreler ve Performans
print("En iyi parametreler:", grid_search.best_params_)
print(f"En iyi CV (cross-val) skoru: {grid_search.best_score_:.4f}")

# 7. Test Setinde Performansı Değerlendirme
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test Seti Doğruluğu: {test_score:.4f}")
