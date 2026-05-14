import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Путь к файлам
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "processed_dataset.csv")

# Создаем папку plots, если ее нет
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Загружаем датасет
df = pd.read_csv(data_path)

# Разделяем признаки и целевую переменную
X = df.drop(columns=["price", "price_tier_encoded"], errors="ignore")
y = df["price"]

# Разделяем данные 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Используем Random Forest как лучшую и удобную модель для анализа признаков
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Обучаем модель
model.fit(X_train, y_train)

# Делаем прогноз
y_pred = model.predict(X_test)


# ==========================================
# 1. FEATURE IMPORTANCE
# ==========================================

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
plt.close()


# ==========================================
# 2. PREDICTED VS ACTUAL
# ==========================================

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Prices")

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value])

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"))
plt.close()


# ==========================================
# 3. LEARNING CURVE
# ==========================================

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    random_state=42
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, marker="o", label="Training score")
plt.plot(train_sizes, test_mean, marker="o", label="Validation score")
plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.title("Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "learning_curve.png"))
plt.close()


# ==========================================
# 4. RESIDUALS PLOT
# ==========================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "residuals_plot.png"))
plt.close()


# ==========================================
# 5. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 6))
plt.hist(y, bins=20)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "price_distribution.png"))
plt.close()


print("All plots were created successfully!")
print("R2 score:", round(r2_score(y_test, y_pred), 3))
print("Plots saved in folder:", plots_dir)
