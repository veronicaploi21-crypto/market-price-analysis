import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Определяем путь к обработанному датасету
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "processed_dataset.csv")

# Загружаем подготовленный датасет
df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# Разделяем признаки и целевую переменную
X = df.drop(columns=["price", "price_tier_encoded"], errors="ignore")
y = df["price"]


# Делим данные на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# Создаем набор моделей для обучения
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# Список для хранения результатов моделей
results = []

for name, model in models.items():
    print("\nTraining:", name)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Делаем прогноз
    y_pred = model.predict(X_test)

    # Вычисляем метрики качества
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Выполняем кросс-валидацию на 5 частях
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    # Сохраняем результаты
    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std()
    })

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
    print("CV R2 mean:", round(cv_scores.mean(), 3))


# Преобразуем результаты в DataFrame
results_df = pd.DataFrame(results)

# Сохраняем результаты в CSV файл
results_path = os.path.join(base_dir, "model_results.csv")
results_df.to_csv(results_path, index=False)

print("\nAll models trained successfully!")
print(results_df)
print("Results saved as:", results_path)