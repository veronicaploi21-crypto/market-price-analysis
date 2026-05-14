import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Определяем путь к файлу processed_dataset.csv
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "processed_dataset.csv")

# Загружаем обработанный датасет
df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)



# 1. РЕГРЕССИЯ: предсказание цены price


X_reg = df.drop(columns=["price", "price_tier_encoded"], errors="ignore")
y_reg = df["price"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        random_state=42
    )
}

regression_results = []

for name, model in regression_models.items():
    print("\nRegression model:", name)

    # Обучаем модель
    model.fit(X_train_reg, y_train_reg)

    # Предсказываем цену
    y_pred_reg = model.predict(X_test_reg)

    # Считаем метрики регрессии
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = mean_squared_error(y_test_reg, y_pred_reg) ** 0.5
    r2 = r2_score(y_test_reg, y_pred_reg)

    # Кросс-валидация cv=5
    cv_scores = cross_val_score(
        model,
        X_reg,
        y_reg,
        cv=5,
        scoring="r2"
    )

    regression_results.append({
        "task": "Regression",
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None
    })

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
    print("CV R2 mean:", round(cv_scores.mean(), 3))



# 2. КЛАССИФИКАЦИЯ: предсказание price_tier_encoded


X_clf = df.drop(columns=["price", "price_tier_encoded"], errors="ignore")
y_clf = df["price_tier_encoded"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting Classifier": GradientBoostingClassifier(
        random_state=42
    )
}

classification_results = []

for name, model in classification_models.items():
    print("\nClassification model:", name)

    # Обучаем модель
    model.fit(X_train_clf, y_train_clf)

    # Предсказываем класс ценовой категории
    y_pred_clf = model.predict(X_test_clf)

    # Считаем метрики классификации
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    precision = precision_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    recall = recall_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    f1 = f1_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)

    classification_results.append({
        "task": "Classification",
        "model": name,
        "MAE": None,
        "RMSE": None,
        "R2": None,
        "CV_R2_mean": None,
        "CV_R2_std": None,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))



# 3. ОБЩАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ


all_results = regression_results + classification_results
results_df = pd.DataFrame(all_results)

results_path = os.path.join(base_dir, "evaluation_results.csv")
results_df.to_csv(results_path, index=False)

print("\nEvaluation completed successfully!")
print(results_df)
print("Results saved as:", results_path)



# 4. ВЫБОР ЛУЧШЕЙ РЕГРЕССИОННОЙ МОДЕЛИ


regression_df = results_df[results_df["task"] == "Regression"]
best_regression_model = regression_df.sort_values(by="R2", ascending=False).iloc[0]

print("\nBest regression model:")
print("Model:", best_regression_model["model"])
print("R2:", round(best_regression_model["R2"], 3))
print("MAE:", round(best_regression_model["MAE"], 3))
print("RMSE:", round(best_regression_model["RMSE"], 3))

if best_regression_model["R2"] > 0.65:
    print("The best model satisfies the requirement: R2 > 0.65")
else:
    print("The best model does not satisfy the requirement: R2 > 0.65")



# 5. ВЫБОР ЛУЧШЕЙ КЛАССИФИКАЦИОННОЙ МОДЕЛИ


classification_df = results_df[results_df["task"] == "Classification"]
best_classification_model = classification_df.sort_values(by="F1", ascending=False).iloc[0]

print("\nBest classification model:")
print("Model:", best_classification_model["model"])
print("Accuracy:", round(best_classification_model["Accuracy"], 3))
print("Precision:", round(best_classification_model["Precision"], 3))
print("Recall:", round(best_classification_model["Recall"], 3))
print("F1:", round(best_classification_model["F1"], 3))



# 6. КОММЕНТАРИИ ДЛЯ САМОПРОВЕРКИ B3 И B4


print("\nB3 comment:")
print("The feature price_per_100g is important because it directly describes the normalized product price.")
print("It helps the model compare products with different weights and usually improves prediction quality.")

print("\nB4 comment:")
print("One-Hot Encoding converts categorical variables into numeric columns.")
print("After encoding, the number of columns increases, but machine learning models can correctly use category, brand, country, packaging and store features.")
