import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


# путь к папке, где лежит сам preprocessing.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# CSV должен лежать рядом с этим файлом
input_path = os.path.join(base_dir, "ndataset.csv")
output_path = os.path.join(base_dir, "processed_dataset.csv")

# загрузка датасета
df = pd.read_csv(input_path)

# убираем лишние пробелы в названиях колонок
df.columns = df.columns.str.strip()

print("Dataset loaded successfully")
print("Original shape:", df.shape)


# 1. ОБРАБОТКА ПРОПУСКОВ
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("Missing values processed")


# 2. ОБЪЕДИНЕНИЕ РЕДКИХ БРЕНДОВ В 'Прочие'
if "brand" in df.columns:
    brand_counts = df["brand"].value_counts()
    rare_brands = brand_counts[brand_counts < 5].index
    df["brand"] = df["brand"].replace(rare_brands, "Прочие")
    print("Rare brands replaced:", len(rare_brands))
else:
    print("Column 'brand' not found")


# 3. ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ
y_price = df["price"]

label_encoder = LabelEncoder()
df["price_tier_encoded"] = label_encoder.fit_transform(df["price_tier"])


# 4. ПРИЗНАКИ ДЛЯ ОБУЧЕНИЯ
X = df.drop(
    columns=["product_id", "price", "price_tier", "price_tier_encoded"],
    errors="ignore"
)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)


# 5. КОДИРОВКА + НОРМАЛИЗАЦИЯ
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)


# 6. НАЗВАНИЯ НОВЫХ КОЛОНОК
cat_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
final_columns = numeric_features + list(cat_columns)


# 7. ПРЕОБРАЗОВАНИЕ В DATAFRAME
X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
    columns=final_columns
)


# 8. ДОБАВЛЯЕМ ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ ОБРАТНО
X_processed_df["price"] = y_price.values
X_processed_df["price_tier_encoded"] = df["price_tier_encoded"].values


# 9. СОХРАНЕНИЕ
X_processed_df.to_csv(output_path, index=False)

print("Preprocessing completed successfully!")
print("Processed shape:", X_processed_df.shape)
print("Saved as:", output_path)