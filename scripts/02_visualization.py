import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# загрузка данных
df = pd.read_csv('ndataset.csv')
print("Размер датасета:", df.shape)
print("\nПропуски:\n", df.isnull().sum())
 #ГИСТОГРАММЫ
plt.figure(figsize=(8,5))
plt.hist(df['price_per_100g'], bins=30)
plt.title('Распределение price_per_100g')
plt.xlabel('Цена за 100г')
plt.ylabel('Частота')
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=30)
plt.title('Распределение общей цены')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()
# BOXPLOT (выбросы)
plt.figure(figsize=(8,5))
sns.boxplot(x=df['price_per_100g'])
plt.title('Boxplot price_per_100g')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['price'])
plt.title('Boxplot price')
plt.show()
# HEATMAP (корреляции)
numeric_cols = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_cols.corr(), annot=True)
plt.title('Корреляционная матрица')
plt.show()

# ВЫЯВЛЕНИЕ ВЫБРОСОВ (IQR)
Q1 = df['price_per_100g'].quantile(0.25)
Q3 = df['price_per_100g'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['price_per_100g'] < lower) | (df['price_per_100g'] > upper)]

print("\nГраницы выбросов:")
print("Нижняя:", round(lower,2))
print("Верхняя:", round(upper,2))

print("Количество выбросов:", len(outliers))
#ОПИСАНИЕ РАСПРЕДЕЛЕНИЯ
print("\nОписание price_per_100g:")
print(df['price_per_100g'].describe())
