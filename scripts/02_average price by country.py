import pandas as pd
import matplotlib.pyplot as plt

# загрузка данных
df = pd.read_csv('фИНАЛ ЕМАЕ 2.csv.')
# проверим, какие столбцы есть (на всякий случай)
print("Колонки:", df.columns)

#  Средняя цена по странам

country_mean = df.groupby('country')['price'].mean().sort_values()

print("\nСредняя цена по странам:")
print(country_mean)



# Bar chart

plt.figure(figsize=(10,6))
country_mean.plot(kind='bar')

plt.title('Средняя цена по странам')
plt.xlabel('Страна')
plt.ylabel('Цена')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Делим на local / import
def local_or_import(country):
    if country in ['Кыргызстан', 'Казахстан']:
        return 'local'
    else:
        return 'import'

df['origin_type'] = df['country'].apply(local_or_import)

# Средние цены
origin_mean = df.groupby('origin_type')['price'].mean()

print("\nСредние цены local vs import:")
print(origin_mean)
#  Разница в процентах (без падений)
if 'local' in origin_mean and 'import' in origin_mean:
    local_price = origin_mean['local']
    import_price = origin_mean['import']

    diff = ((import_price - local_price) / local_price) * 100
    print("\nИмпорт дороже на (%):", round(diff,2))
else:
    print("\nНедостаточно данных для сравнения local vs import")
