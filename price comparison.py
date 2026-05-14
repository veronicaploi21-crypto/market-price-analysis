import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
df = pd.read_csv('фИНАЛ ЕМАЕ 2.csv')
# 1. Средняя цена по категориям и магазинам
# Группируем данные: категория + магазин
# Считаем среднюю цену
grouped = df.groupby(['category', 'store'])['price'].mean().unstack()
print("Средние цены по категориям:")
print(grouped)
# Парный boxplot
# Сравниваем распределение цен между магазинами
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='store', y='price')

plt.title('Сравнение цен в Globus и Narodny')
plt.xlabel('Магазин')
plt.ylabel('Цена')

plt.show()
# асколько один дороже другого %
mean_globus = df[df['store']=='Globus']['price'].mean()
mean_narodny = df[df['store']=='Narodny']['price'].mean()

diff_percent = ((mean_globus - mean_narodny) / mean_narodny) * 100
print("\nСредняя цена Globus:", round(mean_globus,2))
print("Средняя цена Narodny:", round(mean_narodny,2))
print("Разница (%):", round(diff_percent,2))

#T-test проверка значимости
#Проверяемзначима ли разница статистически
globus_prices = df[df['store']=='Globus']['price']
narodny_prices = df[df['store']=='Narodny']['price']

t_stat, p_value = stats.ttest_ind(globus_prices, narodny_prices)

print("\nT-test результаты:")
print("t-статистика:", round(t_stat,3))
print("p-value:", round(p_value,5))
if p_value < 0.05:
    print("Разница статистически значимая")
else:
    print("Разница НЕ значимая")