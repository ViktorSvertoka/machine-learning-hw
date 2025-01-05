# Імпорт необхідних пакетів
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt


# Завантаження набору даних
with open("mod_05_topic_10_various_data.pkl", "rb") as fl:
    datasets = pickle.load(fl)

# Перевірка доступних даних
print("Available datasets:", datasets.keys())

# Вибір набору даних Concrete
data = datasets["concrete"]
print(data.head())

# Створення нової ознаки Components (виключаємо CompressiveStrength)
components_columns = data.columns.difference(["CompressiveStrength"])
data["Components"] = data[components_columns].apply(lambda row: sum(row != 0), axis=1)

# Нормалізація даних
# Не видаляємо 'Components', але виключаємо 'Cluster' та 'CompressiveStrength'
features = data.drop(columns=["Cluster", "CompressiveStrength"], errors="ignore")

scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

# Визначення оптимальної кількості кластерів
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(1, 10))
visualizer.fit(normalized_data)
visualizer.show()

# Отримання оптимальної кількості кластерів
optimal_k = visualizer.elbow_value_
print(f"Optimal number of clusters: {optimal_k}")

# Кластеризація методом k-середніх
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Додавання міток до даних
data["Cluster"] = labels

# Розрахунок описової статистики кластерів
cluster_summary = data.groupby("Cluster").median()

# Розрахунок кількості об'єктів у кожному кластері
cluster_counts = data["Cluster"].value_counts()

# Додавання кількості об'єктів до звіту
cluster_summary["Count"] = cluster_counts

# Виведення звіту
print("Cluster Summary:")
print(cluster_summary)

# Додавання таблиці
table_data = cluster_summary.reset_index()

# Додавання таблиці до графіку
table = plt.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc="center",
    loc="bottom",
    bbox=[0.0, -0.35, 1.0, 0.3],
)
plt.show()


"""
Після проведення кластерного аналізу даних щодо бетону я отримав кілька цікавих висновків, які дозволяють краще зрозуміти структуру та властивості бетону.

Оптимальна кількість кластерів:
Метод Elbow показав, що найкраща кількість кластерів для цього набору даних – 4. 
Це число забезпечує ефективний поділ даних на групи, враховуючи як якість кластеризації, так і складність моделі.

Характеристика кластерів:

- Кластер 0: 89 об'єктів, середній вміст цементу – 315.0, шлаку – 20.00.
- Кластер 1: 83 об'єкти, середній вміст цементу – 339.0, шлаку – 0.00.
- Кластер 2: 98 об'єктів, середній вміст цементу – 199.3, шлаку – 182.45.
- Кластер 3: 157 об'єктів, середній вміст цементу – 165.0, шлаку – 97.10.

Розподіл за кількістю об'єктів:
Найбільше об'єктів у кластері 3 (157), тоді як кластер 1 містить найменше (83).

Аналіз компонентості:
Показник Components, який відображає кількість ненульових значень у кожному рядку, показав, що кластер 3 має найвищу середню кількість компонентів (8.0). 
Це може свідчити про різні методи виробництва або різноманітні типи бетону в даній групі.
"""
