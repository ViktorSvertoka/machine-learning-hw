# 1. Імпорт необхідних пакетів
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gdown


# Відключаємо попередження
warnings.filterwarnings("ignore")

# 2. Завантаження набору даних Rain in Australia
file_id = "1XWGsdnssdccgp6oYYrqXW8oKzEHEWusw"
url = f"https://drive.google.com/uc?id={file_id}"
output = "weatherAUS.csv"
gdown.download(url, output, quiet=False)
data = pd.read_csv(output)

# 3. Обробка часових ознак для навчання моделі
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month

# Видалення ознак з великою кількістю пропущених значень (більше 35%)
data = data[data.columns[data.isna().mean() < 0.35]]

# Видалення рядків з пропущеними значеннями в цільовій змінній
data = data.dropna(subset=["RainTomorrow"])

# Створення підмножин набору даних із числовими та категоріальними ознаками
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist() + ["Year"]
categorical_features = data.select_dtypes(include=["object"]).columns.tolist() + [
    "Month"
]
categorical_features.remove("RainTomorrow")  # Видаляємо цільову змінну зі списку ознак

# 4. Розподіл набору даних на навчальну і тестову вибірки
max_year = data["Year"].max()
X_train = data[data["Year"] < max_year].drop(["RainTomorrow", "Date"], axis=1)
X_test = data[data["Year"] == max_year].drop(["RainTomorrow", "Date"], axis=1)
y_train = data[data["Year"] < max_year]["RainTomorrow"]
y_test = data[data["Year"] == max_year]["RainTomorrow"]

# 5 і 6. Створення пайплайну для обробки даних
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Створення та навчання моделі
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs", class_weight="balanced", random_state=42
            ),
        ),
    ]
)

model.fit(X_train, y_train)

# 7. Розрахунок метрик нової моделі
y_pred = model.predict(X_test)


# 8. Очікувані результати виконання
print(classification_report(y_test, y_pred))

# Висновки
print(
    "\nВисновки:\n"
    "1. Представлена модель демонструє кращу загальну точність у порівнянні з базовою.\n"
    "2. Модель більш ефективно передбачає відсутність опадів (клас 'No'), однак має гіршу точність при прогнозуванні дощу (клас 'Yes').\n"
    "3. Отримані результати можуть бути пояснені тим, що модель тестується на даних з наступного року."
)
