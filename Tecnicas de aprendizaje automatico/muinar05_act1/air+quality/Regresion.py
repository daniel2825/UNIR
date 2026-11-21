import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1. Importar librerías y cargar el dataset

print("# Paso 1. Importar librerías y cargar el dataset")
df = pd.read_csv(
    "AirQualityUCI.csv",
    sep=";",
    decimal=","
)

df = df.dropna(axis=1, how="all")
df = df.dropna(axis=0, how="all")

df.head()


# Paso 2. Identificar las dimensiones

print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# Paso 3. Identificar los tipos de datos

df.info()
print(df.dtypes)

# Paso 4. Identificar variables categóricas y continuas

print(df.select_dtypes(include=["object", "category"]).columns)

print(df.select_dtypes(include=["number"]).columns)

# Paso 5. Analizar los valores faltantes

print(df.isnull().sum())

columnas_numericas = df.select_dtypes(
    include=["number"]
).columns

df[columnas_numericas] = df[columnas_numericas].replace(
    -200, np.nan
)

print(df.isnull().sum())

# Paso 6. Seleccionar la variable objetivo

objetivo = "CO(GT)"

# Paso 7. Analizar la distribución de CO, NOx y NO2

columnas = ["CO(GT)", "NOx(GT)", "NO2(GT)"]

print(df[columnas].describe())

df[columnas].hist(
    bins=30,
    figsize=(12, 4)
)

plt.tight_layout()
plt.show()

# Paso 8. Analizar las correlaciones

correlacion = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlacion,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.show()

# Paso 9. Identificar las tres variables más correlacionadas

correlaciones_objetivo = (
    correlacion[objetivo]
    .drop(objetivo)
    .dropna()
)

top_3 = (
    correlaciones_objetivo
    .abs()
    .sort_values(ascending=False)
    .head(3)
)

print(top_3)

# Paso 10. Identificar variables con correlación débil

print(
    correlaciones_objetivo
    .abs()
    .sort_values()
    .head(5)
)

# Paso 11. Redactar las conclusiones del EDA

# Parte 2. Implementar la regresión lineal

# Paso 12. Preparar los datos 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X = df.select_dtypes(include=["number"]).drop(
    columns=[objetivo]
)

y = df[objetivo]

# Eliminar registros sin variable objetivo
validos = y.notna()
X = X.loc[validos]
y = y.loc[validos]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Imputar valores faltantes
imputador = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputador.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputador.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# Escalar las variables
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# Paso 13. Regresión lineal simple

from sklearn.linear_model import LinearRegression

variable_simple = top_3.index[0]

modelo_simple = LinearRegression()

modelo_simple.fit(
    X_train_scaled[[variable_simple]],
    y_train
)

# Paso 14. Interpretar los parámetros

print("Intercepto:", modelo_simple.intercept_)
print("Coeficiente:", modelo_simple.coef_)

# Paso 15. Realizar predicciones

pred_simple = modelo_simple.predict(
    X_test_scaled[[variable_simple]]
)

# Paso 16. Evaluar el error

from sklearn.metrics import mean_absolute_error, r2_score

mae_simple = mean_absolute_error(
    y_test, pred_simple
)

r2_simple = r2_score(
    y_test, pred_simple
)

print("MAE:", mae_simple)
print("R²:", r2_simple)

# Paso 17. Regresión lineal múltiple

modelo_multiple = LinearRegression()

modelo_multiple.fit(
    X_train_scaled,
    y_train
)

pred_multiple = modelo_multiple.predict(
    X_test_scaled
)

mae_multiple = mean_absolute_error(
    y_test, pred_multiple
)

r2_multiple = r2_score(
    y_test, pred_multiple
)

print("MAE:", mae_multiple)
print("R²:", r2_multiple)

# Paso 18. Comparar los dos modelos

# Paso 19. Explicar los hiperparámetros

# Paso 20. Entrenar el árbol de decisión

from sklearn.tree import DecisionTreeRegressor

modelo_arbol = DecisionTreeRegressor(
    criterion="poisson",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features=None,
    random_state=42
)

modelo_arbol.fit(X_train, y_train)

pred_arbol = modelo_arbol.predict(X_test)

# Paso 21. Calcular MAE, R² y RMSLE

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_log_error
)

mae_arbol = mean_absolute_error(
    y_test, pred_arbol
)

r2_arbol = r2_score(
    y_test, pred_arbol
)

rmsle_arbol = np.sqrt(
    mean_squared_log_error(
        y_test, pred_arbol
    )
)

print("MAE:", mae_arbol)
print("R²:", r2_arbol)
print("RMSLE:", rmsle_arbol)

# Paso 22. Comprobar si existe overfitting

pred_train = modelo_arbol.predict(X_train)

mae_train = mean_absolute_error(
    y_train, pred_train
)

mae_test = mean_absolute_error(
    y_test, pred_arbol
)

print("MAE entrenamiento:", mae_train)
print("MAE prueba:", mae_test)


# Paso 23. Comparar el árbol con los modelos lineales

resultados = pd.DataFrame({
    "Modelo": [
        "Regresión simple",
        "Regresión múltiple",
        "Árbol de decisión"
    ],
    "MAE": [
        mae_simple,
        mae_multiple,
        mae_arbol
    ],
    "R²": [
        r2_simple,
        r2_multiple,
        r2_arbol
    ]
})

print(resultados)

# Paso 24. Determinar el mejor modelo

sns.barplot(
    data=resultados,
    x="Modelo",
    y="MAE"
)

plt.xticks(rotation=15)
plt.title("Comparación del MAE")
plt.show()

# 