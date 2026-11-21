
import pathlib
import spacy
import pandas as pd
from spacy import displacy
import csv
import es_core_news_md

# Cargar el modelo de lenguaje español de spaCy
nlp = es_core_news_md.load()

# Leer el dataset y limitar el análisis a 500 registros
filename = "./02Dataset_anonimizado.csv"
lines_number = 500

data = pd.read_csv(
    filename, delimiter=';', encoding='utf-8',
    low_memory=False, encoding_errors="ignore"
)
lines_number = min(lines_number, len(data))

# Inicializar contadores generales
total_odio = 0
total_no_odio = 0
total_sin_clasificar = 0
total_palabras = 0
palabras_odio = 0
palabras_no_odio = 0
no_odio_con_ner = 0
odio_con_ner = 0

# Contadores de signos de exclamación
exclamaciones_odio = 0
exclamaciones_no_odio = 0

# Acumuladores de riqueza léxica
riqueza_lexica_odio = 0
riqueza_lexica_no_odio = 0

# Contadores de pronombres y palabras por grupo
pronombres_odio = 0
pronombres_no_odio = 0
palabras_pronombres_odio = 0
palabras_pronombres_no_odio = 0

# Definir algunas palabras para detectar coincidencias de odio
palabras_diccionario_odio = {
    "idiota", "estúpido", "imbécil", "inútil",
    "basura", "ignorante", "asco", "detestable"
}

# Inicializar contadores de frecuencia del diccionario
frecuencia_terminos_odio = 0
frecuencia_terminos_no_odio = 0
comentarios_con_terminos_odio = 0
comentarios_con_terminos_no_odio = 0
frecuencia_por_termino = {}

# Diccionario para corregir errores de codificación de caracteres
replacements = {
    "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó",
    "Ãº": "ú", "â": "a", "Ã‰": "É", "Ã“": "Ó",
    "Ãš": "Ú", "Ã±": "ñ", "Ã‘": "Ñ", "í±": "ñ",
    "í§": "ç", "Ã§": "ç", "Ã¼": "ü", "í¼": "ü",
    "â€œ": "“", "â€": "”", "â€˜": "‘", "â€™": "’",
    "â€“": "–", "â€”": "—", "â€¦": "…",
    "Â¿": "¿", "Â¡": "¡", "Ã¿": "¿",
    "Ã¢ÂÂ": "'", "Ã¢ÂÂ,": ",", "Ã¢ÂÂ.": ".",
    "Ã¢ÂÂ¦": "…", "Ã¢ÂÂ ": " ", "ÃÂ": "-"
}

# Corregir caracteres dañados y reemplazar valores vacíos
def fix_mojibake(s):
    if pd.isna(s):
        return ""

    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s

data["CONTENIDO A ANALIZAR"] = [
    fix_mojibake(t)
    for t in data["CONTENIDO A ANALIZAR"].values
]

# Convertir la intensidad a números para clasificar los comentarios
data["INTENSIDAD"] = pd.to_numeric(
    data["INTENSIDAD"], errors="coerce"
)

# Guardar los textos procesados y sus valores de intensidad
doc = []
value = []

for i in range(0, lines_number):

    # Analizar lingüísticamente el comentario con spaCy
    tmp_doc = nlp(data["CONTENIDO A ANALIZAR"][i])
    doc.append(tmp_doc)

    # Obtener la etiqueta numérica del comentario
    tmp_value = data["INTENSIDAD"][i]
    value.append(tmp_value)

    # Contar palabras excluyendo espacios y signos de puntuación
    numero_palabras = sum(
        1 for token in tmp_doc
        if not token.is_space and not token.is_punct
    )
    total_palabras += numero_palabras

    # Clasificar el comentario según su intensidad
    if tmp_value > 0:
        total_odio += 1
        palabras_odio += numero_palabras
        print(f"Comentario {i+1}: Odio")

    elif tmp_value == 0:
        total_no_odio += 1
        palabras_no_odio += numero_palabras
        print(f"Comentario {i+1}: No odio")

    else:
        total_sin_clasificar += 1
        print(f"Comentario {i+1}: Sin clasificación")

    # Comprobar si spaCy detectó entidades nombradas (NER)
    tiene_ner = len(tmp_doc.ents) > 0

    # IMPORTANTE: no volver a incrementar total_odio ni
    # total_no_odio aquí, porque ya se incrementaron arriba.
    if tmp_value > 0:
        if tiene_ner:
            odio_con_ner += 1

    elif tmp_value == 0:
        if tiene_ner:
            no_odio_con_ner += 1

    # Contar signos de exclamación del comentario
    numero_exclamaciones = tmp_doc.text.count("!")

    if tmp_value > 0:
        exclamaciones_odio += numero_exclamaciones
    elif tmp_value == 0:
        exclamaciones_no_odio += numero_exclamaciones

    # Calcular riqueza léxica: lemas únicos / palabras alfabéticas
    palabras_unicas = set(
        token.lemma_.lower()
        for token in tmp_doc
        if token.is_alpha
    )

    total_palabras_lexicas = sum(
        1 for token in tmp_doc if token.is_alpha
    )

    riqueza_lexica = (
        len(palabras_unicas) / total_palabras_lexicas
        if total_palabras_lexicas > 0 else 0
    )

    # Acumular la riqueza léxica según la clasificación
    if tmp_value > 0:
        riqueza_lexica_odio += riqueza_lexica
    elif tmp_value == 0:
        riqueza_lexica_no_odio += riqueza_lexica

    # Contar pronombres identificados por spaCy
    numero_pronombres = sum(
        1 for token in tmp_doc if token.pos_ == "PRON"
    )

    # Acumular pronombres y palabras de cada grupo
    if tmp_value > 0:
        pronombres_odio += numero_pronombres
        palabras_pronombres_odio += numero_palabras
    elif tmp_value == 0:
        pronombres_no_odio += numero_pronombres
        palabras_pronombres_no_odio += numero_palabras

    # Buscar coincidencias exactas con el diccionario de odio
    terminos_en_comentario = [
        token.text.lower()
        for token in tmp_doc
        if token.is_alpha
        and token.text.lower() in palabras_diccionario_odio
    ]

    numero_terminos_odio = len(terminos_en_comentario)

    # Contar cuántas veces aparece cada término
    for termino in terminos_en_comentario:
        frecuencia_por_termino[termino] = (
            frecuencia_por_termino.get(termino, 0) + 1
        )

    # Separar las frecuencias según la intensidad del comentario
    if tmp_value > 0:
        frecuencia_terminos_odio += numero_terminos_odio
        if numero_terminos_odio > 0:
            comentarios_con_terminos_odio += 1

    elif tmp_value == 0:
        frecuencia_terminos_no_odio += numero_terminos_odio
        if numero_terminos_odio > 0:
            comentarios_con_terminos_no_odio += 1

# Calcular promedios generales y evitar divisiones por cero
promedio_palabras = (
    total_palabras / lines_number if lines_number > 0 else 0
)

porcentaje_odio_ner = (
    odio_con_ner / total_odio * 100 if total_odio > 0 else 0
)

porcentaje_no_odio_ner = (
    no_odio_con_ner / total_no_odio * 100
    if total_no_odio > 0 else 0
)

promedio_odio = (
    palabras_odio / total_odio if total_odio > 0 else 0
)

promedio_no_odio = (
    palabras_no_odio / total_no_odio if total_no_odio > 0 else 0
)

# Promedio de exclamaciones por comentario
promedio_exclamaciones_odio = (
    exclamaciones_odio / total_odio if total_odio > 0 else 0
)

promedio_exclamaciones_no_odio = (
    exclamaciones_no_odio / total_no_odio
    if total_no_odio > 0 else 0
)

# Promedio de riqueza léxica por comentario
promedio_riqueza_odio = (
    riqueza_lexica_odio / total_odio if total_odio > 0 else 0
)

promedio_riqueza_no_odio = (
    riqueza_lexica_no_odio / total_no_odio
    if total_no_odio > 0 else 0
)

# Normalizar la frecuencia de pronombres por cada 100 palabras
promedio_pronombres_odio = (
    pronombres_odio / palabras_pronombres_odio * 100
    if palabras_pronombres_odio > 0 else 0
)

promedio_pronombres_no_odio = (
    pronombres_no_odio / palabras_pronombres_no_odio * 100
    if palabras_pronombres_no_odio > 0 else 0
)

# Calcular el promedio de términos del diccionario por comentario
promedio_terminos_odio = (
    frecuencia_terminos_odio / total_odio if total_odio > 0 else 0
)

promedio_terminos_no_odio = (
    frecuencia_terminos_no_odio / total_no_odio
    if total_no_odio > 0 else 0
)

# Mostrar el resumen del análisis del corpus
print("\n===== ANÁLISIS DEL CORPUS =====")

# Número total de palabras y registros analizados
total_palabras_corpus = (
    data["CONTENIDO A ANALIZAR"]
    .dropna().astype(str).str.split().str.len().sum()
)
print("Total de palabras en el corpus:", total_palabras_corpus)
print("Total de registros del corpus:", len(data))
print("Total de registros analizados:", lines_number)

print("\n--- Estadísticas generales ---")
print("Total de palabras:", total_palabras_corpus)
print("Promedio de palabras:", round(promedio_palabras, 2))

print("\n--- Comentarios con odio ---")
print("Cantidad:", total_odio)
print("Promedio de palabras:", round(promedio_odio, 2))

print("\n--- Comentarios sin odio ---")
print("Cantidad:", total_no_odio)
print("Promedio de palabras:", round(promedio_no_odio, 2))

print("\n--- Comentarios sin clasificación ---")
print("Cantidad:", total_sin_clasificar)

# Mostrar el porcentaje de comentarios con entidades nombradas
print("Porcentaje con NER no odio:", round(porcentaje_no_odio_ner, 2), "%")
print("Porcentaje con NER odio:", round(porcentaje_odio_ner, 2), "%")

# Mostrar estadísticas de los signos de exclamación
print("\n--- Signos de exclamación ---")
print("Total en comentarios con odio:", exclamaciones_odio)
print("Promedio por comentario con odio:", round(promedio_exclamaciones_odio, 2))
print("Total en comentarios sin odio:", exclamaciones_no_odio)
print("Promedio por comentario sin odio:", round(promedio_exclamaciones_no_odio, 2))

# Comparar la diversidad de vocabulario entre los grupos
print("\n--- Riqueza léxica ---")
print("Promedio con odio:", round(promedio_riqueza_odio, 3))
print("Promedio sin odio:", round(promedio_riqueza_no_odio, 3))

# Mostrar la frecuencia de pronombres por cada 100 palabras
print("\n--- Frecuencia de pronombres ---")
print("Total en comentarios con odio:", pronombres_odio)
print("Pronombres por cada 100 palabras (ODIO):", round(promedio_pronombres_odio, 2))
print("Total en comentarios sin odio:", pronombres_no_odio)
print("Pronombres por cada 100 palabras (NO-ODIO):", round(promedio_pronombres_no_odio, 2))

# Comparar la presencia de palabras del diccionario de odio
print("\n--- Frecuencia de términos de odio ---")
print("Frecuencia en comentarios con odio:", frecuencia_terminos_odio)
print("Frecuencia en comentarios sin odio:", frecuencia_terminos_no_odio)
print("Comentarios con odio que contienen términos:", comentarios_con_terminos_odio)
print("Comentarios sin odio que contienen términos:", comentarios_con_terminos_no_odio)
print("Promedio por comentario con odio:", round(promedio_terminos_odio, 2))
print("Promedio por comentario sin odio:", round(promedio_terminos_no_odio, 2))

# Ordenar las palabras por frecuencia, de mayor a menor
print("\n--- Frecuencia por palabra ---")
for termino, frecuencia in sorted(
    frecuencia_por_termino.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{termino}: {frecuencia}")
