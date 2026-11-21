# ============================================================
# 1. LIBRERÍAS NECESARIAS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import tkinter as tk
from tkinter import filedialog
import cv2
import pandas as pd

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)


# ============================================================
# 2. SELECCIONAR Y CARGAR IMAGEN
# ============================================================

root = tk.Tk()
root.withdraw()

ruta_imagen = filedialog.askopenfilename(
    title="Seleccionar imagen",
    filetypes=[
        ("Imágenes", "*.png *.jpg *.jpeg")
    ]
)

root.destroy()

if not ruta_imagen:
    raise SystemExit("No se seleccionó ninguna imagen.")

imagen = Image.open(ruta_imagen).convert("RGB")
I_rgb = np.asarray(imagen)

print("Dimensiones:", I_rgb.shape)
print("Tipo:", I_rgb.dtype)

plt.imshow(I_rgb)
plt.title("Imagen original RGB")
plt.axis("off")
plt.show()


# ============================================================
# 3. CONVERSIÓN A ESCALA DE GRISES
# ============================================================

# Matriz 2D de intensidades entre 0 y 255
I_gray = np.asarray(imagen.convert("L"))

# Normalización al intervalo [0, 1]
I = I_gray.astype(np.float64) / 255.0

print("RGB:", I_rgb.shape, I_rgb.dtype)
print("Grises:", I_gray.shape, I_gray.dtype)
print(
    "Normalizada:",
    I.shape,
    I.dtype,
    "rango:",
    I.min(),
    I.max()
)

plt.imshow(I_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen en escala de grises")
plt.axis("off")
plt.show()


# ============================================================
# 4. TRANSFORMACIÓN ARITMÉTICA: SUMA DE LUMINOSIDAD
# ============================================================

delta = 80

J = np.clip(
    I_gray.astype(np.float64) + delta,
    0,
    255
).astype(np.uint8)

saturacion = 100 * np.mean(J == 255)

print(f"Píxeles saturados con suma +80: {saturacion:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

for a, im, titulo in zip(
    ax,
    [I_gray, J],
    ["Original", "Brillo +80"]
):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(titulo)
    a.axis("off")

plt.tight_layout()
plt.show()


# Histogramas de la suma
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].hist(I_gray.ravel(), bins=256, range=(0, 256))
ax[0].set_title("Original")

ax[1].hist(J.ravel(), bins=256, range=(0, 256))
ax[1].set_title("Brillo +80")

for a in ax:
    a.set_xlabel("Intensidad")
    a.set_ylabel("N.º de píxeles")

plt.tight_layout()
plt.show()


# ============================================================
# 5. NEGATIVO Y TRANSFORMACIONES GAMMA
# ============================================================

negativo = 1 - I

gamma_clara = I ** 0.4
gamma_oscura = I ** 2

fig, ax = plt.subplots(1, 3, figsize=(11, 3))

for a, im, titulo in zip(
    ax,
    [negativo, gamma_clara, gamma_oscura],
    ["Negativo", "Gamma 0.4", "Gamma 2"]
):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    a.set_title(titulo)
    a.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 6. TRANSFORMACIÓN LOGARÍTMICA
# ============================================================

k = 255

logaritmica = np.log1p(k * I) / np.log1p(k)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

for a, im, titulo in zip(
    ax,
    [I, logaritmica],
    ["Original", f"Logarítmica k={k}"]
):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    a.set_title(titulo)
    a.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 7. UMBRALIZACIÓN BINARIA
# ============================================================

umbrales = [10, 100, 150, 200]

fig, ax = plt.subplots(1, 5, figsize=(15, 4))

ax[0].imshow(I_gray, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original")
ax[0].axis("off")

resultados_umbral = {}

for i, T in enumerate(umbrales):

    resultado_umbral = np.where(
        I_gray >= T,
        255,
        0
    ).astype(np.uint8)

    resultados_umbral[T] = resultado_umbral

    ax[i + 1].imshow(
        resultado_umbral,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax[i + 1].set_title(f"Umbral T={T}")
    ax[i + 1].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 8. TRANSFORMACIÓN IDENTIDAD
# ============================================================

identidad = I.copy()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

for a, im, titulo in zip(
    ax,
    [I, identidad],
    ["Imagen original", "Transformación identidad"]
):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    a.set_title(titulo)
    a.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 9. COMPARACIÓN DE TRANSFORMACIONES PUNTO A PUNTO
# ============================================================

imagenes_punto = [
    I,
    gamma_clara,
    gamma_oscura,
    logaritmica
]

titulos_punto = [
    "Original",
    "Gamma 0.4",
    "Gamma 2",
    "Logarítmica"
]

fig, ax = plt.subplots(1, 4, figsize=(14, 4))

for a, im, titulo in zip(
    ax,
    imagenes_punto,
    titulos_punto
):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    a.set_title(titulo)
    a.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 10. ECUALIZACIÓN GLOBAL DEL HISTOGRAMA
# ============================================================

# 1. Calcular histograma
histograma = np.bincount(
    I_gray.ravel(),
    minlength=256
)

# 2. Normalizar
probabilidades = histograma / I_gray.size

# 3. Calcular CDF
cdf = np.cumsum(probabilidades)

# 4. Crear tabla de transformación
tabla_transformacion = np.round(
    255 * cdf
).astype(np.uint8)

# 5. Aplicar transformación
ecualizada_manual = tabla_transformacion[I_gray]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(I_gray, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Imagen original")
ax[0].axis("off")

ax[1].imshow(
    ecualizada_manual,
    cmap="gray",
    vmin=0,
    vmax=255
)
ax[1].set_title("Ecualización global manual")
ax[1].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 11. ECUALIZACIÓN PONDERADA
# ============================================================

alpha = 0.5

ecualizada_ponderada = (
    (1 - alpha) * I_gray.astype(np.float64)
    + alpha * ecualizada_manual.astype(np.float64)
)

ecualizada_ponderada = np.clip(
    ecualizada_ponderada,
    0,
    255
).astype(np.uint8)

plt.imshow(
    ecualizada_ponderada,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(f"Ecualización ajustada: alpha={alpha}")
plt.axis("off")
plt.show()


# ============================================================
# 12. CLAHE: COMPARACIÓN DE CONFIGURACIONES
# ============================================================

configuraciones = [
    (1.0, (8, 8)),
    (2.0, (8, 8)),
    (4.0, (8, 8)),
    (2.0, (4, 4)),
    (2.0, (16, 16))
]

resultados_clahe = {}

fig, ax = plt.subplots(2, 3, figsize=(13, 8))
ax = ax.ravel()

ax[0].imshow(I_gray, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original")
ax[0].axis("off")

for i, (clip, grid) in enumerate(configuraciones, start=1):

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=grid
    )

    imagen_resultado = clahe.apply(I_gray)

    resultados_clahe[(clip, grid)] = imagen_resultado

    ax[i].imshow(
        imagen_resultado,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax[i].set_title(f"Clip={clip}, Grid={grid}")
    ax[i].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 13. MÉTRICAS PARA LOS MÉTODOS SELECCIONADOS DE LA MEMORIA
# ============================================================

from pathlib import Path

def calcular_metricas(imagen):
    imagen = np.asarray(imagen)
    if imagen.dtype != np.uint8 or imagen.ndim != 2:
        raise ValueError("Se espera una imagen de grises uint8 de 2 dimensiones")
    return {
        "Intensidad media": round(float(np.mean(imagen)), 2),
        "Desviación estándar": round(float(np.std(imagen)), 2),
        "Píxeles saturados (%)": round(float(100 * np.mean(imagen == 255)), 2),
    }

# Se reutilizan los resultados de las secciones anteriores,
# sin volver a calcular las transformaciones con otros parámetros.
metodos_memoria = {
    "Original": I_gray,
    "Suma +80": J,
    "Gamma 0.4": np.round(255 * gamma_clara).astype(np.uint8),
    "Logarítmica": np.round(255 * logaritmica).astype(np.uint8),
    "Ecualización global": ecualizada_manual,
    "Ecualización 0.5": ecualizada_ponderada,
    "CLAHE clip=4, grid=(8,8)": resultados_clahe[(4.0, (8, 8))],
}

filas_metricas = []
for nombre, im in metodos_memoria.items():
    filas_metricas.append({
        "Método": nombre,
        **calcular_metricas(im),
    })

tabla_memoria = pd.DataFrame(filas_metricas)
print("\nTABLA RESUMIDA PARA LA MEMORIA\n")
print(tabla_memoria.to_string(index=False))

# ============================================================
# 14. FIGURA: IMÁGENES E HISTOGRAMAS, MISMA ESCALA
# ============================================================

n = len(metodos_memoria)
fig, ejes = plt.subplots(n, 2, figsize=(12, 3 * n))
max_frecuencia = max(
    np.bincount(im.ravel(), minlength=256).max()
    for im in metodos_memoria.values()
)

for i, (nombre, im) in enumerate(metodos_memoria.items()):
    ejes[i, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
    ejes[i, 0].set_title(nombre)
    ejes[i, 0].axis("off")

    ejes[i, 1].hist(im.ravel(), bins=256, range=(0, 256))
    ejes[i, 1].set_xlim(0, 255)
    ejes[i, 1].set_ylim(0, max_frecuencia * 1.05)
    ejes[i, 1].set_xlabel("Intensidad (0-255)")
    ejes[i, 1].set_ylabel("Número de píxeles")
    ejes[i, 1].set_title("Histograma: " + nombre)

fig.suptitle("Comparación de técnicas de mejora", fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.985))

# Archivos con el nombre de la imagen para evitar sobrescrituras.
carpeta_resultados = Path(ruta_imagen).parent / "resultados_memoria"
carpeta_resultados.mkdir(parents=True, exist_ok=True)
nombre_base = Path(ruta_imagen).stem
ruta_figura = carpeta_resultados / f"{nombre_base}_comparacion.png"
fig.savefig(ruta_figura, dpi=180, bbox_inches="tight")
plt.show()

# ============================================================
# 15. EXPORTAR TABLA PARA WORD / EXCEL
# ============================================================

ruta_csv = carpeta_resultados / f"{nombre_base}_metricas_resumidas.csv"
tabla_memoria.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

print("\nArchivos generados:")
print(ruta_csv)
print(ruta_figura)
print("\nRepite el procedimiento con las otras 3 imágenes.")
