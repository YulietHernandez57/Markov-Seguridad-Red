
import csv
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

# 
# ANALISIS ESTADISTICO DEL EXPERIMENTO HMM
# 

CARPETA = Path(__file__).resolve().parent / "resultados"
ARCHIVO = CARPETA / "resultados_repeticiones.csv"

ESCENARIOS = [
    "Actividad habitual",
    "Mayor actividad sospechosa",
    "Observaciones ambiguas"
]

# 1. Leer resultados de las 300 simulaciones
datos = {nombre: [] for nombre in ESCENARIOS}

with open(ARCHIVO, encoding="utf-8-sig", newline="") as archivo:
    lector = csv.DictReader(archivo)

    for fila in lector:
        nombre = fila["escenario"]
        if nombre in datos:
            datos[nombre].append(float(fila["exactitud"]))

# 2. verificar cantidad de datos
print("\nANALISIS ESTADISTICO DEL HMM")
print("=" * 60)

for nombre, valores in datos.items():
    print(f"{nombre}: {len(valores)} repeticiones")

if any(len(valores) < 2 for valores in datos.values()):
    raise ValueError("No hay suficientes datos para comparar.")

# 3. Prueba global de Kruskal-Wallis
estadistico, p_valor = kruskal(
    *(datos[nombre] for nombre in ESCENARIOS)
)

print("\nPRUEBA DE KRUSKAL-WALLIS")
print("-" * 60)
print(f"Estadistico H: {estadistico:.4f}")
print(f"Valor p: {p_valor:.8g}")

if p_valor < 0.05:
    print("Resultado: existen diferencias estadisticamente significativas.")
else:
    print("Resultado: no se encontraron diferencias significativas.")

# 4. Comparaciones por parejas
# Correccion de Bonferroni por 3 comparaciones

print("\nCOMPARACIONES POR PAREJAS")
print("-" * 60)

comparaciones = [
    (ESCENARIOS[0], ESCENARIOS[1]),
    (ESCENARIOS[0], ESCENARIOS[2]),
    (ESCENARIOS[1], ESCENARIOS[2])
]

resultados = []

for nombre_a, nombre_b in comparaciones:

    u, p = mannwhitneyu(
        datos[nombre_a],
        datos[nombre_b],
        alternative="two-sided",
        method="asymptotic"
    )

    p_ajustado = min(p * len(comparaciones), 1.0)

    resultado = {
        "escenario_1": nombre_a,
        "escenario_2": nombre_b,
        "estadistico_U": float(u),
        "p_original": float(p),
        "p_ajustado_bonferroni": float(p_ajustado),
        "significativo": p_ajustado < 0.05
    }

    resultados.append(resultado)

    print(f"\n{nombre_a} vs. {nombre_b}")
    print(f"Estadistico U: {u:.2f}")
    print(f"Valor p ajustado: {p_ajustado:.8g}")

    if p_ajustado < 0.05:
        print("Diferencia significativa.")
    else:
        print("Diferencia no significativa.")

# 5. Guardar resultados estadisticos

ruta_salida = CARPETA / "pruebas_estadisticas.csv"

with open(
    ruta_salida,
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(resultados[0].keys())
    )

    escritor.writeheader()
    escritor.writerows(resultados)

print("\n" + "=" * 60)
print("Analisis finalizado correctamente.")
print(f"Archivo guardado: {ruta_salida.name}")
