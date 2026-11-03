
import csv
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# =========================================================
# PROYECTO: HMM - SEGURIDAD EN RED
# PRUEBAS ESTADISTICAS DE SENSIBILIDAD DE PI
# =========================================================

CARPETA = Path(__file__).resolve().parent / "resultados" / "sensibilidad"
ENTRADA = CARPETA / "sensibilidad_repeticiones.csv"
SALIDA = CARPETA / "pruebas_sensibilidad.csv"

DISTRIBUCIONES = ["90/10", "70/30", "50/50"]
METRICAS = ["exactitud", "sensibilidad", "especificidad"]
ALFA = 0.05

if not ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ENTRADA}\n"
        "Ejecuta primero analisis_sensibilidad.py"
    )

# Leer resultados individuales
datos = {}

with open(ENTRADA, "r", encoding="utf-8-sig", newline="") as archivo:
    lector = csv.DictReader(archivo)

    for fila in lector:
        pi = fila["pi"]
        repeticion = int(fila["repeticion"])

        datos[(pi, repeticion)] = {
            metrica: float(fila[metrica])
            for metrica in METRICAS
        }

resultados = []

print("\nPRUEBAS ESTADISTICAS DE SENSIBILIDAD")
print("=" * 65)

for metrica in METRICAS:
    # Conservar solamente repeticiones con datos validos
    # en las tres distribuciones.
    repeticiones = sorted({
        rep for pi, rep in datos
        if pi == DISTRIBUCIONES[0]
    })

    grupos = [[] for _ in DISTRIBUCIONES]

    for rep in repeticiones:
        valores = []

        for pi in DISTRIBUCIONES:
            fila = datos.get((pi, rep))

            if fila is None:
                valores = []
                break

            valores.append(fila[metrica])

        if len(valores) == 3 and np.all(np.isfinite(valores)):
            for i, valor in enumerate(valores):
                grupos[i].append(valor)

    grupos = [np.asarray(g, dtype=float) for g in grupos]
    n = len(grupos[0])

    if n < 3:
        print(f"\n{metrica}: datos insuficientes")
        continue

    # Prueba global para muestras emparejadas
    estadistico, p_global = friedmanchisquare(*grupos)

    resultados.append({
        "metrica": metrica,
        "prueba": "Friedman",
        "comparacion": "90/10 vs 70/30 vs 50/50",
        "n_parejas": n,
        "estadistico": estadistico,
        "p_original": p_global,
        "p_ajustado": p_global,
        "significativo_005": p_global < ALFA
    })

    print(f"\nMETRICA: {metrica}")
    print(f"Repeticiones completas: {n}")
    print(f"Friedman: estadistico={estadistico:.4f}")
    print(f"Valor p={p_global:.8f}")

    # Comparaciones por parejas con correccion Bonferroni
    parejas = [(0, 1), (0, 2), (1, 2)]
    numero_comparaciones = len(parejas)

    for i, j in parejas:
        diferencias = grupos[i] - grupos[j]

        if np.allclose(diferencias, 0):
            estadistico_w = 0.0
            p_original = 1.0
        else:
            estadistico_w, p_original = wilcoxon(
                grupos[i],
                grupos[j],
                zero_method="wilcox",
                alternative="two-sided"
            )

        p_ajustado = min(float(p_original) * numero_comparaciones, 1.0)

        comparacion = (
            f"{DISTRIBUCIONES[i]} vs {DISTRIBUCIONES[j]}"
        )

        resultados.append({
            "metrica": metrica,
            "prueba": "Wilcoxon + Bonferroni",
            "comparacion": comparacion,
            "n_parejas": n,
            "estadistico": estadistico_w,
            "p_original": p_original,
            "p_ajustado": p_ajustado,
            "significativo_005": p_ajustado < ALFA
        })

        print(
            f"{comparacion}: "
            f"p ajustado={p_ajustado:.8f} | "
            f"significativo={p_ajustado < ALFA}"
        )

# Guardar evidencia
if resultados:
    with open(
        SALIDA, "w", encoding="utf-8-sig", newline=""
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(resultados[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    print("\n" + "=" * 65)
    print("ANALISIS FINALIZADO CORRECTAMENTE")
    print(f"Archivo generado: {SALIDA.name}")
else:
    print("No fue posible realizar las pruebas.")
