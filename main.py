
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PROYECTO: SEGURIDAD EN RED MEDIANTE HMM
# SIMULACION, VITERBI Y ANALISIS ESTADISTICO
# =========================================================

ESTADOS = ["Normal", "Sospechoso"]
OBSERVACIONES = ["Bajo", "Medio", "Alto"]

REPETICIONES = 100
HORAS = 24
SEMILLA = 42

PI = np.array([0.90, 0.10])

ESCENARIOS = {
    "Actividad habitual": {
        "A": np.array([
            [0.90, 0.10],
            [0.30, 0.70]
        ]),
        "B": np.array([
            [0.80, 0.15, 0.05],
            [0.10, 0.30, 0.60]
        ])
    },
    "Mayor actividad sospechosa": {
        "A": np.array([
            [0.70, 0.30],
            [0.15, 0.85]
        ]),
        "B": np.array([
            [0.80, 0.15, 0.05],
            [0.10, 0.30, 0.60]
        ])
    },
    "Observaciones ambiguas": {
        "A": np.array([
            [0.90, 0.10],
            [0.30, 0.70]
        ]),
        "B": np.array([
            [0.50, 0.30, 0.20],
            [0.30, 0.35, 0.35]
        ])
    }
}

CARPETA = Path(__file__).resolve().parent / "resultados"
CARPETA.mkdir(exist_ok=True)


# =========================================================
# 1. VALIDAR LAS PROBABILIDADES
# =========================================================

def validar_probabilidades(A, B, pi):
    for nombre, matriz in [("A", A), ("B", B)]:
        if not np.allclose(matriz.sum(axis=1), 1):
            raise ValueError(
                f"Las filas de {nombre} no suman 1"
            )
        if np.any(matriz <= 0):
            raise ValueError(
                f"{nombre} debe tener probabilidades positivas"
            )

    if not np.isclose(pi.sum(), 1):
        raise ValueError("PI no suma 1")

    if np.any(pi <= 0):
        raise ValueError("PI debe ser positivo")


# =========================================================
# 2. SIMULAR ACTIVIDAD DE RED
# =========================================================

def simular_hmm(A, B, pi, horas, rng):
    reales = []
    observadas = []

    estado = rng.choice(2, p=pi)

    for _ in range(horas):
        reales.append(estado)

        observacion = rng.choice(
            3, p=B[estado]
        )
        observadas.append(observacion)

        estado = rng.choice(
            2, p=A[estado]
        )

    return np.array(reales), np.array(observadas)


# =========================================================
# 3. ALGORITMO DE VITERBI
# =========================================================

def viterbi(observadas, A, B, pi):
    n = len(observadas)
    k = len(pi)

    puntuaciones = np.zeros((n, k))
    caminos = np.zeros((n, k), dtype=int)

    log_A = np.log(A)
    log_B = np.log(B)
    log_pi = np.log(pi)

    puntuaciones[0] = (
        log_pi + log_B[:, observadas[0]]
    )

    for t in range(1, n):
        for estado in range(k):
            opciones = (
                puntuaciones[t - 1]
                + log_A[:, estado]
            )

            caminos[t, estado] = np.argmax(opciones)

            puntuaciones[t, estado] = (
                np.max(opciones)
                + log_B[estado, observadas[t]]
            )

    resultado = np.zeros(n, dtype=int)
    resultado[-1] = np.argmax(puntuaciones[-1])

    for t in range(n - 2, -1, -1):
        resultado[t] = caminos[
            t + 1, resultado[t + 1]
        ]

    return resultado


# =========================================================
# 4. CALCULAR METRICAS
# =========================================================

def dividir_seguro(numerador, denominador):
    if denominador == 0:
        return float("nan")
    return 100 * numerador / denominador


def calcular_metricas(reales, estimados):
    vp = int(np.sum((reales == 1) & (estimados == 1)))
    vn = int(np.sum((reales == 0) & (estimados == 0)))
    fp = int(np.sum((reales == 0) & (estimados == 1)))
    fn = int(np.sum((reales == 1) & (estimados == 0)))

    return {
        "exactitud": dividir_seguro(
            vp + vn, len(reales)
        ),
        "sensibilidad": dividir_seguro(
            vp, vp + fn
        ),
        "precision": dividir_seguro(
            vp, vp + fp
        ),
        "especificidad": dividir_seguro(
            vn, vn + fp
        ),
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn
    }


# =========================================================
# 5. ESTADISTICAS E INTERVALO DE CONFIANZA
# =========================================================

def estadisticas(valores):
    datos = np.array(valores, dtype=float)
    datos = datos[np.isfinite(datos)]

    n = len(datos)

    if n == 0:
        return {
            "n": 0,
            "media": float("nan"),
            "desviacion": float("nan"),
            "minimo": float("nan"),
            "maximo": float("nan"),
            "ic_inferior": float("nan"),
            "ic_superior": float("nan")
        }

    media = float(np.mean(datos))

    if n > 1:
        desviacion = float(np.std(datos, ddof=1))
        margen = 1.96 * desviacion / np.sqrt(n)
    else:
        desviacion = float("nan")
        margen = float("nan")

    return {
        "n": n,
        "media": media,
        "desviacion": desviacion,
        "minimo": float(np.min(datos)),
        "maximo": float(np.max(datos)),
        "ic_inferior": media - margen,
        "ic_superior": media + margen
    }


# =========================================================
# 6. GUARDAR ARCHIVOS CSV
# =========================================================

def guardar_csv(nombre, filas):
    if not filas:
        return

    ruta = CARPETA / nombre

    with open(
        ruta, "w", newline="", encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=list(filas[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"Archivo guardado: {ruta.name}")


# =========================================================
# 7. EJECUTAR EXPERIMENTO
# =========================================================

rng = np.random.default_rng(SEMILLA)

filas_repeticiones = []
filas_horas = []
filas_resumen = []

print("\nEXPERIMENTO HMM - SEGURIDAD EN RED")
print("=" * 60)

for nombre, matrices in ESCENARIOS.items():
    A = matrices["A"]
    B = matrices["B"]

    validar_probabilidades(A, B, PI)

    resultados = []

    for repeticion in range(1, REPETICIONES + 1):
        reales, observadas = simular_hmm(
            A, B, PI, HORAS, rng
        )

        estimados = viterbi(
            observadas, A, B, PI
        )

        metricas = calcular_metricas(
            reales, estimados
        )

        fila = {
            "escenario": nombre,
            "repeticion": repeticion,
            **metricas
        }

        filas_repeticiones.append(fila)
        resultados.append(fila)

        for hora in range(HORAS):
            filas_horas.append({
                "escenario": nombre,
                "repeticion": repeticion,
                "hora": hora + 1,
                "estado_real": ESTADOS[reales[hora]],
                "observacion": OBSERVACIONES[observadas[hora]],
                "estado_estimado": ESTADOS[estimados[hora]],
                "acierto": int(
                    reales[hora] == estimados[hora]
                )
            })

    resumen = {
        "escenario": nombre,
        "repeticiones": REPETICIONES,
        "horas_por_repeticion": HORAS
    }

    for metrica in [
        "exactitud",
        "sensibilidad",
        "precision",
        "especificidad"
    ]:
        valores = [
            fila[metrica] for fila in resultados
        ]

        stats = estadisticas(valores)

        for clave, valor in stats.items():
            resumen[f"{metrica}_{clave}"] = valor

    filas_resumen.append(resumen)

    print(f"\nESCENARIO: {nombre}")
    print("-" * 60)

    for metrica in [
        "exactitud",
        "sensibilidad",
        "precision",
        "especificidad"
    ]:
        media = resumen[f"{metrica}_media"]
        n = resumen[f"{metrica}_n"]
        inferior = resumen[f"{metrica}_ic_inferior"]
        superior = resumen[f"{metrica}_ic_superior"]

        print(
            f"{metrica.capitalize()}: {media:.2f}% "
            f"(n={n}, IC 95%: "
            f"{inferior:.2f}% a {superior:.2f}%)"
        )


# =========================================================
# 8. EXPORTAR RESULTADOS
# =========================================================

guardar_csv(
    "resultados_repeticiones.csv",
    filas_repeticiones
)

guardar_csv(
    "resultados_por_hora.csv",
    filas_horas
)

guardar_csv(
    "resumen_estadistico.csv",
    filas_resumen
)


# =========================================================
# 9. GRAFICA 1: COMPARACION DE ESCENARIOS
# =========================================================

nombres = [
    fila["escenario"] for fila in filas_resumen
]

exactitudes = [
    fila["exactitud_media"] for fila in filas_resumen
]

sensibilidades = [
    fila["sensibilidad_media"] for fila in filas_resumen
]

x = np.arange(len(nombres))
ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - ancho / 2,
    exactitudes,
    ancho,
    label="Exactitud"
)

ax.bar(
    x + ancho / 2,
    sensibilidades,
    ancho,
    label="Sensibilidad"
)

ax.set_title(
    "Rendimiento del HMM por escenario"
)
ax.set_ylabel("Porcentaje promedio (%)")
ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(
    ["Habitual", "Más sospechosa", "Ambigua"]
)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    CARPETA / "grafica_comparacion.png",
    dpi=300
)
plt.close(fig)


# =========================================================
# 10. GRAFICA 2: VARIACION DE EXACTITUD
# =========================================================

datos_grafica = [
    [
        fila["exactitud"]
        for fila in filas_repeticiones
        if fila["escenario"] == nombre
    ]
    for nombre in nombres
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    datos_grafica,
    tick_labels=[
        "Habitual",
        "Más sospechosa",
        "Ambigua"
    ]
)

ax.set_title(
    "Distribución de exactitud en 100 repeticiones"
)
ax.set_ylabel("Exactitud (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    CARPETA / "grafica_distribucion.png",
    dpi=300
)
plt.close(fig)

print("\n" + "=" * 60)
print("EXPERIMENTO FINALIZADO CORRECTAMENTE")
print(f"Total de simulaciones: {len(filas_repeticiones)}")
print(f"Total de horas: {len(filas_horas)}")
print(f"Resultados guardados en: {CARPETA}")
