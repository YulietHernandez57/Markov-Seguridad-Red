
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

# ============================================
# VERIFICACION DE REPRODUCIBILIDAD DEL HMM
# ============================================

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"
PROGRAMA = CARPETA_PROYECTO / "main.py"

ARCHIVOS = [
    "resultados_repeticiones.csv",
    "resultados_por_hora.csv",
    "resumen_estadistico.csv"
]


# Calcular una huella digital de cada archivo
def calcular_hash(ruta):
    sha256 = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(65536), b""
        ):
            sha256.update(bloque)

    return sha256.hexdigest()


# Ejecutar el simulador
def ejecutar_simulacion():
    subprocess.run(
        [sys.executable, str(PROGRAMA)],
        cwd=CARPETA_PROYECTO,
        check=True,
        capture_output=True,
        text=True
    )


# Obtener las huellas de los resultados
def obtener_huellas():
    huellas = {}

    for nombre in ARCHIVOS:
        ruta = CARPETA_RESULTADOS / nombre

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro: {nombre}"
            )

        huellas[nombre] = calcular_hash(ruta)

    return huellas


print("\nVERIFICACION DE REPRODUCIBILIDAD")
print("=" * 55)

print("Ejecutando simulacion 1...")
ejecutar_simulacion()
primera = obtener_huellas()

print("Ejecutando simulacion 2...")
ejecutar_simulacion()
segunda = obtener_huellas()

print("\nCOMPARACION DE ARCHIVOS")
print("-" * 55)

resultados = []

for nombre in ARCHIVOS:
    iguales = primera[nombre] == segunda[nombre]

    estado = "IDENTICOS" if iguales else "DIFERENTES"

    print(f"{nombre}: {estado}")

    resultados.append({
        "archivo": nombre,
        "hash_primera_ejecucion": primera[nombre],
        "hash_segunda_ejecucion": segunda[nombre],
        "reproducible": iguales
    })

# Guardar evidencia
ruta_reporte = (
    CARPETA_RESULTADOS
    / "verificacion_reproducibilidad.csv"
)

with open(
    ruta_reporte,
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

print("\n" + "=" * 55)

if all(fila["reproducible"] for fila in resultados):
    print("RESULTADO: REPRODUCIBILIDAD VERIFICADA")
    print("Los archivos CSV son identicos.")
else:
    print("RESULTADO: REPRODUCIBILIDAD NO VERIFICADA")

print(f"\nReporte guardado: {ruta_reporte.name}")
