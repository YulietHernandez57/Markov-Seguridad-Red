"""Análisis de sensibilidad de la distribución inicial pi de un HMM sintético.

Mantiene A y B del escenario 'Actividad habitual' de main.py.
Ejecutar: py analisis_sensibilidad.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ESTADOS = ('Normal', 'Sospechoso')
OBSERVACIONES = ('Bajo', 'Medio', 'Alto')
A = np.array([[0.90, 0.10], [0.30, 0.70]], dtype=float)
B = np.array([[0.80, 0.15, 0.05], [0.10, 0.30, 0.60]], dtype=float)
DISTRIBUCIONES = {
    '90/10': np.array([0.90, 0.10]),
    '70/30': np.array([0.70, 0.30]),
    '50/50': np.array([0.50, 0.50]),
}
REPETICIONES = 100
HORAS = 24
SEMILLA = 42
SALIDA = Path(__file__).resolve().parent / 'resultados' / 'sensibilidad'
SALIDA.mkdir(parents=True, exist_ok=True)


def simular(pi, rng):
    reales = np.empty(HORAS, dtype=int)
    observaciones = np.empty(HORAS, dtype=int)
    estado = int(rng.choice(2, p=pi))
    for t in range(HORAS):
        reales[t] = estado
        observaciones[t] = rng.choice(3, p=B[estado])
        estado = rng.choice(2, p=A[estado])
    return reales, observaciones


def viterbi(observaciones, pi):
    n = len(observaciones)
    puntajes = np.empty((n, 2))
    anteriores = np.zeros((n, 2), dtype=int)
    puntajes[0] = np.log(pi) + np.log(B[:, observaciones[0]])
    for t in range(1, n):
        for j in range(2):
            candidatos = puntajes[t - 1] + np.log(A[:, j])
            anteriores[t, j] = int(np.argmax(candidatos))
            puntajes[t, j] = np.max(candidatos) + np.log(B[j, observaciones[t]])
    estimados = np.empty(n, dtype=int)
    estimados[-1] = int(np.argmax(puntajes[-1]))
    for t in range(n - 2, -1, -1):
        estimados[t] = anteriores[t + 1, estimados[t + 1]]
    return estimados


def metricas(reales, estimados):
    vp = np.sum((reales == 1) & (estimados == 1))
    vn = np.sum((reales == 0) & (estimados == 0))
    fp = np.sum((reales == 0) & (estimados == 1))
    fn = np.sum((reales == 1) & (estimados == 0))
    return {
        'exactitud': 100 * (vp + vn) / len(reales),
        'sensibilidad': 100 * vp / (vp + fn) if vp + fn else float('nan'),
        'precision': 100 * vp / (vp + fp) if vp + fp else float('nan'),
        'especificidad': 100 * vn / (vn + fp) if vn + fp else float('nan'),
        'proporcion_sospechosa_real': 100 * np.mean(reales == 1),
    }


def resumen(valores):
    x = np.asarray(valores, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, media=float('nan'), desviacion=float('nan'), minimo=float('nan'),
                    maximo=float('nan'), ic_inferior=float('nan'), ic_superior=float('nan'))
    media = float(np.mean(x))
    desviacion = float(np.std(x, ddof=1)) if n > 1 else float('nan')
    margen = 1.96 * desviacion / np.sqrt(n) if n > 1 else float('nan')
    return dict(n=n, media=media, desviacion=desviacion, minimo=float(np.min(x)),
                maximo=float(np.max(x)), ic_inferior=media - margen, ic_superior=media + margen)


def guardar_csv(nombre, filas):
    with (SALIDA / nombre).open('w', newline='', encoding='utf-8-sig') as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def main():
    for matriz in (A, B):
        assert np.all(matriz >= 0) and np.allclose(matriz.sum(axis=1), 1)
    filas = []
    filas_resumen = []
    # Una semilla por repetición compartida entre condiciones: diseño pareado.
    semillas = np.random.SeedSequence(SEMILLA).spawn(REPETICIONES)
    print('\nANALISIS DE SENSIBILIDAD DE PI - HMM')
    print('=' * 62)
    print(f'{REPETICIONES} repeticiones de {HORAS} horas por distribucion')
    for nombre, pi in DISTRIBUCIONES.items():
        assert np.all(pi >= 0) and np.isclose(pi.sum(), 1)
        datos_condicion = []
        for i, semilla in enumerate(semillas, start=1):
            # Misma semilla por repetición y condición, para reducir variabilidad aleatoria.
            rng = np.random.default_rng(semilla)
            reales, observaciones = simular(pi, rng)
            estimados = viterbi(observaciones, pi)
            fila = {'pi': nombre, 'repeticion': i, **metricas(reales, estimados)}
            filas.append(fila)
            datos_condicion.append(fila)
        print(f'\nDistribucion inicial pi = {nombre}')
        for metrica in ('exactitud', 'sensibilidad', 'precision', 'especificidad',
                        'proporcion_sospechosa_real'):
            r = resumen([fila[metrica] for fila in datos_condicion])
            filas_resumen.append({'pi': nombre, 'metrica': metrica, **r})
            print(f'  {metrica}: {r["media"]:.2f}% | n={r["n"]} | '
                  f'DE={r["desviacion"]:.2f} | rango={r["minimo"]:.2f}-{r["maximo"]:.2f} | '
                  f'IC95%={r["ic_inferior"]:.2f}-{r["ic_superior"]:.2f}')
    guardar_csv('sensibilidad_repeticiones.csv', filas)
    guardar_csv('sensibilidad_resumen.csv', filas_resumen)
    etiquetas = list(DISTRIBUCIONES)
    x = np.arange(len(etiquetas))
    ancho = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    for desplazamiento, metrica, etiqueta in [(-ancho/2, 'exactitud', 'Exactitud'),
                                              (ancho/2, 'sensibilidad', 'Sensibilidad')]:
        valores = [next(r['media'] for r in filas_resumen
                        if r['pi'] == nombre and r['metrica'] == metrica) for nombre in etiquetas]
        ax.bar(x + desplazamiento, valores, ancho, label=etiqueta)
    ax.set_xticks(x, etiquetas)
    ax.set_xlabel('Distribucion inicial pi (Normal/Sospechoso)')
    ax.set_ylabel('Promedio (%)')
    ax.set_title('Sensibilidad del HMM a cambios en la distribucion inicial')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / 'grafica_sensibilidad_pi.png', dpi=300)
    plt.close(fig)
    print('\nArchivos generados en:', SALIDA)
    print('ANALISIS FINALIZADO CORRECTAMENTE')


if __name__ == '__main__':
    main()
