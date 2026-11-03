# Simulación de actividad sospechosa en una red informática por medio de un Modelo Oculto de Markov (HMM)

## 1. Descripción del proyecto

Este proyecto implementa un Modelo Oculto de Markov (HMM) para simular actividad que es potencialmente sospechosa en una red informática mediante datos sintéticos.

Se consideran dos estados ocultos: Normal y Sospechoso. Las observaciones corresponden a tres categorías de intentos fallidos de inicio de sesión: Bajo (0 a 5), Medio (6 a 15) y Alto (16 o más).

El algoritmo de Viterbi permite estimar la secuencia de estados ocultos más probable a partir de las observaciones.

## 2. Objetivo

Evaluar cómo las probabilidades de transición y emisión influyen en la exactitud y sensibilidad de un Modelo Oculto de Markov para identificar actividad sospechosa simulada.

## 3. Metodología

Se desarrollaron tres escenarios experimentales:

1. Actividad habitual.
2. Mayor actividad sospechosa.
3. Observaciones ambiguas.

Cada escenario contiene 100 simulaciones independientes de 24 horas, para un total de 300 simulaciones y 7.200 horas simuladas.

Se utiliza la semilla aleatoria 42 para garantizar resultados reproducibles bajo la misma configuración.

## 4. Tecnologías utilizadas

- Python 3.14
- NumPy
- Matplotlib
- SciPy

## 5. Instalación y ejecución

Instalar las bibliotecas:

```bash
py -m pip install -r requirements.txt
```

Ejecución las simulaciones:

```bash
py main.py
```

Realizar las pruebas estadísticas:

```bash
py analisis_estadistico.py
```

Validar la reproducibilidad:

```bash
py verificar_reproducibilidad.py
```

## 6. Resultados

| Escenario | Exactitud promedio | Sensibilidad promedio |
|---|---:|---:|
| Actividad habitual | 90,37 % | 64,26 % |
| Mayor actividad sospechosa | 88,33 % | 91,26 % |
| Observaciones ambiguas | 76,96 % | 0,00 % |

La prueba de Kruskal-Wallis identificó diferencias estadísticamente significativas entre las distribuciones de exactitud de los escenarios (H = 67,3465; p = 2,376309 × 10⁻¹⁵).

Las comparaciones por parejas mostraron diferencias significativas entre el escenario de observaciones ambiguas y los otros dos escenarios.

## 7. Archivos del proyecto

- `main.py`: simulación HMM, Viterbi, métricas y gráficas.
- `analisis_estadistico.py`: pruebas Kruskal-Wallis y Mann-Whitney.
- `verificar_reproducibilidad.py`: verificación mediante SHA-256.
- `requirements.txt`: dependencias del proyecto.
- `resultados/`: archivos CSV y gráficas experimentales.

## 8. Reproducibilidad

Se validó que dos ejecuciones consecutivas con la misma semilla generaron archivos CSV idénticos. La evidencia se encuentra en `resultados/verificacion_reproducibilidad.csv`.

## 9. Limitaciones

Los datos y las probabilidades empleados son sintéticos y fueron definidos con fines académicos. Los resultados no constituyen una validación con tráfico real ni permiten confirmar ataques informáticos.
