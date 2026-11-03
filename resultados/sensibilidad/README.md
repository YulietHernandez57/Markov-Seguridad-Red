# Análisis de sensibilidad del HMM

Esta carpeta contiene los resultados del experimento de sensibilidad a cambios en la distribución inicial de estados (π).

Se evaluaron tres configuraciones: 90/10, 70/30 y 50/50, con 100 repeticiones de 24 horas cada una.

Archivos:
- sensibilidad_repeticiones.csv: resultados individuales de las simulaciones.
- sensibilidad_resumen.csv: estadísticas descriptivas e intervalos de confianza.
- pruebas_sensibilidad.csv: pruebas de Friedman y Wilcoxon con corrección de Bonferroni.
- grafica_sensibilidad_pi.png: comparación visual de exactitud y sensibilidad.

Para reproducir el experimento, ejecutar desde la raíz del proyecto:

python analisis_sensibilidad.py
python pruebas_sensibilidad.py

Los datos corresponden a simulaciones sintéticas, no a registros reales de una red informática.
