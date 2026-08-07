# -*- coding: utf-8 -*-
"""
generar_datos_ejemplo.py
-------------------------
Genera un archivo CSV sintético (datos_ejemplo.csv) con el mismo formato que
exportaría Google Forms para el cuestionario "¿Qué casa te representa?".
Sirve para probar la aplicación (app.py) sin depender todavía de respuestas
reales recolectadas por el formulario.

Uso:
    python generar_datos_ejemplo.py
"""

import random
from datetime import datetime, timedelta

import pandas as pd

from logica import PREGUNTAS, CASAS

random.seed(42)

OCUPACIONES = ["Estudiante", "Docente", "Empleado", "Freelance", "Desempleado"]
GENEROS = ["Femenino", "Masculino", "Prefiero no decirlo"]
NIVELES_ESTRES = ["Bajo", "Medio", "Alto"]

N_RESPUESTAS = 60


def elegir_respuesta_sesgada(opciones_por_casa: dict, casa_dominante: str) -> str:
    """Elige una opción de la pregunta, con mayor probabilidad de escoger la
    opción de la 'casa dominante' del respondiente simulado (para que existan
    grupos/clusters reconocibles en los datos de prueba)."""
    opciones = list(opciones_por_casa.keys())
    pesos = [
        4 if opciones_por_casa[op] == casa_dominante else 1 for op in opciones
    ]
    return random.choices(opciones, weights=pesos, k=1)[0]


def generar_fila(fecha_base: datetime, idx: int) -> dict:
    casa_dominante = random.choice(CASAS)

    fila = {
        "Marca temporal": (fecha_base + timedelta(minutes=idx * 7)).strftime("%Y-%m-%d %H:%M:%S"),
        "Dirección de correo electrónico": f"persona{idx}@ejemplo.com",
        "Edad": random.randint(18, 45),
        "Ocupación": random.choice(OCUPACIONES),
        "Género": random.choice(GENEROS),
        "Horas de sueño promedio por noche": round(random.uniform(4.5, 9.0), 1),
        "Nivel de estrés autopercibido": random.choice(NIVELES_ESTRES),
    }

    for i, opciones_por_casa in enumerate(PREGUNTAS, start=1):
        fila[f"Pregunta {i}"] = elegir_respuesta_sesgada(opciones_por_casa, casa_dominante)

    return fila


def main():
    fecha_base = datetime(2026, 7, 1, 9, 0, 0)
    filas = [generar_fila(fecha_base, i) for i in range(1, N_RESPUESTAS + 1)]
    df = pd.DataFrame(filas)
    df.to_csv("datos_ejemplo.csv", index=False, encoding="utf-8-sig")
    print(f"Generado datos_ejemplo.csv con {len(df)} respuestas.")


if __name__ == "__main__":
    main()
