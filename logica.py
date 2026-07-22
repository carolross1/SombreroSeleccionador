# -*- coding: utf-8 -*-
"""
logica.py
---------
Lógica de negocio del aplicativo "Test de Personalidad: ¿Qué Casa te representa?"

Contiene:
    - El mapeo de cada una de las 16 preguntas del cuestionario a las 4 casas
      (Gryffindor, Slytherin, Ravenclaw, Hufflepuff), reconstruido a partir del
      contenido de cada opción (no de su posición, a diferencia del script de
      Google Apps Script original, que asumía que el orden de las opciones
      siempre era Gryffindor-Slytherin-Ravenclaw-Hufflepuff).
    - Los mismos pesos por pregunta definidos en el script original.
    - Funciones de estadística descriptiva implementadas "a mano" (sin depender
      únicamente de pandas.describe()), tal como pide el SABER HACER.
    - Funciones de agrupación (K-Means) y reducción de dimensionalidad (PCA).

Este módulo es independiente de Streamlit para que pueda probarse por separado.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CASAS = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

EMOJIS = {
    "Gryffindor": "🦁",
    "Slytherin": "🐍",
    "Ravenclaw": "🦅",
    "Hufflepuff": "🦡",
}

COLORES = {
    "Gryffindor": "#740001",
    "Slytherin": "#1A472A",
    "Ravenclaw": "#0E1A40",
    "Hufflepuff": "#916300",
}

COLORES_SECUNDARIOS = {
    "Gryffindor": "#D3A625",
    "Slytherin": "#AAAAAA",
    "Ravenclaw": "#946B2D",
    "Hufflepuff": "#372E29",
}

ANIMAL = {
    "Gryffindor": "León",
    "Slytherin": "Serpiente",
    "Ravenclaw": "Águila",
    "Hufflepuff": "Tejón",
}

# Rasgo y descripción breve, texto original (no son citas de libros/películas),
# pensados solo para dar contexto visual al resultado del clustering.
RASGO = {
    "Gryffindor": "Valentía",
    "Slytherin": "Ambición",
    "Ravenclaw": "Sabiduría",
    "Hufflepuff": "Lealtad",
}

DESCRIPCION = {
    "Gryffindor": "Actúa primero y piensa después; no rehúye el riesgo cuando hay algo importante en juego.",
    "Slytherin": "Piensa en estrategia y resultados; sabe reconocer y aprovechar una oportunidad.",
    "Ravenclaw": "Prefiere entender antes que actuar; disfruta analizar un problema desde todos los ángulos.",
    "Hufflepuff": "Prioriza al grupo por encima de sí mismo; es constante y de confianza.",
}

# Lemas breves, texto original escrito para este proyecto (no son citas de
# los libros ni de las películas), pensados solo como cierre visual de cada
# tarjeta de casa.
LEMAS = {
    "Gryffindor": "El valor no es la ausencia de miedo, sino actuar a pesar de él.",
    "Slytherin": "Toda gran meta empieza con una jugada inteligente.",
    "Ravenclaw": "La mente que pregunta es la que llega más lejos.",
    "Hufflepuff": "Ningún logro vale la pena si se deja a alguien atrás.",
}

# Pesos por pregunta, en el orden en que aparecen las 16 preguntas
# (idénticos a los del script de Google Apps Script proporcionado).
PESOS: List[int] = [2, 2, 3, 1, 1, 2, 1, 2, 3, 3, 2, 3, 2, 1, 3, 1]
MAX_POSIBLE = sum(PESOS)  # 30

# Mapeo pregunta -> {texto de opción: casa}. Reconstruido leyendo el
# contenido/intención de cada opción del cuestionario (valentía -> Gryffindor,
# ambición/conveniencia propia -> Slytherin, sabiduría/análisis -> Ravenclaw,
# lealtad/cuidado del grupo -> Hufflepuff).
PREGUNTAS: List[Dict[str, str]] = [
    {
        "Me aseguro de que la persona afectada esté bien, ante todo": "Hufflepuff",
        "Confronto la situación de inmediato, sin pensarlo mucho": "Gryffindor",
        "Evalúo primero si me conviene involucrarme": "Slytherin",
        "Busco la manera más inteligente y estratégica de resolverlo": "Ravenclaw",
    },
    {
        "Planear la estrategia más eficiente para destacar": "Slytherin",
        "Investigar a fondo antes de proponer una solución": "Ravenclaw",
        "Asegurarme de que todos colaboren y se sientan incluidos": "Hufflepuff",
        "Tomar el liderazgo y avanzar rápido, aunque haya riesgos": "Gryffindor",
    },
    {
        "Que sea valiente y esté dispuesto a defenderme": "Gryffindor",
        "Que sea inteligente y tenga conversaciones interesantes": "Ravenclaw",
        "Que sea astuto y me ayude a conseguir lo que quiero": "Slytherin",
        "Que sea leal y esté ahí siempre, pase lo que pase": "Hufflepuff",
    },
    {
        "Buscar la forma más rápida de sacar ventaja de la situación": "Slytherin",
        "Analizar todas las opciones antes de decidir": "Ravenclaw",
        "Actuar de inmediato, aunque no tenga todo resuelto": "Gryffindor",
        "Buscar ayuda y resolverlo en conjunto con otros": "Hufflepuff",
    },
    {
        "Me interesa más entender cómo funciona el juego que ganar": "Ravenclaw",
        "Quiero ganar por mi propio esfuerzo y valentía": "Gryffindor",
        "Disfruto competir, pero me importa más que todos se diviertan": "Hufflepuff",
        "Quiero ganar, y usaré cualquier ventaja disponible": "Slytherin",
    },
    {
        "Analizo qué salió mal para no repetir el error y ganar la próxima vez": "Slytherin",
        "Me levanto rápido e intento de nuevo con más fuerza": "Gryffindor",
        "Busco apoyo en otros y no me rindo aunque tarde más": "Hufflepuff",
        "Reflexiono profundamente sobre las causas del fracaso": "Ravenclaw",
    },
    {
        "Una actividad grupal donde todos participen": "Hufflepuff",
        "Un juego de estrategia donde pueda ganar": "Slytherin",
        "Un deporte de alto riesgo o adrenalina": "Gryffindor",
        "Resolver acertijos o aprender algo nuevo": "Ravenclaw",
    },
    {
        "Considerar cómo afecta mi decisión a los demás": "Hufflepuff",
        "Detenerme a analizar la lógica de la situación, aunque tarde más": "Ravenclaw",
        "Confiar en mi instinto y actuar sin dudar": "Gryffindor",
        "Pensar qué me conviene más a largo plazo": "Slytherin",
    },
    {
        "Uno que cuida del bienestar de todo su equipo": "Hufflepuff",
        "Uno ambicioso que sabe cómo conseguir resultados": "Slytherin",
        "Uno que toma decisiones basadas en datos y lógica": "Ravenclaw",
        "Uno que inspira con el ejemplo y va al frente": "Gryffindor",
    },
    {
        "Que sea justa para todos los involucrados": "Hufflepuff",
        "Que tenga sentido lógico y esté bien fundamentada": "Ravenclaw",
        "Hacer lo correcto, aunque sea arriesgado": "Gryffindor",
        "Que me beneficie a mí y a mis metas": "Slytherin",
    },
    {
        "El que se asegura de que nadie quede atrás": "Hufflepuff",
        "El que actúa primero y pide perdón después": "Gryffindor",
        "El que mantiene la calma y piensa con claridad": "Ravenclaw",
        "El que ve la oportunidad detrás del caos": "Slytherin",
    },
    {
        "Mi ambición y capacidad de lograr lo que me propongo": "Slytherin",
        "Mi inteligencia y curiosidad": "Ravenclaw",
        "Mi valentía": "Gryffindor",
        "Mi lealtad y buen corazón": "Hufflepuff",
    },
    {
        "Analizando el problema desde una perspectiva objetiva": "Ravenclaw",
        "Buscando un acuerdo donde todos queden bien": "Hufflepuff",
        "Buscando la manera de salir ganando yo": "Slytherin",
        "Enfrentándolo directamente, sin rodeos": "Gryffindor",
    },
    {
        "El reconocimiento y el éxito": "Slytherin",
        "El desafío en sí mismo": "Gryffindor",
        "El aprender algo nuevo en el proceso": "Ravenclaw",
        "Ayudar o apoyar a alguien más": "Hufflepuff",
    },
    {
        "Capacidad de generar confianza en los demás": "Hufflepuff",
        "Coraje inquebrantable": "Gryffindor",
        "Habilidad para conseguir lo que quiero": "Slytherin",
        "Inteligencia excepcional": "Ravenclaw",
    },
    {
        "Me lanzo sin pensarlo mucho, con energía": "Gryffindor",
        "Investigo bien antes de empezar": "Ravenclaw",
        "Ya tengo un plan de cómo sacarle el mayor provecho": "Slytherin",
        "Pienso en cómo involucrar y apoyar a los demás": "Hufflepuff",
    },
]

N_PREGUNTAS = len(PREGUNTAS)  # 16

# Palabras clave para detectar automáticamente las columnas de metadatos
# dentro de un export de Google Forms (los nombres exactos de columna pueden
# variar ligeramente, por eso se busca por coincidencia parcial).
METADATA_KEYWORDS = {
    "marca_temporal": ["marca temporal", "timestamp"],
    "correo": ["correo"],
    "edad": ["edad"],
    "ocupacion": ["ocupaci"],
    "genero": ["gener"],
    "sueno": ["sueño", "sueno"],
    "estres": ["estrés", "estres"],
}


def _normalizar(texto: str) -> str:
    """Quita acentos, espacios extra y pasa a minúsculas para comparar texto
    de forma robusta ante pequeñas diferencias de captura."""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return texto


@dataclass
class ColumnasDetectadas:
    metadatos: Dict[str, str]
    preguntas: List[str]


def detectar_columnas(df: pd.DataFrame) -> ColumnasDetectadas:
    """Detecta cuáles columnas del CSV son metadatos (correo, edad, género,
    etc.) y cuáles corresponden a las 16 preguntas del cuestionario, asumiendo
    que las columnas de preguntas conservan el orden original del formulario.
    """
    metadatos: Dict[str, str] = {}
    columnas_restantes: List[str] = []

    for col in df.columns:
        col_norm = _normalizar(col)
        asignada = False
        for clave, palabras in METADATA_KEYWORDS.items():
            if clave in metadatos:
                continue
            if any(p in col_norm for p in palabras):
                metadatos[clave] = col
                asignada = True
                break
        if not asignada:
            columnas_restantes.append(col)

    return ColumnasDetectadas(metadatos=metadatos, preguntas=columnas_restantes)


def calcular_puntajes(df: pd.DataFrame, columnas_preguntas: List[str]) -> pd.DataFrame:
    """Calcula, para cada respondiente (fila), el puntaje absoluto y el
    porcentaje obtenido en cada una de las 4 casas, replicando la lógica de
    ponderación del script de Google Apps Script pero basada en el contenido
    de la respuesta (no en su posición).
    """
    n = min(len(columnas_preguntas), N_PREGUNTAS)
    filas = []
    for _, fila in df.iterrows():
        puntos = {c: 0 for c in CASAS}
        for i in range(n):
            col = columnas_preguntas[i]
            respuesta = _normalizar(fila.get(col, ""))
            mapeo = {_normalizar(k): v for k, v in PREGUNTAS[i].items()}
            casa = mapeo.get(respuesta)
            if casa:
                puntos[casa] += PESOS[i]
        filas.append(puntos)

    puntajes = pd.DataFrame(filas, index=df.index)
    for c in CASAS:
        puntajes[f"{c}_pct"] = round(puntajes[c] / MAX_POSIBLE * 100, 1)
    puntajes["Casa_dominante"] = puntajes[CASAS].idxmax(axis=1)
    return puntajes


# ---------------------------------------------------------------------------
# Estadística descriptiva "propia" (sin usar df.describe())
# ---------------------------------------------------------------------------

def media_propia(valores: List[float]) -> float:
    valores = [v for v in valores if v is not None and not pd.isna(v)]
    if not valores:
        return float("nan")
    return sum(valores) / len(valores)


def desviacion_estandar_propia(valores: List[float]) -> float:
    valores = [v for v in valores if v is not None and not pd.isna(v)]
    n = len(valores)
    if n < 2:
        return float("nan")
    m = media_propia(valores)
    varianza = sum((v - m) ** 2 for v in valores) / (n - 1)
    return varianza ** 0.5


def moda_propia(valores: List) -> object:
    valores = [v for v in valores if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not valores:
        return None
    conteo: Dict[object, int] = {}
    for v in valores:
        conteo[v] = conteo.get(v, 0) + 1
    return max(conteo.items(), key=lambda kv: kv[1])[0]


def tabla_frecuencias(valores: List) -> pd.Series:
    conteo: Dict[object, int] = {}
    for v in valores:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        conteo[v] = conteo.get(v, 0) + 1
    return pd.Series(conteo).sort_values(ascending=False)


def resumen_estadistico(df: pd.DataFrame, columnas_numericas: List[str]) -> pd.DataFrame:
    filas = []
    for col in columnas_numericas:
        valores = pd.to_numeric(df[col], errors="coerce").tolist()
        valores_validos = [v for v in valores if not pd.isna(v)]
        filas.append(
            {
                "Variable": col,
                "n": len(valores_validos),
                "Media": round(media_propia(valores_validos), 2) if valores_validos else float("nan"),
                "Desv. estándar": round(desviacion_estandar_propia(valores_validos), 2) if valores_validos else float("nan"),
                "Mínimo": min(valores_validos) if valores_validos else float("nan"),
                "Máximo": max(valores_validos) if valores_validos else float("nan"),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Agrupación (K-Means) y reducción de dimensionalidad (PCA)
# ---------------------------------------------------------------------------

def preparar_features(puntajes: pd.DataFrame) -> np.ndarray:
    columnas = [f"{c}_pct" for c in CASAS]
    return puntajes[columnas].to_numpy(dtype=float)


def calcular_inercias(X_esc: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """Inercia (método del codo) para distintos valores de k, útil para
    justificar la elección del número de clusters en el reporte."""
    filas = []
    k_max = min(k_max, max(k_min, X_esc.shape[0] - 1))
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X_esc)
        filas.append({"k": k, "Inercia": modelo.inertia_})
    return pd.DataFrame(filas)


def entrenar_kmeans(X: np.ndarray, k: int) -> Tuple[StandardScaler, KMeans, PCA, np.ndarray, np.ndarray]:
    """Entrena el escalador, el modelo K-Means y un PCA de 2 componentes para
    visualización/reducción de dimensionalidad. Devuelve también las
    etiquetas de cluster y las coordenadas PCA."""
    escalador = StandardScaler()
    X_esc = escalador.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(X_esc)

    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(X_esc)

    return escalador, kmeans, pca, etiquetas, coords_pca
