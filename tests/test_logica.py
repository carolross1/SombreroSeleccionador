# -*- coding: utf-8 -*-
"""
tests/test_logica.py
---------------------
Pruebas unitarias para logica.py. Se enfocan en las funciones "hechas a
mano" (estadistica propia, calculo de puntajes, deteccion de columnas) y en
las validaciones agregadas para robustez, ya que son las que mas facil se
rompen con un refactor silencioso.

Ejecutar con:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logica


# ---------------------------------------------------------------------------
# Estadistica propia
# ---------------------------------------------------------------------------

def test_media_propia():
    assert logica.media_propia([1, 2, 3, 4]) == 2.5


def test_media_propia_ignora_nan():
    assert logica.media_propia([1, 2, float("nan"), 3]) == 2.0


def test_media_propia_lista_vacia():
    assert np.isnan(logica.media_propia([]))


def test_desviacion_estandar_propia():
    # Desviacion estandar muestral (n-1) de [2, 4, 4, 4, 5, 5, 7, 9] es 2.138...
    valores = [2, 4, 4, 4, 5, 5, 7, 9]
    assert logica.desviacion_estandar_propia(valores) == pytest.approx(2.1381, rel=1e-3)


def test_desviacion_estandar_un_solo_valor():
    assert np.isnan(logica.desviacion_estandar_propia([5]))


def test_moda_propia_unimodal():
    assert logica.moda_propia([1, 1, 2, 3]) == [1]


def test_moda_propia_bimodal():
    resultado = logica.moda_propia([1, 1, 2, 2, 3])
    assert set(resultado) == {1, 2}


def test_moda_propia_vacia():
    assert logica.moda_propia([]) == []


# ---------------------------------------------------------------------------
# Deteccion de columnas y calculo de puntajes
# ---------------------------------------------------------------------------

def _df_ejemplo():
    return pd.read_csv(Path(__file__).resolve().parent.parent / "datos_ejemplo.csv")


def test_detectar_columnas_encuentra_las_16_preguntas():
    df = _df_ejemplo()
    columnas = logica.detectar_columnas(df)
    assert len(columnas.preguntas) == logica.N_PREGUNTAS


def test_calcular_puntajes_suma_100_por_fila():
    df = _df_ejemplo()
    columnas = logica.detectar_columnas(df)
    puntajes = logica.calcular_puntajes(df, columnas.preguntas)
    # Cada fila reparte el 100% del puntaje maximo entre las 4 casas
    # (composicional: no aportan informacion 4 dimensiones independientes).
    suma_pct = puntajes[[f"{c}_pct" for c in logica.CASAS]].sum(axis=1)
    assert (suma_pct.round(0) == 100).all()


def test_calcular_puntajes_respeta_orden_de_columnas():
    df = pd.DataFrame(
        {
            "Pregunta 1": ["Confronto la situación de inmediato, sin pensarlo mucho"],
        }
    )
    puntajes = logica.calcular_puntajes(df, ["Pregunta 1"])
    # Peso de la pregunta 1 es 2, y la opcion elegida es Gryffindor
    assert puntajes.loc[0, "Gryffindor"] == 2
    assert puntajes.loc[0, "Casa_dominante"] == "Gryffindor"


# ---------------------------------------------------------------------------
# Validaciones de robustez
# ---------------------------------------------------------------------------

def test_validar_columnas_preguntas_ok():
    ok, msg = logica.validar_columnas_preguntas([f"Pregunta {i}" for i in range(1, 17)])
    assert ok and msg == ""


def test_validar_columnas_preguntas_insuficientes():
    ok, msg = logica.validar_columnas_preguntas(["Pregunta 1", "Pregunta 2"])
    assert not ok
    assert "16" in msg


def test_validar_columnas_preguntas_vacia():
    ok, msg = logica.validar_columnas_preguntas([])
    assert not ok


def test_contar_respuestas_faltantes():
    df = pd.DataFrame(
        {
            "Pregunta 1": ["Confronto la situación de inmediato, sin pensarlo mucho", ""],
        }
    )
    faltantes = logica.contar_respuestas_faltantes(df, ["Pregunta 1"])
    assert faltantes.tolist() == [0, 1]


def test_validar_modelo_features_compatible():
    X = np.random.rand(10, 4)
    modelo = {"features": ["a", "b", "c", "d"], "escalador": None}
    ok, msg = logica.validar_modelo_features(modelo, X)
    assert ok and msg == ""


def test_validar_modelo_features_incompatible():
    X = np.random.rand(10, 4)
    modelo = {"features": ["a"], "escalador": None}
    ok, msg = logica.validar_modelo_features(modelo, X)
    assert not ok


# ---------------------------------------------------------------------------
# Consistencia del modelo guardado (aplicar_modelo_guardado == entrenar_kmeans)
# ---------------------------------------------------------------------------

def test_aplicar_modelo_guardado_es_consistente_con_el_entrenamiento():
    df = _df_ejemplo()
    columnas = logica.detectar_columnas(df)
    puntajes = logica.calcular_puntajes(df, columnas.preguntas)
    X = logica.preparar_features(puntajes)

    escalador, kmeans, pca, etiquetas, coords = logica.entrenar_kmeans(X, k=4)
    modelo = {
        "escalador": escalador,
        "kmeans": kmeans,
        "pca": pca,
        "k": 4,
        "features": [f"{c}_pct" for c in logica.CASAS],
    }

    etiquetas2, coords2 = logica.aplicar_modelo_guardado(modelo, X)

    assert (etiquetas == etiquetas2).all()
    assert np.allclose(coords, coords2)


# ---------------------------------------------------------------------------
# Cuestionario configurable (soporte para otros cuestionarios, no solo Hogwarts)
# ---------------------------------------------------------------------------

CUESTIONARIO_JSON_EJEMPLO = """
{
  "nombre": "Estilos de trabajo",
  "categorias": ["Analitico", "Creativo", "Ejecutor"],
  "preguntas": [
    {
      "enunciado": "¿Como resuelves un problema nuevo?",
      "peso": 2,
      "opciones": {
        "Reviso datos y evidencia primero": "Analitico",
        "Pruebo varias ideas distintas": "Creativo",
        "Empiezo a actuar de inmediato": "Ejecutor"
      }
    },
    {
      "enunciado": "¿Que valoras mas en un equipo?",
      "peso": 3,
      "opciones": {
        "Precision": "Analitico",
        "Originalidad": "Creativo",
        "Velocidad": "Ejecutor"
      }
    }
  ]
}
"""


def test_cargar_cuestionario_json_valido():
    cuestionario = logica.cargar_cuestionario_json(CUESTIONARIO_JSON_EJEMPLO)
    assert cuestionario.categorias == ["Analitico", "Creativo", "Ejecutor"]
    assert cuestionario.n_preguntas == 2
    assert cuestionario.pesos == [2, 3]
    assert cuestionario.max_posible == 5


def test_cargar_cuestionario_json_invalido_da_mensaje_claro():
    with pytest.raises(ValueError):
        logica.cargar_cuestionario_json("{esto no es json valido")


def test_cargar_cuestionario_json_categoria_no_declarada():
    malo = '{"categorias": ["X", "Y"], "preguntas": [{"opciones": {"a": "X", "b": "Z"}}]}'
    with pytest.raises(ValueError, match="no declaradas"):
        logica.cargar_cuestionario_json(malo)


def test_calcular_puntajes_con_cuestionario_personalizado():
    cuestionario = logica.cargar_cuestionario_json(CUESTIONARIO_JSON_EJEMPLO)
    df = pd.DataFrame(
        {
            "Pregunta 1": ["Reviso datos y evidencia primero"],
            "Pregunta 2": ["Precision"],
        }
    )
    puntajes = logica.calcular_puntajes(df, ["Pregunta 1", "Pregunta 2"], cuestionario)
    assert puntajes.loc[0, "Casa_dominante"] == "Analitico"
    assert puntajes.loc[0, "Analitico_pct"] == 100.0


def test_pipeline_completo_con_cuestionario_personalizado():
    """El motor de K-Means/PCA debe funcionar igual de bien con un
    cuestionario distinto al de Hogwarts (distinto numero de categorias)."""
    cuestionario = logica.cargar_cuestionario_json(CUESTIONARIO_JSON_EJEMPLO)
    df = pd.DataFrame(
        {
            "Pregunta 1": [
                "Reviso datos y evidencia primero",
                "Pruebo varias ideas distintas",
                "Empiezo a actuar de inmediato",
            ],
            "Pregunta 2": ["Precision", "Originalidad", "Velocidad"],
        }
    )
    ok, _ = logica.validar_columnas_preguntas(["Pregunta 1", "Pregunta 2"], cuestionario)
    assert ok

    puntajes = logica.calcular_puntajes(df, ["Pregunta 1", "Pregunta 2"], cuestionario)
    X = logica.preparar_features(puntajes, cuestionario.categorias)
    assert X.shape == (3, 3)

    escalador, kmeans, pca, etiquetas, coords = logica.entrenar_kmeans(X, k=3)
    assert len(etiquetas) == 3


def test_plantilla_cuestionario_json_es_recargable():
    """La plantilla exportada del cuestionario de Hogwarts debe poder
    volver a cargarse sin errores (round-trip)."""
    plantilla = logica.plantilla_cuestionario_json(logica.CUESTIONARIO_HOGWARTS)
    recargado = logica.cargar_cuestionario_json(plantilla)
    assert recargado.categorias == logica.CUESTIONARIO_HOGWARTS.categorias
    assert recargado.n_preguntas == logica.CUESTIONARIO_HOGWARTS.n_preguntas
