# -*- coding: utf-8 -*-
"""
app.py
------
Aplicativo (Streamlit) para la Actividad 1 - Unidad IV: Análisis No Supervisado.
Materia: Extracción de Conocimientos en Base de Datos.
Temática visual: "El Sombrero Seleccionador" (Hogwarts).

Funcionalidades cubiertas (según lista de cotejo):
    1. Carga de datos (CSV exportado del Google Form).
    2. Muestra la información que se carga.
    3. Filtro con base a categorías de la información.
    4. Generación de información estadística básica (algoritmos propios).
    5. Inicio de aprendizaje del algoritmo, genera el modelo de entrenamiento y lo guarda.
    6. Generación de resultados después del proceso de aprendizaje.
    7. Descarga de datos previamente filtrados y de los resultados generados.
    8. Aplica el algoritmo no supervisado deseado: K-Means (agrupación)
       + PCA (reducción de dimensionalidad).

La información se organiza en pestañas (Datos, Estadísticas, Entrenamiento,
Resultados, Selección individual, Descargas) y los filtros viven en la barra
lateral, siempre visibles sin importar la pestaña activa.

Ejecutar con:
    streamlit run app.py
"""

import io
import time
import base64
import uuid

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import estilo
import logica

st.set_page_config(page_title="El Sombrero Seleccionador — Análisis No Supervisado", page_icon="", layout="wide")
st.markdown(estilo.CSS, unsafe_allow_html=True)

# Encabezado principal y reproducción automática del video local Sombrero.mp4
st.markdown("<h1 style='text-align: center; font-family: Cinzel Decorative, serif; color: #c9a227;'>El Sombrero Seleccionador</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-family: Cinzel, serif; color: #cbb994; font-style: italic;'>Analisis No Supervisado · Extraccion de Conocimientos en Base de Datos · Unidad IV</p>", unsafe_allow_html=True)

col_v1, col_v_video, col_v3 = st.columns([1, 1.2, 1])
with col_v_video:
    try:
        with open("Sombrero.mp4", "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes, format="video/mp4", autoplay=True, muted=True, loop=True)
    except Exception:
        st.warning("No se pudo cargar el video Sombrero.mp4 en la carpeta del proyecto.")

# ---------------------------------------------------------------------------
# EDITOR VISUAL DE CUESTIONARIO — helpers
# ---------------------------------------------------------------------------
def _cuestionario_a_editor(cuestionario: "logica.Cuestionario") -> dict:
    """Convierte un Cuestionario a la estructura que usa el editor visual
    (con identificadores estables por pregunta/opcion, para que los widgets
    de Streamlit no se confundan al agregar o quitar filas)."""
    datos = logica.cuestionario_a_dict(cuestionario)
    return {
        "nombre": datos["nombre"],
        "categorias": list(datos["categorias"]),
        "preguntas": [
            {
                "id": uuid.uuid4().hex,
                "enunciado": p["enunciado"],
                "peso": p["peso"],
                "opciones": [
                    {"id": uuid.uuid4().hex, "texto": texto, "categoria": cat}
                    for texto, cat in p["opciones"].items()
                ],
            }
            for p in datos["preguntas"]
        ],
    }


def _editor_a_cuestionario(editor: dict) -> "logica.Cuestionario":
    """Convierte la estructura del editor visual de vuelta a un Cuestionario,
    validando el resultado (lanza ValueError con un mensaje claro si algo
    falta, por ejemplo una pregunta sin opciones)."""
    contenido = {
        "nombre": editor["nombre"],
        "categorias": editor["categorias"],
        "preguntas": [
            {
                "enunciado": p["enunciado"],
                "peso": p["peso"],
                "opciones": {op["texto"]: op["categoria"] for op in p["opciones"] if op["texto"].strip()},
            }
            for p in editor["preguntas"]
        ],
    }
    return logica.dict_a_cuestionario(contenido)


def _pregunta_nueva(categorias: list) -> dict:
    cat_defecto = categorias[0] if categorias else ""
    return {
        "id": uuid.uuid4().hex,
        "enunciado": "Escribe aqui el texto de la pregunta",
        "peso": 1,
        "opciones": [
            {"id": uuid.uuid4().hex, "texto": "", "categoria": cat_defecto},
            {"id": uuid.uuid4().hex, "texto": "", "categoria": cat_defecto},
        ],
    }


def _render_editor_cuestionario() -> None:
    """Dibuja el editor visual completo del cuestionario en el area
    principal: nombre, categorias y preguntas con sus opciones, todo con
    botones (sin necesidad de escribir ni entender JSON)."""
    if "editor_cuestionario" not in st.session_state:
        base = st.session_state.get("cuestionario_guardado", logica.CUESTIONARIO_HOGWARTS)
        st.session_state.editor_cuestionario = _cuestionario_a_editor(base)

    editor = st.session_state.editor_cuestionario

    st.markdown("## Editor de cuestionario")
    st.caption(
        "Aqui puedes cambiar el nombre, las categorias (los grupos de personalidad "
        "en los que se puede clasificar a alguien) y las preguntas con sus opciones "
        "de respuesta. No hace falta tocar ningun archivo."
    )

    editor["nombre"] = st.text_input("Nombre del cuestionario", value=editor["nombre"])

    st.markdown("#### Categorias")
    st.caption("Los grupos posibles en los que puede caer una persona al final del test.")
    for i, cat in enumerate(editor["categorias"]):
        col_cat, col_del = st.columns([6, 1])
        editor["categorias"][i] = col_cat.text_input(
            f"Categoria {i + 1}", value=cat, key=f"cat_txt_{i}", label_visibility="collapsed"
        )
        if col_del.button("Quitar", key=f"cat_del_{i}", use_container_width=True):
            if len(editor["categorias"]) <= 2:
                st.warning("Debe haber al menos 2 categorias.")
            else:
                cat_eliminada = editor["categorias"].pop(i)
                for p in editor["preguntas"]:
                    for op in p["opciones"]:
                        if op["categoria"] == cat_eliminada:
                            op["categoria"] = editor["categorias"][0]
                st.rerun()
    if st.button("Agregar categoria"):
        editor["categorias"].append(f"Categoria {len(editor['categorias']) + 1}")
        st.rerun()

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    st.markdown("#### Preguntas")

    for i, pregunta in enumerate(editor["preguntas"]):
        titulo_exp = pregunta["enunciado"].strip() or f"Pregunta {i + 1}"
        with st.expander(f"Pregunta {i + 1}: {titulo_exp[:60]}", expanded=False):
            pregunta["enunciado"] = st.text_area(
                "Enunciado (el texto que vera quien responde)",
                value=pregunta["enunciado"],
                key=f"preg_enun_{pregunta['id']}",
            )
            pregunta["peso"] = st.number_input(
                "Importancia de esta pregunta (peso)",
                min_value=1,
                max_value=10,
                value=int(pregunta["peso"]),
                key=f"preg_peso_{pregunta['id']}",
                help="Una pregunta con mas peso influye mas en el resultado final.",
            )

            st.caption("Opciones de respuesta y a que categoria pertenece cada una:")
            opciones_a_quitar = None
            for j, opcion in enumerate(pregunta["opciones"]):
                col_txt, col_cat, col_del = st.columns([4, 3, 1])
                opcion["texto"] = col_txt.text_input(
                    "Texto de la opcion",
                    value=opcion["texto"],
                    key=f"op_txt_{opcion['id']}",
                    label_visibility="collapsed",
                    placeholder=f"Opcion {j + 1}",
                )
                cat_actual = opcion["categoria"] if opcion["categoria"] in editor["categorias"] else editor["categorias"][0]
                opcion["categoria"] = col_cat.selectbox(
                    "Categoria",
                    editor["categorias"],
                    index=editor["categorias"].index(cat_actual),
                    key=f"op_cat_{opcion['id']}",
                    label_visibility="collapsed",
                )
                if col_del.button("Quitar", key=f"op_del_{opcion['id']}", use_container_width=True):
                    opciones_a_quitar = j

            if opciones_a_quitar is not None:
                if len(pregunta["opciones"]) <= 2:
                    st.warning("Cada pregunta necesita al menos 2 opciones.")
                else:
                    pregunta["opciones"].pop(opciones_a_quitar)
                    st.rerun()

            col_add_op, col_del_preg = st.columns(2)
            if col_add_op.button("Agregar opcion", key=f"op_add_{pregunta['id']}", use_container_width=True):
                pregunta["opciones"].append(
                    {"id": uuid.uuid4().hex, "texto": "", "categoria": editor["categorias"][0]}
                )
                st.rerun()
            if col_del_preg.button("Eliminar esta pregunta", key=f"preg_del_{pregunta['id']}", use_container_width=True):
                editor["preguntas"] = [p for p in editor["preguntas"] if p["id"] != pregunta["id"]]
                st.rerun()

    if st.button("Agregar pregunta nueva"):
        editor["preguntas"].append(_pregunta_nueva(editor["categorias"]))
        st.rerun()

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    col_guardar, col_reiniciar = st.columns(2)
    with col_guardar:
        if st.button("Guardar y usar este cuestionario", type="primary", use_container_width=True):
            try:
                cuestionario_guardado = _editor_a_cuestionario(editor)
            except ValueError as e:
                st.error(f"No se pudo guardar: {e}")
            else:
                st.session_state.cuestionario_guardado = cuestionario_guardado
                st.success(
                    f"Cuestionario guardado ({cuestionario_guardado.n_preguntas} preguntas, "
                    f"{len(cuestionario_guardado.categorias)} categorias). Ya puedes cargar el CSV de respuestas."
                )
    with col_reiniciar:
        if st.button("Reiniciar al de Hogwarts", use_container_width=True):
            st.session_state.editor_cuestionario = _cuestionario_a_editor(logica.CUESTIONARIO_HOGWARTS)
            st.rerun()

    with st.expander("Guardar una copia de respaldo (.json) — opcional"):
        st.caption(
            "Solo si quieres guardar este cuestionario como archivo, por ejemplo para "
            "compartirlo con alguien mas o usarlo despues en la opcion 'Subir un archivo'."
        )
        try:
            cuestionario_actual = _editor_a_cuestionario(editor)
            st.download_button(
                "Descargar copia (.json)",
                data=logica.plantilla_cuestionario_json(cuestionario_actual),
                file_name="cuestionario.json",
                mime="application/json",
            )
        except ValueError:
            st.caption("Completa el cuestionario para poder descargar la copia.")

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR: carga de datos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## El Gran Comedor")
    st.caption("Carga el pergamino de respuestas para comenzar la ceremonia.")

    archivo = st.file_uploader(
        "CSV exportado de Google Forms",
        type=["csv"],
        help="Menu de respuestas del formulario, luego el icono de opciones, y Descargar CSV.",
    )
    usar_ejemplo = st.checkbox("No tengo un CSV a la mano: usar datos de ejemplo")

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    st.markdown("## Cuestionario")
    st.caption("Elige que preguntas se van a usar para analizar las respuestas.")
    modo_cuestionario = st.radio(
        "Cuestionario a usar",
        [
            "Usar el de Hogwarts (recomendado)",
            "Editar o crear el mio",
            "Subir un archivo .json (avanzado)",
        ],
        key="modo_cuestionario",
        label_visibility="collapsed",
    )

    archivo_cuestionario = None
    if modo_cuestionario == "Subir un archivo .json (avanzado)":
        st.caption(
            "Pensado para quien ya tiene un archivo de cuestionario preparado, o quiere "
            "compartir el suyo con alguien mas."
        )
        archivo_cuestionario = st.file_uploader(
            "Cuestionario personalizado (.json)",
            type=["json"],
            help="Usa 'Descargar plantilla' abajo para ver el formato esperado.",
        )
        st.download_button(
            "Descargar plantilla (.json)",
            data=logica.plantilla_cuestionario_json(logica.CUESTIONARIO_HOGWARTS),
            file_name="cuestionario_plantilla.json",
            mime="application/json",
            help="Plantilla con el cuestionario de Hogwarts; editala para tu propio cuestionario.",
        )
    elif modo_cuestionario == "Editar o crear el mio":
        st.caption("El editor aparece en la parte de arriba de la pagina principal.")
        if "cuestionario_guardado" in st.session_state:
            cg = st.session_state.cuestionario_guardado
            st.success(f"Guardado: '{cg.nombre}' ({cg.n_preguntas} preguntas).")
        else:
            st.info("Aun no has guardado cambios; por ahora se usa el de Hogwarts.")

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    st.markdown("## Baul del Sombrero")
    st.caption("Opcional: sube un modelo ya entrenado (.pkl) para aplicarlo a este CSV sin reentrenar.")
    archivo_modelo = st.file_uploader(
        "Modelo entrenado (.pkl)",
        type=["pkl"],
        help="El archivo que descargaste antes en la pestaña Descargas → 'Modelo entrenado (.pkl)'.",
    )

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    if st.button("Reiniciar todo", use_container_width=True, help="Borra el CSV, el cuestionario editado, el modelo entrenado y los filtros; empieza de cero."):
        st.session_state.clear()
        st.rerun()

cuestionario = logica.CUESTIONARIO_HOGWARTS

if modo_cuestionario == "Editar o crear el mio":
    _render_editor_cuestionario()
    cuestionario = st.session_state.get("cuestionario_guardado", logica.CUESTIONARIO_HOGWARTS)
elif archivo_cuestionario is not None:
    try:
        cuestionario = logica.cargar_cuestionario_json(archivo_cuestionario.getvalue())
        with st.sidebar:
            st.success(
                f"Cuestionario cargado: '{cuestionario.nombre}' "
                f"({cuestionario.n_preguntas} preguntas, {len(cuestionario.categorias)} categorias)."
            )
    except ValueError as e:
        with st.sidebar:
            st.error(f"El cuestionario .json no es valido: {e}")
        st.stop()

modelo_cargado = None
if archivo_modelo is not None:
    try:
        modelo_cargado = joblib.load(archivo_modelo)
        with st.sidebar:
            st.success(f"Modelo cargado (entrenado con k={modelo_cargado.get('k', '?')}).")
    except Exception as e:
        with st.sidebar:
            st.error(f"No se pudo leer el archivo .pkl: {e}")

df = None
if archivo is not None:
    df = pd.read_csv(archivo)
elif usar_ejemplo:
    try:
        df = pd.read_csv("datos_ejemplo.csv")
        with st.sidebar:
            st.info("Usando datos_ejemplo.csv (respuestas sinteticas)")
    except FileNotFoundError:
        with st.sidebar:
            st.error(
                "No se encontro datos_ejemplo.csv. Ejecuta primero "
                "`python generar_datos_ejemplo.py`."
            )

if df is None:
    st.markdown(estilo.bienvenida_html(), unsafe_allow_html=True)
    st.stop()

columnas = logica.detectar_columnas(df)

columnas_ok, mensaje_columnas = logica.validar_columnas_preguntas(columnas.preguntas, cuestionario)
if not columnas_ok:
    st.error(mensaje_columnas)
    st.stop()
elif mensaje_columnas:
    st.warning(mensaje_columnas)

puntajes_completo = logica.calcular_puntajes(df, columnas.preguntas, cuestionario)
faltantes_completo = logica.contar_respuestas_faltantes(df, columnas.preguntas, cuestionario)
n_incompletos = int((faltantes_completo > 0).sum())

# ---------------------------------------------------------------------------
# SIDEBAR: filtros (siempre visibles, sobre el pergamino completo)
# ---------------------------------------------------------------------------
col_genero = columnas.metadatos.get("genero")
col_ocupacion = columnas.metadatos.get("ocupacion")
col_estres = columnas.metadatos.get("estres")
col_edad = columnas.metadatos.get("edad")
col_correo = columnas.metadatos.get("correo")

CLAVES_FILTRO = ["f_genero", "f_ocupacion", "f_estres", "f_edad", "f_casa", "f_busqueda"]

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filtros")

    mask = pd.Series(True, index=df.index)

    if col_correo:
        busqueda = st.text_input(
            "Buscar por correo", key="f_busqueda", placeholder="ej. persona1@ejemplo.com"
        )
        if busqueda:
            mask &= df[col_correo].astype(str).str.contains(busqueda, case=False, na=False)

    if col_genero:
        opciones = sorted(df[col_genero].dropna().unique().tolist())
        sel = st.multiselect("Genero", opciones, default=opciones, key="f_genero")
        mask &= df[col_genero].isin(sel)

    if col_ocupacion:
        opciones = sorted(df[col_ocupacion].dropna().unique().tolist())
        sel = st.multiselect("Ocupacion", opciones, default=opciones, key="f_ocupacion")
        mask &= df[col_ocupacion].isin(sel)

    if col_estres:
        valores_estres = df[col_estres].dropna().astype(str).str.strip().str.title()
        orden_estandar = ["Bajo", "Medio", "Alto"]
        presentes_ordenados = [n for n in orden_estandar if n in valores_estres.unique()]
        if presentes_ordenados and set(valores_estres.unique()) <= set(orden_estandar):
            rango_sel = st.select_slider(
                "Nivel de estres",
                options=presentes_ordenados,
                value=(presentes_ordenados[0], presentes_ordenados[-1]),
                key="f_estres",
            )
            i0, i1 = presentes_ordenados.index(rango_sel[0]), presentes_ordenados.index(rango_sel[1])
            permitidos = presentes_ordenados[i0:i1 + 1]
            mask &= valores_estres.reindex(df.index).isin(permitidos)
        else:
            opciones = sorted(df[col_estres].dropna().unique().tolist())
            sel = st.multiselect("Nivel de estres", opciones, default=opciones, key="f_estres")
            mask &= df[col_estres].isin(sel)

    if col_edad:
        edades = pd.to_numeric(df[col_edad], errors="coerce").dropna()
        if not edades.empty:
            e_min, e_max = int(edades.min()), int(edades.max())
            if e_min == e_max:
                e_max += 1
            rango = st.slider("Rango de edad", e_min, e_max, (e_min, e_max), key="f_edad")
            edad_num = pd.to_numeric(df[col_edad], errors="coerce")
            mask &= edad_num.between(rango[0], rango[1])

    casas_presentes = sorted(puntajes_completo["Casa_dominante"].dropna().unique().tolist())
    if casas_presentes:
        sel_casas = st.multiselect(
            "Casa dominante", casas_presentes, default=casas_presentes, key="f_casa"
        )
        mask &= puntajes_completo["Casa_dominante"].isin(sel_casas)

    if st.button("Restablecer filtros", use_container_width=True):
        for clave in CLAVES_FILTRO:
            st.session_state.pop(clave, None)
        st.rerun()

    st.markdown("---")
    n_filtrado, n_total = int(mask.sum()), len(df)
    proporcion = (n_filtrado / n_total) if n_total else 0
    st.progress(proporcion)
    st.markdown(
        f'<div class="hp-contador">{n_filtrado} de {n_total} aspirantes seleccionados</div>',
        unsafe_allow_html=True,
    )

df_filtrado = df[mask].reset_index(drop=True)
puntajes = puntajes_completo[mask].reset_index(drop=True)
faltantes_filtrado = faltantes_completo[mask].reset_index(drop=True)

if len(df_filtrado) < 4:
    st.error("El Sombrero necesita al menos 4 aspirantes despues del filtro para poder agrupar en casas.")
    st.stop()

# ---------------------------------------------------------------------------
# PESTANAS
# ---------------------------------------------------------------------------
modelo_listo = bool(st.session_state.get("modelo_entrenado"))

st.markdown(
    f"""
    <div class="hp-pasos">
        <div class="hp-paso hp-paso-hecho">1. Revisa tus datos</div>
        <div class="hp-paso hp-paso-hecho">2. Consulta estadisticas</div>
        <div class="hp-paso {'hp-paso-hecho' if modelo_listo else 'hp-paso-activo'}">3. Entrena el modelo</div>
        <div class="hp-paso {'hp-paso-activo' if modelo_listo else ''}">4. Mira los resultados</div>
        <div class="hp-paso">5. Descarga lo que necesites</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_datos, tab_stats, tab_train, tab_resultados, tab_individual, tab_descargas = st.tabs(
    [
        "1. Datos",
        "2. Estadisticas",
        "3. Entrenamiento",
        "4. Resultados",
        "5. Seleccion individual",
        "6. Descargas",
    ]
)

# ===========================================================================
# TAB 1: DATOS — carga, columnas detectadas y las cuatro casas
# ===========================================================================
with tab_datos:
    st.markdown("<h2>El pergamino de respuestas</h2>", unsafe_allow_html=True)
    st.caption(
        "Este es tu punto de partida: revisa que tus datos se hayan cargado bien "
        "antes de seguir a la pestaña 'Estadisticas' o 'Entrenamiento'."
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Aspirantes totales", n_total)
    col_m2.metric("Aspirantes tras filtro", n_filtrado)
    col_m3.metric("Preguntas detectadas", len(columnas.preguntas))
    col_m4.metric("Con respuestas incompletas", int((faltantes_filtrado > 0).sum()))

    if (faltantes_filtrado > 0).any():
        st.warning(
            f"{int((faltantes_filtrado > 0).sum())} de {n_filtrado} aspirante(s) dejaron "
            "al menos una de las 16 preguntas sin responder (o con una opcion que no se "
            "reconoce). Esas preguntas simplemente no suman puntos a ninguna casa, lo que "
            "puede sesgar levemente su puntaje total hacia abajo."
        )

    st.dataframe(df_filtrado, use_container_width=True)

    with st.expander("Ver columnas detectadas automaticamente"):
        st.write("Metadatos:", columnas.metadatos)
        st.write(f"Columnas de preguntas detectadas ({len(columnas.preguntas)}):", columnas.preguntas)

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)

    if cuestionario is logica.CUESTIONARIO_HOGWARTS:
        st.markdown("#### Las cuatro casas de Hogwarts", unsafe_allow_html=True)

        # Crear 2 filas de 2 columnas para imagenes mas grandes
        for i in range(0, 4, 2):
            cols_casas = st.columns(2)
            for j, casa in enumerate(cuestionario.categorias[i:i+2]):
                with cols_casas[j]:
                    color = logica.COLORES.get(casa, "#c9a227")
                    img_path = logica.CASA_IMG_MAP.get(casa, "")
                    desc = logica.CASA_DESCRIPCIONES.get(casa, {})
                
                # Contenedor con fondo oscuro y borde del color de la casa
                st.markdown(
                    f"""
                    <div style='
                        background: rgba(15, 15, 30, 0.92);
                        backdrop-filter: blur(10px);
                        border-radius: 16px;
                        padding: 25px 20px;
                        text-align: center;
                        margin-bottom: 20px;
                        border: 3px solid {color};
                        box-shadow: 0 4px 30px rgba(0,0,0,0.5), inset 0 0 30px {color}22;
                    '>
                    """,
                    unsafe_allow_html=True
                )
                
                # Imagen de la casa (centrada)
                try:
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: center; margin-bottom: 15px;'>
                            <img src='data:image/png;base64,{img_base64}' 
                                 style='width: 160px; height: auto;'/>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except:
                    color_txt = logica.COLORES.get(casa, "#c9a227")
                    st.markdown(
                        f"<div style='text-align:center; font-family: Cinzel Decorative, serif; "
                        f"font-size:22px; color:{color_txt}; padding:20px 0;'>{casa}</div>",
                        unsafe_allow_html=True
                    )
                
                # Nombre de la casa - BLANCO con sombra
                st.markdown(
                    f"""
                    <h2 style='
                        color: #FFFFFF;
                        font-family: Cinzel Decorative, serif;
                        font-size: 32px;
                        margin: 5px 0;
                        text-shadow: 0 0 30px {color}, 0 0 60px {color}66, 0 4px 8px rgba(0,0,0,0.8);
                        letter-spacing: 3px;
                        font-weight: 900;
                    '>{casa.upper()}</h2>
                    """,
                    unsafe_allow_html=True
                )
                
                # Subtitulo - BLANCO
                st.markdown(
                    f"""
                    <p style='
                        color: #FFFFFF;
                        font-size: 18px;
                        font-weight: 700;
                        margin: 5px 0 10px 0;
                        letter-spacing: 4px;
                        text-transform: uppercase;
                        text-shadow: 0 0 20px {color}, 0 2px 4px rgba(0,0,0,0.8);
                    '>{desc.get('subtitulo', '')}</p>
                    """,
                    unsafe_allow_html=True
                )
                
                # Linea decorativa
                st.markdown(
                    f"""
                    <hr style='
                        border: none;
                        height: 2px;
                        background: linear-gradient(90deg, transparent, {color}, transparent);
                        margin: 12px auto;
                        width: 60%;
                    '>
                    """,
                    unsafe_allow_html=True
                )
                
                # Descripcion - BLANCO con sombra
                st.markdown(
                    f"""
                    <p style='
                        color: #FFFFFF;
                        font-size: 16px;
                        line-height: 1.8;
                        text-align: center;
                        margin: 12px 0;
                        max-width: 400px;
                        margin-left: auto;
                        margin-right: auto;
                        font-weight: 600;
                        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
                    '>{desc.get('descripcion', '')}</p>
                    """,
                    unsafe_allow_html=True
                )
                
                # Cita - BLANCO
                st.markdown(
                    f"""
                    <p style='
                        color: #FFFFFF;
                        font-size: 15px;
                        font-style: italic;
                        text-align: center;
                        margin: 15px 0 0 0;
                        border-top: 2px solid {color}66;
                        padding-top: 15px;
                        font-weight: 700;
                        letter-spacing: 0.5px;
                        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
                    '>{desc.get('cita', '')}</p>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"#### Categorias de '{cuestionario.nombre}'", unsafe_allow_html=True)
        st.caption(
            "Cuestionario personalizado cargado: se omiten las tarjetas ilustradas de Hogwarts "
            "porque no aplican a estas categorias."
        )
        cols_cat = st.columns(min(4, len(cuestionario.categorias)))
        for i, cat in enumerate(cuestionario.categorias):
            with cols_cat[i % len(cols_cat)]:
                st.metric(cat, f"peso total: {sum(p for j, p in enumerate(cuestionario.pesos) if cat in cuestionario.preguntas[j].values())}")

# ===========================================================================
# TAB 2: ESTADISTICAS
# ===========================================================================
with tab_stats:
    st.markdown("<h2>Estadistica basica</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Puntaje por casa (variables numericas)")
        columnas_pct = [f"{c}_pct" for c in cuestionario.categorias]
        resumen = logica.resumen_estadistico(puntajes, columnas_pct)
        st.dataframe(resumen, use_container_width=True)

    with col_b:
        st.subheader("Casa dominante (moda y frecuencias)")
        frecuencias = logica.tabla_frecuencias(puntajes["Casa_dominante"].tolist())
        modas = logica.moda_propia(puntajes["Casa_dominante"].tolist())
        if modas:
            etiqueta = "Casas mas frecuentes (moda)" if len(modas) > 1 else "Casa mas frecuente (moda)"
            badges = " ".join(estilo.badge_casa_html(casa) for casa in modas)
            st.markdown(f"{etiqueta}: {badges}", unsafe_allow_html=True)
        else:
            st.markdown("Sin datos suficientes.", unsafe_allow_html=True)
        tabla_frec = frecuencias.reset_index()
        tabla_frec.columns = ["Casa", "Respondientes"]
        fig_frec = px.bar(
            tabla_frec,
            x="Casa",
            y="Respondientes",
            color="Casa",
            color_discrete_map=logica.COLORES,
        )
        fig_frec.update_layout(
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_frec, use_container_width=True)

    if col_edad and col_edad in df_filtrado.columns:
        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.subheader("Otras variables numericas")
        otras_cols = [c for c in [col_edad, columnas.metadatos.get("sueno")] if c]
        if otras_cols:
            st.dataframe(logica.resumen_estadistico(df_filtrado, otras_cols), use_container_width=True)

# ===========================================================================
# TAB 3: ENTRENAMIENTO
# ===========================================================================
with tab_train:
    _mensaje_toast = st.session_state.pop("toast_modelo", None)
    if _mensaje_toast:
        st.toast(_mensaje_toast)
    st.markdown("<h2>El Sombrero aprende a agrupar</h2>", unsafe_allow_html=True)
    st.caption(
        "El sistema analiza los porcentajes de cada aspirante en cada categoria y los agrupa "
        "automaticamente segun que tan parecidas son sus respuestas."
    )
    with st.expander("Ver detalle tecnico (opcional)"):
        st.caption(
            "Se ajustan, en este orden: StandardScaler -> KMeans(k) -> PCA(2), sobre los "
            "porcentajes por categoria de los aspirantes filtrados."
        )

    X = logica.preparar_features(puntajes, cuestionario.categorias)

    with st.expander("¿No sabes cuantos grupos (k) usar? Aqui hay una guia"):
        st.markdown(
            f"Estas dos graficas ayudan a decidir cuantos grupos (k) formar. Con la "
            f"{estilo.tooltip_html('inercia', 'Que tan cerca queda cada aspirante del centro de su grupo. Siempre baja al agregar mas grupos, por eso sola no basta para decidir.')} "
            f"sola no alcanza, porque siempre mejora al agregar mas grupos. Por eso se compara junto con la "
            f"{estilo.tooltip_html('silueta', 'Que tan bien separados quedan los grupos entre si. Va de -1 a 1; mientras mas alto, mejor separados quedan.')}, "
            "que si penaliza cuando los grupos quedan mal separados.",
            unsafe_allow_html=True,
        )
        X_esc_referencia = StandardScaler().fit_transform(X)
        col_codo, col_silueta = st.columns(2)

        with col_codo:
            inercias = logica.calcular_inercias(X_esc_referencia)
            fig_codo = px.line(inercias, x="k", y="Inercia", markers=True, title="Inercia vs. k")
            fig_codo.update_traces(line_color="#c9a227")
            fig_codo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_codo, use_container_width=True)

        with col_silueta:
            siluetas = logica.calcular_siluetas(X_esc_referencia)
            fig_silueta = px.line(siluetas, x="k", y="Silueta", markers=True, title="Silueta vs. k")
            fig_silueta.update_traces(line_color="#7a1f1f")
            fig_silueta.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_silueta, use_container_width=True)

    k = st.slider(
        "Numero de grupos a formar (k)",
        min_value=2,
        max_value=6,
        value=4,
        help="En el cuestionario de Hogwarts esto equivale a las 4 casas; puedes usar entre 2 y 6 grupos.",
    )

    if "modelo_entrenado" not in st.session_state:
        st.session_state.modelo_entrenado = False

    if st.session_state.get("modelo_entrenado"):
        origen = st.session_state.get("modelo_origen", "entrenado")
        col_badge, col_reset = st.columns([3, 1])
        with col_badge:
            if origen == "cargado":
                st.info(
                    f"Modelo activo: **cargado desde .pkl** (k={st.session_state.get('k_entrenado')}), "
                    "aplicado a este CSV con transform()/predict()."
                )
            else:
                st.info(f"Modelo activo: **entrenado en esta sesion** (k={st.session_state.get('k_entrenado')}).")
        with col_reset:
            if st.button("Limpiar modelo", use_container_width=True):
                for clave in [
                    "modelo_entrenado", "modelo_origen", "k_entrenado", "escalador", "kmeans",
                    "pca", "etiquetas", "coords_pca", "puntajes", "df_filtrado", "metricas_modelo",
                ]:
                    st.session_state.pop(clave, None)
                st.rerun()

    if modelo_cargado is not None:
        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.subheader("Usar el modelo ya entrenado que subiste")
        st.caption(
            f"Se aplicara el modelo cargado (k={modelo_cargado.get('k', '?')}) a este CSV "
            "usando solo transform()/predict(), sin reentrenar."
        )
        modelo_ok, mensaje_modelo = logica.validar_modelo_features(modelo_cargado, X, cuestionario.categorias)
        if not modelo_ok:
            st.error(mensaje_modelo)
        elif st.button("Aplicar modelo cargado a este CSV"):
            with st.spinner("El Sombrero recuerda lo aprendido…"):
                time.sleep(0.8)
                etiquetas, coords_pca = logica.aplicar_modelo_guardado(modelo_cargado, X)
                X_esc_aplicado = modelo_cargado["escalador"].transform(X)
                try:
                    silueta_modelo = silhouette_score(X_esc_aplicado, etiquetas)
                except ValueError:
                    silueta_modelo = float("nan")
            st.session_state.modelo_entrenado = True
            st.session_state.modelo_origen = "cargado"
            st.session_state.k_entrenado = modelo_cargado.get("k")
            st.session_state.escalador = modelo_cargado["escalador"]
            st.session_state.kmeans = modelo_cargado["kmeans"]
            st.session_state.pca = modelo_cargado["pca"]
            st.session_state.etiquetas = etiquetas
            st.session_state.coords_pca = coords_pca
            st.session_state.puntajes = puntajes
            st.session_state.df_filtrado = df_filtrado
            st.session_state.metricas_modelo = {
                "inercia": getattr(modelo_cargado["kmeans"], "inertia_", None),
                "silueta": silueta_modelo,
                "varianza_explicada_pca": modelo_cargado["pca"].explained_variance_ratio_.tolist(),
            }
            st.session_state.toast_modelo = f"Se aplico el modelo cargado a {len(df_filtrado)} aspirante(s)."
            st.rerun()
        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.caption("O, si prefieres, entrena un modelo nuevo con estos datos:")

    if st.button("Susurrarle al Sombrero: entrenar modelo"):
        if k > len(df_filtrado):
            st.markdown(
                estilo.advertencia_html(
                    titulo="El Sombrero no puede continuar...",
                    mensaje=(
                        f"Tienes solo <b>{len(df_filtrado)}</b> aspirante(s) frente a él, pero le pides "
                        f"repartirlos en <b>{k}</b> casas. Ni con toda la magia de Hogwarts puede formar "
                        "mas casas que aspirantes hay disponibles.<br><br>"
                        "Reduce el numero de grupos (k) o amplia tu seleccion de aspirantes con los "
                        "filtros de la barra lateral, e intenta de nuevo."
                    ),
                ),
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Dificil… muy dificil… el Sombrero esta pensando…"):
                time.sleep(1.2)
                escalador, kmeans, pca, etiquetas, coords_pca = logica.entrenar_kmeans(X, k)
                X_esc_entrenado = escalador.transform(X)
                try:
                    silueta_modelo = silhouette_score(X_esc_entrenado, etiquetas)
                except ValueError:
                    silueta_modelo = float("nan")
            st.session_state.modelo_entrenado = True
            st.session_state.modelo_origen = "entrenado"
            st.session_state.k_entrenado = k
            st.session_state.escalador = escalador
            st.session_state.kmeans = kmeans
            st.session_state.pca = pca
            st.session_state.etiquetas = etiquetas
            st.session_state.coords_pca = coords_pca
            st.session_state.puntajes = puntajes
            st.session_state.df_filtrado = df_filtrado
            st.session_state.metricas_modelo = {
                "inercia": kmeans.inertia_,
                "silueta": silueta_modelo,
                "varianza_explicada_pca": pca.explained_variance_ratio_.tolist(),
            }
            st.session_state.toast_modelo = f"El Sombrero termino de ordenar a {len(df_filtrado)} aspirantes en {k} casas."
            st.rerun()


# ===========================================================================
# TAB 4: RESULTADOS
# ===========================================================================
with tab_resultados:
    if st.session_state.get("modelo_entrenado"):
        st.markdown("<h2>Resultados de la ceremonia de seleccion</h2>", unsafe_allow_html=True)
        st.caption(
            "Estos son los grupos que el sistema formo a partir de las respuestas, sin que se le "
            "dijera antes a que casa pertenecia cada aspirante."
        )

        etiquetas = st.session_state.etiquetas
        coords_pca = st.session_state.coords_pca
        puntajes_r = st.session_state.puntajes.copy()
        puntajes_r["Cluster"] = etiquetas.astype(str)
        puntajes_r["PCA_1"] = coords_pca[:, 0]
        puntajes_r["PCA_2"] = coords_pca[:, 1]

        resultados = pd.concat(
            [st.session_state.df_filtrado.reset_index(drop=True), puntajes_r.reset_index(drop=True)],
            axis=1,
        )

        metricas = st.session_state.get("metricas_modelo", {})
        if metricas:
            varianza = metricas.get("varianza_explicada_pca")
            col_met1, col_met2, col_met3 = st.columns(3)
            if metricas.get("silueta") is not None:
                col_met1.metric(
                    "Coeficiente de silueta",
                    f"{metricas['silueta']:.3f}",
                    help="Que tan bien separados quedan los grupos entre si. Va de -1 a 1; mas alto es mejor.",
                )
            if metricas.get("inercia") is not None:
                col_met2.metric(
                    "Inercia (K-Means)",
                    f"{metricas['inercia']:.1f}",
                    help="Que tan compactos quedan los aspirantes dentro de su propio grupo. Sirve para comparar entre distintos valores de k, no tiene un 'bueno' absoluto.",
                )
            if varianza:
                col_met3.metric(
                    "Varianza explicada por el PCA 2D",
                    f"{sum(varianza) * 100:.1f}%",
                    help="Que tanta de la informacion original se conserva al resumir todo en el grafico 2D de abajo.",
                )
            st.caption(
                "La varianza explicada indica que tan fiel es el grafico 2D de PCA respecto a las "
                "4 variables originales; si es baja, el scatter de abajo simplifica bastante la realidad."
            )
            st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.subheader("Mapa de grupos (2D)")
            fig_pca = px.scatter(
                puntajes_r,
                x="PCA_1",
                y="PCA_2",
                color="Cluster",
                hover_data=["Casa_dominante"] + [f"{c}_pct" for c in cuestionario.categorias],
                title="Reduccion de dimensionalidad (PCA) coloreada por grupo",
            )
            fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_r2:
            st.subheader("Casa dominante vs. grupo asignado")
            cruce = pd.crosstab(puntajes_r["Casa_dominante"], puntajes_r["Cluster"])
            st.dataframe(cruce, use_container_width=True)
            st.caption(
                "Si cada grupo (cluster) concentra mayormente una casa dominante, el algoritmo "
                "esta agrupando correctamente perfiles similares sin haber usado esa "
                "etiqueta durante el entrenamiento (aprendizaje no supervisado)."
            )

        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.subheader("Perfil promedio de cada grupo")
        perfil_cluster = puntajes_r.groupby("Cluster")[[f"{c}_pct" for c in cuestionario.categorias]].mean().round(1)
        st.dataframe(perfil_cluster, use_container_width=True)

        fig_radar = go.Figure()
        for cluster_id, fila in perfil_cluster.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=fila.values.tolist() + [fila.values[0]],
                    theta=cuestionario.categorias + [cuestionario.categorias[0]],
                    fill="toself",
                    name=f"Grupo {cluster_id}",
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Perfil promedio por grupo (radar)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.subheader("Tabla completa de resultados")
        st.dataframe(resultados, use_container_width=True)
    else:
        st.info("Entrena el modelo en la pestaña 'Entrenamiento' para ver los resultados aqui.")

# ===========================================================================
# TAB 5: SOMBRERO SELECCIONADOR INDIVIDUAL
# ===========================================================================
with tab_individual:
    st.markdown(
        '<div class="hp-pergamino">'
        "<h3>Ceremonia de seleccion individual</h3>"
        "<p>Elige a un aspirante del pergamino de respuestas (ya filtrado) para que el "
        "Sombrero anuncie su casa, con base en el puntaje calculado a partir de sus "
        "16 respuestas.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    etiquetas_aspirantes = (
        df_filtrado[col_correo].astype(str).tolist()
        if col_correo
        else [f"Aspirante {i+1}" for i in range(len(df_filtrado))]
    )

    idx_sel = st.selectbox(
        "Aspirante",
        options=list(range(len(df_filtrado))),
        format_func=lambda i: etiquetas_aspirantes[i],
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        iniciar_seleccion = st.button("¡Que comience la seleccion!", use_container_width=True)

    if iniciar_seleccion:
        fila_puntajes = puntajes.iloc[idx_sel]
        casa_final = fila_puntajes["Casa_dominante"]

        # Contenedor para la animacion
        contenedor = st.empty()
        
        # Mostrar el video del Sombrero mientras piensa
        try:
            with open("Sombrero.mp4", "rb") as f:
                video_bytes = f.read()
            contenedor.video(video_bytes, format="video/mp4", autoplay=True, muted=True, loop=False)
        except:
            contenedor.markdown(
                "<p style='text-align:center; font-family: Cinzel Decorative, serif; "
                "font-size:28px; color:#c9a227; padding:30px 0;'>El Sombrero esta pensando...</p>",
                unsafe_allow_html=True,
            )
        
        # Mensajes del Sombrero
        mensajes = [
            "El Sombrero se coloca sobre la cabeza del aspirante…",
            "Mmm… interesante combinacion de respuestas…",
            "¡Ya se donde ponerte!",
        ]
        
        for mensaje in mensajes:
            contenedor.markdown(
                f"""
                <div style='text-align:center;'>
                    <p style='font-style:italic; color:#cbb994; font-size:22px; margin-top:15px;'>
                        {mensaje}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(1.2)
        
        contenedor.empty()
        
        # Mostrar resultado con imagen de la casa
        st.markdown("---")
        
        col_resultado1, col_resultado2, col_resultado3 = st.columns([1, 2, 1])
        
        with col_resultado2:
            img_path = logica.CASA_IMG_MAP.get(casa_final, "")
            try:
                with open(img_path, "rb") as f:
                    casa_img = f.read()
                img_base64 = base64.b64encode(casa_img).decode()
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: center;'>
                        <img src='data:image/png;base64,{img_base64}' 
                             style='width: 280px; height: auto;'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                color = logica.COLORES.get(casa_final, "#c9a227")
                st.markdown(
                    f"""
                    <div style='text-align:center;'>
                        <h1 style='color:#FFFFFF; font-family: Cinzel Decorative, serif; 
                                   font-size:42px; text-shadow: 0 0 30px {color}, 0 0 60px {color}66, 0 4px 8px rgba(0,0,0,0.8);'>
                            {casa_final.upper()}
                        </h1>
                        <p style='color:#FFFFFF; font-size:20px; text-shadow: 0 2px 8px rgba(0,0,0,0.8);'>¡El Sombrero ha decidido!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                color = logica.COLORES.get(casa_final, "#c9a227")
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:40px; border-radius:20px; 
                         background: linear-gradient(135deg, #1a1a2e, #16213e); 
                         border: 3px solid {color};'>
                        <h1 style='color:#FFFFFF; font-size:48px; font-family: Cinzel Decorative, serif; 
                                   text-shadow: 0 0 30px {color}, 0 4px 8px rgba(0,0,0,0.8);'>
                            {casa_final.upper()}
                        </h1>
                        <p style='color:#FFFFFF; font-size:22px; text-shadow: 0 2px 8px rgba(0,0,0,0.8);'>¡El Sombrero ha decidido!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.balloons()
        
        st.write("")
        st.markdown("---")
        st.subheader("Detalle de puntajes por casa")
        
        cols_puntajes = st.columns(len(cuestionario.categorias))
        for col, casa in zip(cols_puntajes, cuestionario.categorias):
            with col:
                img_path = logica.CASA_IMG_MAP.get(casa, "")
                try:
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: center;'>
                            <img src='data:image/png;base64,{img_base64}' 
                                 style='width: 70px; height: auto;'/>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except:
                    color_txt = logica.COLORES.get(casa, "#cbb994")
                    st.markdown(
                        f"<div style='text-align:center; font-family: Cinzel Decorative, serif; "
                        f"font-size:16px; color:{color_txt};'>{casa}</div>",
                        unsafe_allow_html=True
                    )
                
                puntaje = fila_puntajes[f'{casa}_pct']
                color = logica.COLORES.get(casa, "#cbb994")
                
                border_style = f"border: 3px solid {color};" if casa == casa_final else f"border: 1px solid {color}44;"
                bg_style = f"background: rgba({color.replace('#', '')}22, 0.15);" if casa == casa_final else "background: rgba(255,255,255,0.05);"
                
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:12px; border-radius:12px; 
                         {bg_style} {border_style}'>
                        <p style='color:{color}; font-weight:bold; font-size:14px; margin:0;'>{casa}</p>
                        <p style='color:#FFFFFF; font-size:30px; font-weight:bold; margin:5px 0; text-shadow: 0 2px 8px rgba(0,0,0,0.5);'>{puntaje:.1f}%</p>
                        {'<p style="color:#c9a227; font-size:12px; margin:0; font-weight:bold;">Ganadora</p>' if casa == casa_final else ''}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.write("")
        fig_barras = px.bar(
            x=cuestionario.categorias,
            y=[fila_puntajes[f"{c}_pct"] for c in cuestionario.categorias],
            color=cuestionario.categorias,
            color_discrete_map=logica.COLORES,
            labels={"x": "Casa", "y": "Porcentaje"},
            title="Distribucion de puntajes por casa"
        )
        fig_barras.update_layout(
            showlegend=False, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbb994",
            title_font_color="#c9a227",
            title_font_size=20
        )
        fig_barras.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside',
            textfont_color='#cbb994'
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        if st.session_state.get("modelo_entrenado"):
            cluster_asignado = st.session_state.etiquetas[
                st.session_state.puntajes.index.get_loc(fila_puntajes.name)
            ]
            st.info(f"Segun el modelo entrenado, este aspirante quedo en el **grupo {cluster_asignado}**.")
        else:
            st.caption("Entrena el modelo en la pestaña 'Entrenamiento' para ver tambien su grupo asignado.")

# ===========================================================================
# TAB 6: DESCARGAS
# ===========================================================================
with tab_descargas:
    st.markdown("<h2>Descargas</h2>", unsafe_allow_html=True)

    st.subheader("Datos filtrados")
    st.download_button(
        "Datos filtrados (CSV)",
        data=df_filtrado.to_csv(index=False).encode("utf-8-sig"),
        file_name="datos_filtrados.csv",
        mime="text/csv",
    )

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)

    if st.session_state.get("modelo_entrenado"):
        etiquetas = st.session_state.etiquetas
        coords_pca = st.session_state.coords_pca
        puntajes_r = st.session_state.puntajes.copy()
        puntajes_r["Cluster"] = etiquetas.astype(str)
        puntajes_r["PCA_1"] = coords_pca[:, 0]
        puntajes_r["PCA_2"] = coords_pca[:, 1]
        resultados = pd.concat(
            [st.session_state.df_filtrado.reset_index(drop=True), puntajes_r.reset_index(drop=True)],
            axis=1,
        )

        st.subheader("Resultados del modelo")
        st.download_button(
            "Resultados del modelo (CSV)",
            data=resultados.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultados_clustering.csv",
            mime="text/csv",
        )

        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)

        st.subheader("Modelo entrenado")
        metricas = st.session_state.get("metricas_modelo", {})
        buffer = io.BytesIO()
        joblib.dump(
            {
                "escalador": st.session_state.escalador,
                "kmeans": st.session_state.kmeans,
                "pca": st.session_state.pca,
                "k": st.session_state.k_entrenado,
                "features": [f"{c}_pct" for c in cuestionario.categorias],
                "metricas": metricas,
            },
            buffer,
        )
        buffer.seek(0)
        st.download_button(
            "Modelo entrenado (.pkl)",
            data=buffer,
            file_name="modelo_kmeans_pca.pkl",
            mime="application/octet-stream",
        )

        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)

        st.subheader("Reporte ejecutivo")
        st.caption("Resumen en Word con las metricas del modelo y las tablas de resultados, listo para anexar a tu documento.")
        perfil_cluster = puntajes_r.groupby("Cluster")[[f"{c}_pct" for c in cuestionario.categorias]].mean().round(1)
        try:
            reporte_bytes = logica.generar_reporte_docx(
                k=st.session_state.k_entrenado,
                n_aspirantes=len(st.session_state.df_filtrado),
                metricas=metricas,
                cruce=pd.crosstab(puntajes_r["Casa_dominante"], puntajes_r["Cluster"]),
                perfil_cluster=perfil_cluster,
                origen_modelo=st.session_state.get("modelo_origen", "entrenado"),
                cuestionario=cuestionario,
            )
            st.download_button(
                "Reporte ejecutivo (.docx)",
                data=reporte_bytes,
                file_name="reporte_sombrero_seleccionador.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except ImportError:
            st.info("Instala `python-docx` (ya esta en requirements.txt) para habilitar esta descarga.")
    else:
        st.info("Entrena el modelo en la pestaña 'Entrenamiento' para habilitar la descarga de resultados y del modelo (.pkl).")