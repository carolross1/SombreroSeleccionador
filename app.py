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

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import estilo
import logica

st.set_page_config(page_title="El Sombrero Seleccionador — Análisis No Supervisado", page_icon="🎩", layout="wide")
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
# SIDEBAR: carga de datos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## El Gran Comedor")
    st.caption("Carga el pergamino de respuestas para comenzar la ceremonia.")

    archivo = st.file_uploader(
        "CSV exportado de Google Forms",
        type=["csv"],
        help="Menu de respuestas del formulario → ⋮ → Descargar CSV",
    )
    usar_ejemplo = st.checkbox("No tengo un CSV a la mano: usar datos de ejemplo")

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
puntajes_completo = logica.calcular_puntajes(df, columnas.preguntas)

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

if len(df_filtrado) < 4:
    st.error("El Sombrero necesita al menos 4 aspirantes despues del filtro para poder agrupar en casas.")
    st.stop()

# ---------------------------------------------------------------------------
# PESTANAS
# ---------------------------------------------------------------------------
tab_datos, tab_stats, tab_train, tab_resultados, tab_individual, tab_descargas = st.tabs(
    [
        "Datos",
        "Estadisticas",
        "Entrenamiento",
        "Resultados",
        "Seleccion individual",
        "Descargas",
    ]
)

# ===========================================================================
# TAB 1: DATOS — carga, columnas detectadas y las cuatro casas
# ===========================================================================
with tab_datos:
    st.markdown("<h2>El pergamino de respuestas</h2>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Aspirantes totales", n_total)
    col_m2.metric("Aspirantes tras filtro", n_filtrado)
    col_m3.metric("Preguntas detectadas", len(columnas.preguntas))

    st.dataframe(df_filtrado, use_container_width=True)

    with st.expander("Ver columnas detectadas automaticamente"):
        st.write("Metadatos:", columnas.metadatos)
        st.write(f"Columnas de preguntas detectadas ({len(columnas.preguntas)}):", columnas.preguntas)
        if len(columnas.preguntas) != logica.N_PREGUNTAS:
            st.warning(
                f"Se esperaban {logica.N_PREGUNTAS} columnas de preguntas y se detectaron "
                f"{len(columnas.preguntas)}. Verifica que el CSV conserve el orden original "
                "de las 16 preguntas del formulario."
            )

    st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
    st.markdown("#### Las cuatro casas de Hogwarts", unsafe_allow_html=True)
    
    # Mapa de imagenes de casas
    casa_img_map = {
        "Gryffindor": "gryffindor.png",
        "Hufflepuff": "hufflepuff.png",
        "Ravenclaw": "ravenclaw.png",
        "Slytherin": "slytherin.png"
    }
    
    # Descripciones de las casas
    casa_descripciones = {
        "Gryffindor": {
            "subtitulo": "VALENTIA · LEON",
            "descripcion": "ACTUA PRIMERO Y PIENSA DESPUES; NO REHUYE EL RIESGO CUANDO HAY ALGO IMPORTANTE EN JUEGO.",
            "cita": '"EL VALOR NO ES LA AUSENCIA DE MIEDO, SINO ACTUAR A PESAR DE EL."'
        },
        "Slytherin": {
            "subtitulo": "AMBICION · SERPIENTE",
            "descripcion": "PIENSA EN ESTRATEGIA Y RESULTADOS; SABE RECONOCER Y APROVECHAR UNA OPORTUNIDAD.",
            "cita": '"TODA GRAN META EMPIEZA CON UNA JUGADA INTELIGENTE."'
        },
        "Ravenclaw": {
            "subtitulo": "SABIDURIA · AGUILA",
            "descripcion": "PREFIERE ENTENDER ANTES QUE ACTUAR; DISFRUTA ANALIZAR UN PROBLEMA DESDE TODOS LOS ANGULOS.",
            "cita": '"LA MENTE QUE PREGUNTA ES LA QUE LLEGA MAS LEJOS."'
        },
        "Hufflepuff": {
            "subtitulo": "LEALTAD · TEJON",
            "descripcion": "PRIORIZA AL GRUPO POR ENCIMA DE SI MISMO; ES CONSTANTE Y DE CONFIANZA.",
            "cita": '"NINGUN LOGRO VALE LA PENA SI SE DEJA A ALGUIEN ATRAS."'
        }
    }
    
    # Crear 2 filas de 2 columnas para imagenes mas grandes
    for i in range(0, 4, 2):
        cols_casas = st.columns(2)
        for j, casa in enumerate(logica.CASAS[i:i+2]):
            with cols_casas[j]:
                color = logica.COLORES.get(casa, "#c9a227")
                img_path = casa_img_map.get(casa, "")
                desc = casa_descripciones.get(casa, {})
                
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
                    st.markdown(
                        f"<div style='text-align:center; font-size:60px;'>🏰</div>",
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

# ===========================================================================
# TAB 2: ESTADISTICAS
# ===========================================================================
with tab_stats:
    st.markdown("<h2>Estadistica basica</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Puntaje por casa (variables numericas)")
        columnas_pct = [f"{c}_pct" for c in logica.CASAS]
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
    st.markdown("<h2>El Sombrero aprende (K-Means + PCA)</h2>", unsafe_allow_html=True)
    st.caption(
        "Se ajustan, en este orden: StandardScaler → KMeans(k) → PCA(2), "
        "sobre los porcentajes por casa de los aspirantes filtrados."
    )

    X = logica.preparar_features(puntajes)

    with st.expander("Metodo del codo (ayuda a justificar el numero de clusters, k)"):
        from sklearn.preprocessing import StandardScaler

        inercias = logica.calcular_inercias(StandardScaler().fit_transform(X))
        fig_codo = px.line(inercias, x="k", y="Inercia", markers=True, title="Inercia vs. numero de clusters (k)")
        fig_codo.update_traces(line_color="#c9a227")
        fig_codo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_codo, use_container_width=True)

    k = st.slider("Numero de clusters (k) — casas a formar", min_value=2, max_value=6, value=4)

    if "modelo_entrenado" not in st.session_state:
        st.session_state.modelo_entrenado = False



    if st.button("Susurrarle al Sombrero: entrenar modelo"):
        if k > len(df_filtrado):
            st.markdown(
                estilo.advertencia_html(
                    titulo="🎩 El Sombrero no puede continuar...",
                    mensaje=(
                        f"Tienes solo <b>{len(df_filtrado)}</b> aspirante(s) frente a él, pero le pides "
                        f"repartirlos en <b>{k}</b> casas. Ni con toda la magia de Hogwarts puede formar "
                        "mas casas que aspirantes hay disponibles.<br><br>"
                        "Reduce el numero de clusters (k) o amplia tu seleccion de aspirantes con los "
                        "filtros de la barra lateral, e intenta de nuevo."
                    ),
                ),
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Dificil… muy dificil… el Sombrero esta pensando…"):
                time.sleep(1.2)
                escalador, kmeans, pca, etiquetas, coords_pca = logica.entrenar_kmeans(X, k)
            st.session_state.modelo_entrenado = True
            st.session_state.k_entrenado = k
            st.session_state.escalador = escalador
            st.session_state.kmeans = kmeans
            st.session_state.pca = pca
            st.session_state.etiquetas = etiquetas
            st.session_state.coords_pca = coords_pca
            st.session_state.puntajes = puntajes
            st.session_state.df_filtrado = df_filtrado
            st.success(f"El Sombrero termino de ordenar a {len(df_filtrado)} aspirantes en {k} casas.")   

    # if st.button("Susurrarle al Sombrero: entrenar modelo"):
    #    with st.spinner("Dificil… muy dificil… el Sombrero esta pensando…"):
    #        time.sleep(1.2)
    #        escalador, kmeans, pca, etiquetas, coords_pca = logica.entrenar_kmeans(X, k)
    #    st.session_state.modelo_entrenado = True
    #    st.session_state.k_entrenado = k
    #    st.session_state.escalador = escalador
    #    st.session_state.kmeans = kmeans
    #    st.session_state.pca = pca
    #    st.session_state.etiquetas = etiquetas
    #    st.session_state.coords_pca = coords_pca
    #   st.session_state.puntajes = puntajes
    #    st.session_state.df_filtrado = df_filtrado
    #    st.success(f"El Sombrero termino de ordenar a {len(df_filtrado)} aspirantes en {k} casas.")

    #if not st.session_state.modelo_entrenado:
    #    st.info("Presiona el boton para entrenar el modelo y habilitar las pestañas de Resultados y Descargas.")


# ===========================================================================
# TAB 4: RESULTADOS
# ===========================================================================
with tab_resultados:
    if st.session_state.get("modelo_entrenado"):
        st.markdown("<h2>Resultados de la ceremonia de seleccion</h2>", unsafe_allow_html=True)

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

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.subheader("Dispersion de clusters (PCA, 2D)")
            fig_pca = px.scatter(
                puntajes_r,
                x="PCA_1",
                y="PCA_2",
                color="Cluster",
                hover_data=["Casa_dominante"] + [f"{c}_pct" for c in logica.CASAS],
                title="Reduccion de dimensionalidad (PCA) coloreada por cluster",
            )
            fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_r2:
            st.subheader("Casa dominante vs. cluster asignado")
            cruce = pd.crosstab(puntajes_r["Casa_dominante"], puntajes_r["Cluster"])
            st.dataframe(cruce, use_container_width=True)
            st.caption(
                "Si cada cluster concentra mayormente una casa dominante, el algoritmo "
                "esta agrupando correctamente perfiles similares sin haber usado esa "
                "etiqueta durante el entrenamiento (aprendizaje no supervisado)."
            )

        st.markdown('<hr class="hp-divisor">', unsafe_allow_html=True)
        st.subheader("Perfil promedio de cada cluster")
        perfil_cluster = puntajes_r.groupby("Cluster")[[f"{c}_pct" for c in logica.CASAS]].mean().round(1)
        st.dataframe(perfil_cluster, use_container_width=True)

        fig_radar = go.Figure()
        for cluster_id, fila in perfil_cluster.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=fila.values.tolist() + [fila.values[0]],
                    theta=logica.CASAS + [logica.CASAS[0]],
                    fill="toself",
                    name=f"Cluster {cluster_id}",
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Perfil promedio por cluster (radar)",
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

    # Mapa de imagenes de casas
    casa_img_map = {
        "Gryffindor": "gryffindor.png",
        "Hufflepuff": "hufflepuff.png",
        "Ravenclaw": "ravenclaw.png",
        "Slytherin": "slytherin.png"
    }

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
                "<p style='text-align:center; font-size:80px;'>🎩</p>",
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
            img_path = casa_img_map.get(casa_final, "")
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
                        <div style='font-size:100px;'>🏰</div>
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
        
        cols_puntajes = st.columns(4)
        for col, casa in zip(cols_puntajes, logica.CASAS):
            with col:
                img_path = casa_img_map.get(casa, "")
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
                    st.markdown(
                        f"<div style='text-align:center; font-size:40px;'>🏰</div>",
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
            x=logica.CASAS,
            y=[fila_puntajes[f"{c}_pct"] for c in logica.CASAS],
            color=logica.CASAS,
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
            st.info(f"Segun el modelo K-Means entrenado, este aspirante quedo en el **cluster {cluster_asignado}**.")
        else:
            st.caption("Entrena el modelo en la pestaña 'Entrenamiento' para ver tambien su cluster asignado.")

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
        buffer = io.BytesIO()
        joblib.dump(
            {
                "escalador": st.session_state.escalador,
                "kmeans": st.session_state.kmeans,
                "pca": st.session_state.pca,
                "k": st.session_state.k_entrenado,
                "features": [f"{c}_pct" for c in logica.CASAS],
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
    else:
        st.info("Entrena el modelo en la pestaña 'Entrenamiento' para habilitar la descarga de resultados y del modelo (.pkl).")