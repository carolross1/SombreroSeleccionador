# -*- coding: utf-8 -*-
"""
estilo.py
---------
CSS + fragmentos HTML/SVG con temática de Hogwarts para la app de Streamlit.
Todo el estilo vive aquí para no ensuciar app.py con HTML/CSS mezclado con
la lógica de la interfaz.

Incluye:
    - CSS global (fondo tipo Gran Comedor, vitrales, pestañas, sidebar,
      inputs, tablas, botones).
    - Sombrero Seleccionador y velas flotantes dibujados en SVG/CSS puro
      (no se usa ninguna imagen ni asset con derechos de autor).
    - Escudos de casa en SVG, generados a partir de los colores definidos
      en logica.py.
    - Helpers que arman los bloques HTML reutilizados en app.py.
"""

import logica

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cinzel:wght@400;600&family=MedievalSharp&display=swap');

:root {
    --pergamino: #f4ecd8;
    --pergamino-oscuro: #e8dcc0;
    --tinta: #2b2117;
    --dorado: #c9a227;
    --dorado-claro: #e8c96a;
}

/* ---------------------------------------------------------------- */
/* Fondo general                                                     */
/* ---------------------------------------------------------------- */
.stApp {
    background: radial-gradient(circle at top left, #3a2f22 0%, #16110c 55%, #0b0906 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c140d 0%, #0f0b07 100%);
    border-right: 2px solid var(--dorado);
}

section[data-testid="stSidebar"] * {
    color: var(--pergamino) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--dorado) !important;
}

h1, h2, h3 {
    font-family: 'Cinzel Decorative', 'Cinzel', serif !important;
    color: var(--dorado) !important;
    text-shadow: 0 0 6px rgba(201, 162, 39, 0.35);
}

p, span, label, div, li {
    font-family: 'Cinzel', serif;
}

.hp-subtitulo {
    text-align: center;
    color: #cbb994;
    font-family: 'MedievalSharp', 'Cinzel', serif;
    font-style: italic;
    margin-top: -6px;
    margin-bottom: 0;
    font-size: 1.05rem;
}

.hp-pergamino {
    background: var(--pergamino);
    color: var(--tinta);
    border: 3px double var(--dorado);
    border-radius: 6px;
    padding: 1.2rem 1.6rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}

.hp-pergamino h1, .hp-pergamino h2, .hp-pergamino h3, .hp-pergamino p {
    color: var(--tinta) !important;
    text-shadow: none;
}

.hp-divisor {
    border: none;
    border-top: 2px solid var(--dorado);
    opacity: 0.55;
    margin: 1.8rem 0;
    position: relative;
}

.hp-divisor::after {
    content: "❖";
    position: absolute;
    top: -12px;
    left: 50%;
    transform: translateX(-50%);
    background: #16110c;
    color: var(--dorado);
    padding: 0 12px;
    font-size: 0.9rem;
}

/* ---------------------------------------------------------------- */
/* Banner / hero: Gran Comedor con vitrales, velas y el Sombrero     */
/* ---------------------------------------------------------------- */
.hp-hero {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    border: 2px solid var(--dorado);
    padding: 2.2rem 1rem 2.6rem 1rem;
    margin-bottom: 1.8rem;
    text-align: center;
    background:
        radial-gradient(circle at 12% 20%, rgba(116,0,1,0.35), transparent 42%),
        radial-gradient(circle at 88% 18%, rgba(14,26,64,0.40), transparent 42%),
        radial-gradient(circle at 22% 88%, rgba(145,99,0,0.30), transparent 45%),
        radial-gradient(circle at 82% 85%, rgba(26,71,42,0.32), transparent 42%),
        linear-gradient(180deg, #241a10 0%, #100b06 100%);
    box-shadow: inset 0 0 60px rgba(0,0,0,0.6), 0 10px 30px rgba(0,0,0,0.5);
}

.hp-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(60deg, rgba(0,0,0,0.35) 0 2px, transparent 2px 58px),
        repeating-linear-gradient(-60deg, rgba(0,0,0,0.35) 0 2px, transparent 2px 58px);
    opacity: 0.45;
    pointer-events: none;
}

.hp-hero-content {
    position: relative;
    z-index: 1;
}

.hp-hero-titulo {
    margin: 0.4rem 0 0.2rem 0 !important;
    font-size: 2.4rem !important;
    letter-spacing: 1px;
}

.hp-candelas {
    position: absolute;
    top: 6px;
    left: 0;
    right: 0;
    height: 40px;
    z-index: 0;
    pointer-events: none;
}

.hp-candela {
    position: absolute;
    top: 0;
    font-size: 1.5rem;
    filter: drop-shadow(0 0 6px #ffb84d);
    animation: hp-flicker 2.6s ease-in-out infinite;
}

@keyframes hp-flicker {
    0%, 100% { opacity: 1; filter: brightness(1) drop-shadow(0 0 6px #ffb84d); }
    45%      { opacity: 0.7; filter: brightness(0.8) drop-shadow(0 0 3px #ffb84d); }
    55%      { opacity: 0.95; filter: brightness(1.15) drop-shadow(0 0 9px #ffb84d); }
}

.hp-hero-hat {
    animation: hp-float 4.2s ease-in-out infinite;
    filter: drop-shadow(0 10px 12px rgba(0,0,0,0.6));
}

@keyframes hp-float {
    0%, 100% { transform: translateY(0px) rotate(-1deg); }
    50%      { transform: translateY(-9px) rotate(1deg); }
}

/* ---------------------------------------------------------------- */
/* Escudos / tarjetas de casa                                       */
/* ---------------------------------------------------------------- */
.hp-tarjeta-casa {
    border-radius: 12px;
    padding: 1.3rem 1.1rem 1.1rem 1.1rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
    border: 2px solid rgba(255,255,255,0.15);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.hp-tarjeta-casa:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.6), 0 0 18px rgba(201,162,39,0.35);
}

.hp-tarjeta-casa .escudo-svg {
    display: flex;
    justify-content: center;
    margin-bottom: 0.3rem;
}

.hp-tarjeta-casa .nombre {
    font-family: 'Cinzel Decorative', serif;
    font-size: 1.25rem;
    margin: 0.2rem 0;
}

.hp-tarjeta-casa .rasgo {
    font-style: italic;
    opacity: 0.9;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
}

.hp-tarjeta-casa .descripcion {
    font-size: 0.82rem;
    opacity: 0.88;
}

.hp-tarjeta-casa .lema {
    font-size: 0.78rem;
    font-family: 'MedievalSharp', serif;
    opacity: 0.75;
    margin-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.25);
    padding-top: 0.4rem;
}

.hp-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    color: #fff;
    font-family: 'Cinzel', serif;
    border: 1px solid rgba(255,255,255,0.3);
}

.hp-veredicto {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 2rem 1rem;
    border-radius: 14px;
    border: 3px solid var(--dorado);
    box-shadow: 0 0 30px rgba(201,162,39,0.35);
}

.hp-veredicto .sombrero {
    font-size: 3rem;
}

.hp-veredicto .titulo {
    font-family: 'Cinzel Decorative', serif;
    font-size: 2rem;
    margin: 0.4rem 0;
    text-shadow: 0 0 10px rgba(0,0,0,0.5);
}

/* ---------------------------------------------------------------- */
/* Métricas, botones, pestañas, inputs                              */
/* ---------------------------------------------------------------- */
div[data-testid="stMetric"] {
    background: rgba(244,236,216,0.06);
    border: 1px solid var(--dorado);
    border-radius: 8px;
    padding: 0.6rem;
}

.stButton>button, .stDownloadButton>button {
    font-family: 'Cinzel', serif;
    background: linear-gradient(180deg, #3a2f22, #1c140d);
    color: var(--dorado);
    border: 1px solid var(--dorado);
    border-radius: 8px;
    transition: all 0.2s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background: var(--dorado);
    color: #1c140d;
    box-shadow: 0 0 14px rgba(201,162,39,0.55);
}

/* Pestañas tipo pergamino */
div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid rgba(201,162,39,0.35);
}

button[data-baseweb="tab"] {
    font-family: 'Cinzel Decorative', serif;
    color: #cbb994 !important;
    background: rgba(244,236,216,0.04);
    border-radius: 8px 8px 0 0 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--dorado) !important;
    background: rgba(201,162,39,0.12);
    border-bottom: 3px solid var(--dorado) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--dorado);
    border-radius: 8px;
    background: rgba(244,236,216,0.03);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--dorado);
    border-radius: 8px;
    overflow: hidden;
}

span[data-baseweb="tag"] {
    background-color: var(--dorado) !important;
    color: #1c140d !important;
    font-family: 'Cinzel', serif;
}

div[data-baseweb="select"] > div {
    border-color: rgba(201,162,39,0.5) !important;
    background-color: rgba(244,236,216,0.04) !important;
}

.stTextInput > div > div {
    border-color: rgba(201,162,39,0.5) !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--dorado) !important;
    border-color: var(--pergamino) !important;
}

.hp-contador {
    text-align: center;
    font-family: 'MedievalSharp', serif;
    color: var(--dorado-claro);
    font-size: 0.95rem;
    margin-top: 0.3rem;
}
</style>
"""


def sombrero_svg(size: int = 130) -> str:
    alto = round(size * 1.15)
    return f"""
    <svg class="hp-hero-hat" width="{size}" height="{alto}" viewBox="0 0 200 220"
         xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="hatgrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#5a4632"/>
                <stop offset="55%" stop-color="#2e2216"/>
                <stop offset="100%" stop-color="#150f08"/>
            </linearGradient>
        </defs>
        <ellipse cx="100" cy="196" rx="92" ry="16" fill="url(#hatgrad)"
                 stroke="#c9a227" stroke-width="2.5"/>
        <path d="M100 12
                 C 128 55, 84 92, 142 118
                 C 118 138, 150 150, 150 150
                 C 108 172, 58 176, 44 162
                 C 30 148, 62 122, 40 96
                 C 22 74, 66 40, 100 12 Z"
              fill="url(#hatgrad)" stroke="#c9a227" stroke-width="2.5"/>
        <path d="M52 152 C 88 168, 128 163, 150 150" fill="none"
              stroke="#c9a227" stroke-width="2" opacity="0.55"/>
        <circle cx="101" cy="17" r="4" fill="#c9a227"/>
    </svg>
    """


def candelas_html() -> str:
    posiciones = [6, 20, 38, 62, 80, 94]
    retrasos = [0, 0.7, 1.3, 0.4, 1.0, 0.2]
    velas = "".join(
        f'<span class="hp-candela" style="left:{pos}%; animation-delay:{delay}s;">🕯️</span>'
        for pos, delay in zip(posiciones, retrasos)
    )
    return f'<div class="hp-candelas">{velas}</div>'


def hero_html(titulo: str = "El Sombrero Seleccionador", subtitulo: str = "") -> str:
    return f"""
    <div class="hp-hero">
        {candelas_html()}
        <div class="hp-hero-content">
            {sombrero_svg()}
            <h1 class="hp-hero-titulo">{titulo}</h1>
            <p class="hp-subtitulo">{subtitulo}</p>
        </div>
    </div>
    """


def bienvenida_html() -> str:
    return """
    <div class="hp-pergamino" style="text-align: center; max-width: 700px; margin: 2rem auto;">
        <h3 style="font-family: 'Cinzel Decorative', serif; color: #2b2117 !important; margin-bottom: 1rem;">
            🎩 Bienvenidos a la Ceremonia de Selección
        </h3>
        <p style="margin-bottom: 1rem; line-height: 1.6;">
            Para dar inicio al análisis de las respuestas del cuestionario y revelar los grupos ocultos 
            mediante algoritmos de aprendizaje no supervisado, por favor carga el pergamino de respuestas 
            (archivo CSV) o activa la casilla de datos de ejemplo en la barra lateral.
        </p>
        <p style="font-style: italic; color: #5a4734;">
            "<i>No es nuestras habilidades lo que muestra quién somos, sino nuestras elecciones.</i>" — Albus Dumbledore
        </p>
    </div>
    """


def tarjeta_casa_html(casa: str) -> str:
    color = logica.COLORES.get(casa, "#333")
    color2 = logica.COLORES_SECUNDARIOS.get(casa, "#111")
    rasgo = logica.RASGO.get(casa, "")
    animal = logica.ANIMAL.get(casa, "")
    descripcion = logica.DESCRIPCION.get(casa, "")
    lema = logica.LEMAS.get(casa, "")
    return f"""
    <div class="hp-tarjeta-casa" style="background: linear-gradient(160deg, {color}, {color2}22 120%), {color};">
        <div class="nombre">{casa}</div>
        <div class="rasgo">{rasgo} · {animal}</div>
        <div class="descripcion">{descripcion}</div>
        <div class="lema">"{lema}"</div>
    </div>
    """


def badge_casa_html(casa: str) -> str:
    color = logica.COLORES.get(casa, "#333")
    emoji = logica.EMOJIS.get(casa, "")
    return f'<span class="hp-badge" style="background:{color};">{emoji} {casa}</span>'


def veredicto_html(casa: str, subtitulo: str = "") -> str:
    color = logica.COLORES.get(casa, "#333")
    emoji = logica.EMOJIS.get(casa, "")
    return f"""
    <div class="hp-veredicto" style="background: radial-gradient(circle at top, {color}, #100c08 85%);">
        <div class="sombrero">🎩</div>
        <div class="titulo" style="color:#f4ecd8;">{emoji} ¡{casa.upper()}! {emoji}</div>
        <div style="color:#e8dcc0; font-style: italic; font-size: 1.1rem;">{subtitulo}</div>
    </div>
    """

def advertencia_html(mensaje: str, titulo: str = "El Sombrero no puede continuar...") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, #2b1810, #1c0f0a);
        border: 3px solid #c9a227;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(201,162,39,0.35), inset 0 0 20px rgba(0,0,0,0.4);
        margin: 1rem 0;
    ">
        <div style="font-size: 2.2rem;">🪄</div>
        <div style="
            font-family: 'Cinzel Decorative', serif;
            color: #e8c96a;
            font-size: 1.3rem;
            margin: 0.4rem 0;
            text-shadow: 0 0 10px rgba(201,162,39,0.5);
        ">{titulo}</div>
        <div style="
            font-family: 'Cinzel', serif;
            color: #f4ecd8;
            font-size: 1rem;
            line-height: 1.6;
        ">{mensaje}</div>
    </div>
    """