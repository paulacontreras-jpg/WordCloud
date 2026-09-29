"""
🔎 WordScope — Laboratorio de Palabras
Aplicación Streamlit para analizar y visualizar la frecuencia de palabras.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import io
from collections import Counter
from wordcloud import WordCloud, STOPWORDS


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="WordScope",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(56,189,248,0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(20,184,166,0.07), transparent 25%),
        #F4F9FA;
}


/* ─────────────────────────────
   SIDEBAR
   ───────────────────────────── */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063B4C 0%, #07566A 55%, #087F8C 100%) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stSidebar"] label {
    color: #DDF7FA !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] p {
    color: #D2EFF2 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18) !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #E7FAFC !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input {
    background: #FFFFFF !important;
    color: #12313A !important;
    border: none !important;
    border-radius: 10px !important;
}


/* ─────────────────────────────
   TÍTULOS
   ───────────────────────────── */

h1 {
    color: #063B4C !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
}

h2, h3 {
    color: #07566A !important;
    font-weight: 700 !important;
}

p, li {
    color: #38545B !important;
    line-height: 1.65 !important;
}


/* ─────────────────────────────
   HEADER
   ───────────────────────────── */

.header-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #E8F8FA 100%);
    border: 1px solid #BFE6EA;
    border-left: 7px solid #08A6A6;
    border-radius: 18px;
    padding: 30px 36px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(6,59,76,0.08);
}

.header-card h1 {
    color: #063B4C !important;
}

.header-card p {
    color: #52727A !important;
}


/* ─────────────────────────────
   CARDS
   ───────────────────────────── */

.section-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid #D7EBEE;
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 18px;
    box-shadow: 0 5px 20px rgba(6,59,76,0.06);
}


/* ─────────────────────────────
   INFO ITEMS
   ───────────────────────────── */

.info-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 17px;
    background: #F2FAFB;
    border: 1px solid #D8EEF0;
    border-radius: 12px;
    margin-bottom: 9px;
}

.info-item:hover {
    background: #E6F7F8;
}


/* ─────────────────────────────
   TAGS
   ───────────────────────────── */

.uso-tag {
    display: inline-block;
    background: #E5F7F7;
    border: 1px solid #B8E5E6;
    border-radius: 30px;
    padding: 7px 15px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #087F8C;
    margin: 4px 3px;
}


/* ─────────────────────────────
   BOTONES
   ───────────────────────────── */

.stButton > button {
    background: linear-gradient(135deg, #087F8C, #08A6A6) !important;
    color: white !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 5px 15px rgba(8,166,166,0.25);
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(8,166,166,0.35);
}

[data-testid="stDownloadButton"] button {
    background: #063B4C !important;
    color: white !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #087F8C !important;
}


/* ─────────────────────────────
   MÉTRICAS
   ───────────────────────────── */

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #D7EBEE;
    border-top: 4px solid #08A6A6;
    border-radius: 15px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(6,59,76,0.06);
}

[data-testid="metric-container"] label {
    color: #648087 !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #063B4C !important;
    font-weight: 800 !important;
}


/* ─────────────────────────────
   NUBE
   ───────────────────────────── */

.wc-container {
    background: #FFFFFF;
    border: 1px solid #D7EBEE;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 7px 25px rgba(6,59,76,0.07);
    margin-bottom: 18px;
}


/* ─────────────────────────────
   FRECUENCIA
   ───────────────────────────── */

.freq-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 14px;
    margin: 5px 0;
    background: #FFFFFF;
    border: 1px solid #E0EFF1;
    border-radius: 10px;
    transition: all 0.2s ease;
}

.freq-row:hover {
    background: #EFFBFC;
    transform: translateX(3px);
}

.freq-bar {
    height: 9px;
    background: linear-gradient(90deg, #087F8C, #22C1C3);
    border-radius: 8px;
    display: inline-block;
}

.rank-tag {
    background: #E5F7F7;
    border: 1px solid #C2E9EA;
    border-radius: 7px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #087F8C;
    font-family: 'Space Mono', monospace;
    min-width: 38px;
    text-align: center;
}


/* ─────────────────────────────
   EXPANDER
   ───────────────────────────── */

div[data-testid="stExpander"] {
    border: 1px solid #D7EBEE !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
}


/* ─────────────────────────────
   TABLA
   ───────────────────────────── */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}


/* ─────────────────────────────
   DIVISORES
   ───────────────────────────── */

hr {
    border-color: #D8EBEE !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────

STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con","no","una","su",
    "para","es","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","nos","durante","ni","contra","ese","eso","ante","bajo","tras",
    "que","fue","son","han","ha","ser","era","estan","siendo","sido","he","has","hemos",
    "habian","tiene","tienen","hacer","puede","pueden","asi","tan","parte","todo","todos",
    "todas","cada","otro","otra","otros","otras","mismo","misma","nuestro","nuestra",
    "ellos","ellas","nosotros","les","esa","esos","esas","aquel","aquella","aquellos",
}


def obtener_stopwords(idioma):
    sw = set(STOPWORDS)

    if idioma in ("Español", "Ambos"):
        sw |= STOPWORDS_ES

    return sw


# ─────────────────────────────────────────────
# PALETAS
# ─────────────────────────────────────────────

PALETAS = {
    "Océano": [
        "#063B4C", "#07566A", "#087F8C",
        "#08A6A6", "#22C1C3", "#67DDE0", "#A7EEF0"
    ],

    "Azul eléctrico": [
        "#172554", "#1E3A8A", "#1D4ED8",
        "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD"
    ],

    "Verde menta": [
        "#064E3B", "#047857", "#059669",
        "#10B981", "#34D399", "#6EE7B7", "#A7F3D0"
    ],

    "Atardecer": [
        "#7C2D12", "#C2410C", "#EA580C",
        "#F97316", "#FB923C", "#FDBA74", "#FED7AA"
    ],

    "Índigo": [
        "#1E1B4B", "#312E81", "#3730A3",
        "#4338CA", "#4F46E5", "#6366F1", "#818CF8"
    ],

    "Monocromático": [
        "#0F172A", "#1E293B", "#334155",
        "#475569", "#64748B", "#94A3B8", "#CBD5E1"
    ],
}


FORMAS = {
    "Rectángulo": None,
    "Círculo": "circle",
}


def crear_mascara(forma, size=500):

    if forma == "circle":

        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2

        mascara = np.ones(
            (size, size),
            dtype=np.uint8
        ) * 255

        mascara[
            (x - cx)**2 + (y - cy)**2
            <= (size // 2 - 12)**2
        ] = 0

        return mascara

    return None


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def limpiar_texto(texto, stopwords, min_longitud):

    texto = texto.lower()

    texto = re.sub(
        r"http\S+|www\S+",
        "",
        texto
    )

    texto = re.sub(
        r"[^a-záéíóúüñàâèêîôùûäëïöü\s]",
        " ",
        texto,
        flags=re.UNICODE
    )

    palabras = [
        p for p in texto.split()
        if p not in stopwords
        and len(p) >= min_longitud
    ]

    return " ".join(palabras)


def contar_palabras(texto_limpio):

    return pd.DataFrame(
        Counter(
            texto_limpio.split()
        ).most_common(50),
        columns=["Palabra", "Frecuencia"]
    )


def generar_wordcloud(
    texto_limpio,
    paleta_nombre,
    max_words,
    fondo,
    forma,
    ancho=1000,
    alto=520
):

    import random

    colores = PALETAS[paleta_nombre]

    def color_func(
        word,
        font_size,
        position,
        orientation,
        random_state=None,
        **kwargs
    ):

        rng = random_state or random.Random()

        return colores[
            rng.randint(
                0,
                len(colores) - 1
            )
        ]

    mascara = crear_mascara(
        forma,
        size=min(ancho, alto)
    )

    wc = WordCloud(
        width=ancho,
        height=alto,
        max_words=max_words,
        background_color=fondo,
        color_func=color_func,
        mask=mascara,
        collocations=False,
        min_font_size=11,
        max_font_size=120,
        prefer_horizontal=0.75,
        relative_scaling=0.5,
        margin=5,
    ).generate(texto_limpio)

    fig, ax = plt.subplots(
        figsize=(
            ancho / 100,
            alto / 100
        )
    )

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    fig.patch.set_facecolor(fondo)

    plt.tight_layout(pad=0)

    return fig


def fig_a_bytes(fig):

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    buf.seek(0)

    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:

    st.markdown("## 🔎 WordScope")
    st.caption("Laboratorio de análisis textual")
    st.divider()

    st.markdown("### ✍️ FUENTE DE TEXTO")

    fuente = st.radio(
        "fuente",
        [
            "✍️ Escribir / Pegar",
            "📂 Subir archivo"
        ],
        label_visibility="collapsed"
    )

    texto_input = ""

    if fuente == "✍️ Escribir / Pegar":

        texto_input = st.text_area(
            "Texto:",
            height=190,
            placeholder=(
                "Pega aquí un artículo, "
                "reseña, discurso, encuesta..."
            )
        )

        with st.expander("🧪 Probar con un ejemplo"):

            ejemplos = {

                "Inteligencia Artificial": """
                La inteligencia artificial es una disciplina de la informática orientada a desarrollar
                sistemas capaces de ejecutar tareas que requieren capacidades cognitivas humanas.
                El aprendizaje automático, las redes neuronales profundas y el procesamiento del
                lenguaje natural constituyen los pilares técnicos de los sistemas modernos de
                inteligencia artificial. Los modelos de lenguaje de gran escala, la visión
                computacional y la robótica autónoma representan aplicaciones de vanguardia.
                La inteligencia artificial transforma sectores como la salud, la educación,
                la manufactura, las finanzas y el transporte, generando eficiencias significativas.
                """,

                "Colombia": """
                Colombia es una nación situada en el extremo noroccidental de América del Sur,
                reconocida por su excepcional biodiversidad, riqueza cultural y diversidad de paisajes.
                Bogotá es la capital y principal centro económico, seguida de Medellín, Cali y
                Barranquilla como ciudades de relevancia nacional. El café colombiano goza de
                reconocimiento internacional por su calidad y perfil aromático. La floricultura
                colombiana abastece mercados globales con alta competitividad. El país alberga
                ecosistemas del Amazonas, los Andes, el Caribe y el Pacífico, constituyéndose
                como uno de los territorios con mayor biodiversidad del planeta.
                """,

                "Tecnología 4.0": """
                La cuarta revolución industrial redefine los modelos productivos mediante la
                convergencia de tecnologías digitales avanzadas. El Internet de las cosas,
                la inteligencia artificial, el análisis de grandes datos, la robótica colaborativa
                y la automatización inteligente son pilares estratégicos de la industria moderna.
                Las fábricas inteligentes integran sensores, conectividad y analítica para
                optimizar procesos en tiempo real. La manufactura aditiva, los gemelos digitales
                y la realidad aumentada transforman la ingeniería de producción. La computación
                en la nube y la ciberseguridad son habilitadores fundamentales de la
                transformación digital empresarial.
                """
            }

            ejemplo_sel = st.selectbox(
                "Ejemplo:",
                list(ejemplos.keys()),
                label_visibility="collapsed"
            )

            if st.button("Cargar texto seleccionado"):

                st.session_state["texto_ejemplo"] = \
                    ejemplos[ejemplo_sel]

                st.rerun()

        if (
            "texto_ejemplo" in st.session_state
            and not texto_input
        ):

            texto_input = \
                st.session_state["texto_ejemplo"]

    else:

        archivo = st.file_uploader(
            "Archivo:",
            type=["txt", "csv"],
            label_visibility="collapsed"
        )

        if archivo:

            if archivo.name.endswith(".txt"):

                texto_input = archivo.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            elif archivo.name.endswith(".csv"):

                df_csv = pd.read_csv(archivo)

                col_txt = st.selectbox(
                    "Columna de texto:",
                    df_csv.columns.tolist()
                )

                texto_input = " ".join(
                    df_csv[col_txt]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

            st.success(
                f"Archivo cargado — "
                f"{len(texto_input):,} caracteres"
            )

    st.divider()

    st.markdown("### 🧹 PROCESAMIENTO")

    idioma = st.selectbox(
        "Stopwords:",
        [
            "Español",
            "Inglés",
            "Ambos",
            "Ninguno"
        ]
    )

    min_longitud = st.slider(
        "Longitud mínima de palabra",
        2,
        8,
        3
    )

    palabras_extra = st.text_input(
        "Excluir palabras adicionales:",
        placeholder="ej: también, así, aquí"
    )

    st.divider()

    st.markdown("### 🎨 APARIENCIA")

    paleta_sel = st.selectbox(
        "Paleta:",
        list(PALETAS.keys())
    )

    fondo_sel = st.radio(
        "Fondo:",
        [
            "Blanco",
            "Negro"
        ],
        horizontal=True
    )

    fondo_color = (
        "white"
        if fondo_sel == "Blanco"
        else "black"
    )

    forma_sel = st.selectbox(
        "Forma:",
        list(FORMAS.keys())
    )

    max_words = st.slider(
        "Máximo de palabras:",
        20,
        200,
        80
    )

    st.divider()

    generar = st.button(
        "🔎 GENERAR NUBE",
        use_container_width=True
    )


# ─────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────

st.markdown("""
<div class="header-card">

    <h1 style="margin:0; font-size:2rem;">
        🔎 WordScope
    </h1>

    <p style="margin:7px 0 0 0;">
        Explora un texto, descubre sus palabras principales
        y conviértelas en una visualización.
    </p>

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BIENVENIDA
# ─────────────────────────────────────────────

if not generar or not texto_input.strip():

    col_izq, col_der = st.columns(
        [3, 2],
        gap="large"
    )

    with col_izq:

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.markdown("### 🧠 ¿Qué puedes descubrir aquí?")

        st.markdown("""
        Una **nube de palabras** transforma un texto en una
        representación visual de sus términos más frecuentes.

        Cuanto más aparece una palabra, mayor será su tamaño
        dentro de la nube.
        """)

        for icono, titulo, desc in [

            (
                "📊",
                "Frecuencia",
                "Detecta las palabras que aparecen con mayor frecuencia."
            ),

            (
                "🧹",
                "Limpieza",
                "Filtra palabras comunes que no aportan información relevante."
            ),

            (
                "🎨",
                "Visualización",
                "Personaliza colores, forma, fondo y cantidad de palabras."
            ),

            (
                "📥",
                "Exportación",
                "Descarga tu nube como imagen y los datos como CSV."
            ),

        ]:

            st.markdown(
                f"""
                <div class="info-item">

                    <span style="font-size:1.35rem;">
                        {icono}
                    </span>

                    <div>
                        <strong style="color:#063B4C;">
                            {titulo}
                        </strong>

                        <p style="
                            margin:2px 0 0 0;
                            color:#60777D !important;
                            font-size:0.88rem;
                        ">
                            {desc}
                        </p>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### 🧭 Cómo utilizarlo")

        pasos = [
            "Escribe, pega o sube un texto.",
            "Configura los filtros y la apariencia.",
            "Genera tu nube de palabras.",
            "Explora las frecuencias y descarga los resultados."
        ]

        for i, paso in enumerate(pasos, 1):

            st.markdown(
                f"**{i}.** {paso}"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col_der:

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.markdown("### 🧪 ¿Para qué sirve?")

        casos = [
            "📰 Noticias",
            "📋 Encuestas",
            "💬 Reseñas",
            "🎓 Textos académicos",
            "📚 Literatura",
            "📊 Investigación",
            "💻 Tecnología"
        ]

        for caso in casos:

            st.markdown(
                f'<span class="uso-tag">{caso}</span>',
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.markdown("### 🌊 Paleta recomendada")

        st.markdown("""
        <p>
        La interfaz utiliza una estética inspirada en un
        <strong>laboratorio digital</strong>: tonos azul petróleo,
        turquesa y blanco para representar análisis,
        información y exploración.
        </p>

        <div style="
            display:flex;
            gap:8px;
            margin-top:15px;
        ">

            <div style="
                width:45px;
                height:45px;
                border-radius:10px;
                background:#063B4C;
            "></div>

            <div style="
                width:45px;
                height:45px;
                border-radius:10px;
                background:#087F8C;
            "></div>

            <div style="
                width:45px;
                height:45px;
                border-radius:10px;
                background:#08A6A6;
            "></div>

            <div style="
                width:45px;
                height:45px;
                border-radius:10px;
                background:#67DDE0;
            "></div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    if not texto_input.strip() and generar:

        st.warning(
            "Escribe o carga un texto antes de generar la nube."
        )

    st.stop()


# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────

stopwords_set = (
    obtener_stopwords(idioma)
    if idioma != "Ninguno"
    else set()
)

if palabras_extra.strip():

    stopwords_set |= {
        p.strip().lower()
        for p in palabras_extra.split(",")
        if p.strip()
    }

texto_limpio = limpiar_texto(
    texto_input,
    stopwords_set,
    min_longitud
)

if not texto_limpio.strip():

    st.error(
        "El texto resultante está vacío. "
        "Reduce la longitud mínima o cambia los filtros."
    )

    st.stop()


df_freq = contar_palabras(
    texto_limpio
)

total_palabras = len(
    texto_limpio.split()
)

vocabulario = len(
    df_freq
)


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Palabras procesadas",
    f"{total_palabras:,}"
)

m2.metric(
    "Vocabulario único",
    f"{vocabulario:,}"
)

m3.metric(
    "Término principal",
    df_freq.iloc[0]["Palabra"]
    if not df_freq.empty
    else "—"
)

m4.metric(
    "Frecuencia máxima",
    int(df_freq.iloc[0]["Frecuencia"])
    if not df_freq.empty
    else 0
)


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# NUBE
# ─────────────────────────────────────────────

with st.spinner("🔎 Analizando palabras..."):

    fig_wc = generar_wordcloud(
        texto_limpio,
        paleta_sel,
        max_words,
        fondo_color,
        FORMAS[forma_sel],
        ancho=1000,
        alto=520,
    )


st.markdown(
    '<div class="wc-container">',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    ### ☁️ Mapa de palabras

    **Paleta:** {paleta_sel}
    &nbsp; · &nbsp;
    **Fondo:** {fondo_sel}
    &nbsp; · &nbsp;
    **Máximo:** {max_words} palabras
    """
)

st.pyplot(
    fig_wc,
    use_container_width=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


img_bytes = fig_a_bytes(
    fig_wc
)

st.download_button(
    "⬇️ Descargar nube PNG",
    data=img_bytes,
    file_name="wordcloud.png",
    mime="image/png",
    use_container_width=True,
)


st.divider()


# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────

col_freq, col_tabla = st.columns(
    [3, 2],
    gap="large"
)


with col_freq:

    st.markdown(
        "### 📈 Frecuencia léxica · Top 20"
    )

    top20 = df_freq.head(20)

    max_freq = top20["Frecuencia"].max()

    for rank, (_, row) in enumerate(
        top20.iterrows(),
        1
    ):

        p = row["Palabra"]

        f = int(
            row["Frecuencia"]
        )

        barra_w = max(
            12,
            int(
                (f / max_freq) * 210
            )
        )

        st.markdown(
            f"""
            <div class="freq-row">

                <span class="rank-tag">
                    #{rank:02d}
                </span>

                <span style="
                    font-weight:600;
                    color:#063B4C;
                    min-width:130px;
                    font-size:0.93rem;
                ">
                    {p}
                </span>

                <div
                    class="freq-bar"
                    style="
                        width:{barra_w}px;
                        opacity:{
                            0.5 + 0.5*(f/max_freq)
                        :.2f};
                    "
                ></div>

                <span style="
                    font-family:'Space Mono',monospace;
                    font-size:0.84rem;
                    color:#38545B;
                    min-width:28px;
                    text-align:right;
                    font-weight:700;
                ">
                    {f}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


with col_tabla:

    st.markdown(
        "### 🗂️ Tabla de frecuencias"
    )

    st.dataframe(
        df_freq.head(30)
        .style
        .background_gradient(
            subset=["Frecuencia"],
            cmap="GnBu"
        )
        .format(
            {
                "Frecuencia": "{:,}"
            }
        ),
        use_container_width=True,
        height=500,
    )

    csv_bytes = (
        df_freq
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Exportar tabla CSV",
        data=csv_bytes,
        file_name="frecuencias.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()


# ─────────────────────────────────────────────
# TEXTO PROCESADO
# ─────────────────────────────────────────────

with st.expander(
    "🔍 Ver texto procesado"
):

    preview = (
        texto_limpio[:2500]
        + (
            "..."
            if len(texto_limpio) > 2500
            else ""
        )
    )

    st.markdown(
        f"""
        <p style="
            font-family:'Space Mono', monospace;
            font-size:0.82rem;
            color:#38545B;
            background:#F2FAFB;
            padding:18px;
            border-radius:12px;
            border:1px solid #D7EBEE;
            line-height:1.8;
        ">
            {preview}
        </p>
        """,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<div style="
    text-align:center;
    padding:30px 0 10px 0;
    color:#6C858C;
    font-size:0.82rem;
">
    🔎 WordScope · Explora los patrones escondidos en tus palabras.
</div>
""", unsafe_allow_html=True)


plt.close("all")
