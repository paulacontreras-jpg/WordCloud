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
    layout="wide"
)

# ─────────────────────────────────────────────
# ESTILO
# ─────────────────────────────────────────────

st.markdown("""
<style>
.stApp {
    background: #F4FAFA;
}

[data-testid="stSidebar"] {
    background: #073B4C !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: white !important;
}

h1 {
    color: #073B4C !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #086F7A !important;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #D8ECEE;
    box-shadow: 0 4px 15px rgba(7,59,76,.06);
    margin-bottom: 18px;
}

.stButton > button {
    background: #08A6A6 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: #087F8C !important;
}

[data-testid="stDownloadButton"] button {
    background: #073B4C !important;
    color: white !important;
    border-radius: 10px !important;
}

[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    border-top: 4px solid #08A6A6;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────

STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","un","por","con",
    "no","una","su","para","es","al","lo","como","mas","pero","sus","le",
    "ya","o","este","si","porque","esta","entre","cuando","muy","sin",
    "sobre","tambien","me","hasta","hay","donde","quien","desde","nos",
    "durante","ni","contra","ese","eso","ante","bajo","tras","que","fue",
    "son","han","ha","ser","era","estan","siendo","sido","he","has",
    "hemos","habian","tiene","tienen","hacer","puede","pueden","asi",
    "tan","parte","todo","todos","todas","cada","otro","otra","otros",
    "otras","mismo","misma","nuestro","nuestra","ellos","ellas",
    "nosotros","les","esa","esos","esas"
}

def obtener_stopwords(idioma):
    sw = set(STOPWORDS)
    if idioma in ["Español", "Ambos"]:
        sw |= STOPWORDS_ES
    return sw


# ─────────────────────────────────────────────
# PALETAS
# ─────────────────────────────────────────────

PALETAS = {
    "Océano": ["#073B4C","#087F8C","#08A6A6","#22C1C3","#67DDE0"],
    "Azul": ["#172554","#1D4ED8","#2563EB","#60A5FA","#93C5FD"],
    "Verde": ["#064E3B","#047857","#059669","#10B981","#6EE7B7"],
    "Atardecer": ["#7C2D12","#C2410C","#EA580C","#F97316","#FDBA74"],
    "Índigo": ["#1E1B4B","#3730A3","#4F46E5","#6366F1","#818CF8"]
}

FORMAS = {
    "Rectángulo": None,
    "Círculo": "circle"
}


def crear_mascara(forma, size=500):
    if forma != "circle":
        return None

    y, x = np.ogrid[:size, :size]
    centro = size // 2

    mascara = np.ones((size, size), dtype=np.uint8) * 255
    mascara[
        (x-centro)**2 + (y-centro)**2 <= (size//2-12)**2
    ] = 0

    return mascara


# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def limpiar_texto(texto, stopwords, minimo):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(
        r"[^a-záéíóúüñàâèêîôùûäëïöü\s]",
        " ",
        texto
    )

    palabras = [
        p for p in texto.split()
        if p not in stopwords and len(p) >= minimo
    ]

    return " ".join(palabras)


def generar_wordcloud(texto, paleta, max_words, fondo, forma):

    colores = PALETAS[paleta]

    def color_func(*args, **kwargs):
        import random
        return random.choice(colores)

    wc = WordCloud(
        width=1000,
        height=520,
        max_words=max_words,
        background_color=fondo,
        color_func=color_func,
        mask=crear_mascara(forma),
        collocations=False,
        min_font_size=11
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(fondo)
    plt.tight_layout(pad=0)

    return fig


def fig_a_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    buffer.seek(0)
    return buffer.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:

    st.markdown("## 🔎 WordScope")
    st.caption("Laboratorio de palabras")
    st.divider()

    st.markdown("### Texto")

    fuente = st.radio(
        "Fuente",
        ["✍️ Escribir / pegar", "📂 Subir archivo"],
        label_visibility="collapsed"
    )

    texto_input = ""

    if fuente == "✍️ Escribir / pegar":

        texto_input = st.text_area(
            "Texto",
            height=180,
            placeholder="Pega aquí tu texto..."
        )

        ejemplos = {
            "Inteligencia Artificial":
            "La inteligencia artificial transforma la educación, la salud, la tecnología y la industria mediante sistemas capaces de analizar información y aprender de los datos.",

            "Colombia":
            "Colombia es un país reconocido por su biodiversidad, cultura, café, paisajes y diversidad de regiones como los Andes, el Caribe, el Pacífico y el Amazonas."
        }

        ejemplo = st.selectbox(
            "Ejemplo",
            ["Ninguno"] + list(ejemplos.keys())
        )

        if ejemplo != "Ninguno":
            texto_input = ejemplos[ejemplo]

    else:

        archivo = st.file_uploader(
            "Subir archivo",
            type=["txt", "csv"]
        )

        if archivo:

            if archivo.name.endswith(".txt"):
                texto_input = archivo.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            else:
                df = pd.read_csv(archivo)
                columna = st.selectbox(
                    "Columna de texto",
                    df.columns
                )
                texto_input = " ".join(
                    df[columna].dropna().astype(str)
                )

    st.divider()

    st.markdown("### Filtros")

    idioma = st.selectbox(
        "Stopwords",
        ["Español", "Inglés", "Ambos", "Ninguno"]
    )

    minimo = st.slider(
        "Longitud mínima",
        2, 8, 3
    )

    extras = st.text_input(
        "Excluir palabras",
        placeholder="palabra1, palabra2"
    )

    st.divider()

    st.markdown("### Diseño")

    paleta = st.selectbox(
        "Paleta",
        list(PALETAS.keys())
    )

    fondo_sel = st.radio(
        "Fondo",
        ["Blanco", "Negro"],
        horizontal=True
    )

    forma = st.selectbox(
        "Forma",
        list(FORMAS.keys())
    )

    max_words = st.slider(
        "Máximo de palabras",
        20, 200, 80
    )

    generar = st.button(
        "🔎 GENERAR NUBE",
        use_container_width=True
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="card">
    <h1>🔎 WordScope</h1>
    <p>
        Descubre las palabras más importantes de un texto
        mediante una visualización interactiva.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BIENVENIDA
# ─────────────────────────────────────────────

if not generar or not texto_input.strip():

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🧠 ¿Qué hace?</h3>
        <p>
        Analiza la frecuencia de las palabras de un texto
        y las representa visualmente. Las palabras que más
        aparecen tendrán mayor tamaño.
        </p>
        <p>📊 Analiza frecuencias</p>
        <p>🧹 Filtra palabras comunes</p>
        <p>🎨 Personaliza la visualización</p>
        <p>📥 Exporta tus resultados</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🧪 ¿Para qué puedes usarlo?</h3>
        <p>📰 Noticias y artículos</p>
        <p>📋 Encuestas</p>
        <p>💬 Reseñas</p>
        <p>🎓 Textos académicos</p>
        <p>📚 Literatura</p>
        <p>📊 Investigación</p>
        </div>
        """, unsafe_allow_html=True)

    if generar:
        st.warning("Ingresa un texto antes de generar la nube.")

    st.stop()


# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────

stopwords = (
    obtener_stopwords(idioma)
    if idioma != "Ninguno"
    else set()
)

if extras.strip():
    stopwords |= {
        p.strip().lower()
        for p in extras.split(",")
        if p.strip()
    }

texto_limpio = limpiar_texto(
    texto_input,
    stopwords,
    minimo
)

if not texto_limpio:
    st.error("No quedaron palabras para analizar.")
    st.stop()

frecuencias = Counter(
    texto_limpio.split()
)

df_freq = pd.DataFrame(
    frecuencias.most_common(50),
    columns=["Palabra", "Frecuencia"]
)

total = len(texto_limpio.split())
unicas = len(frecuencias)


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

m1.metric("Palabras", f"{total:,}")
m2.metric("Palabras únicas", f"{unicas:,}")
m3.metric("Más frecuente", df_freq.iloc[0]["Palabra"])
m4.metric("Frecuencia", int(df_freq.iloc[0]["Frecuencia"]))


# ─────────────────────────────────────────────
# NUBE
# ─────────────────────────────────────────────

fondo = "white" if fondo_sel == "Blanco" else "black"

with st.spinner("Generando nube..."):

    fig = generar_wordcloud(
        texto_limpio,
        paleta,
        max_words,
        fondo,
        FORMAS[forma]
    )

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown("### ☁️ Nube de palabras")

st.pyplot(
    fig,
    use_container_width=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


st.download_button(
    "⬇️ Descargar PNG",
    data=fig_a_bytes(fig),
    file_name="wordcloud.png",
    mime="image/png",
    use_container_width=True
)


# ─────────────────────────────────────────────
# FRECUENCIAS
# ─────────────────────────────────────────────

st.markdown("### 📊 Palabras más frecuentes")

col1, col2 = st.columns([3, 2])

with col1:

    top20 = df_freq.head(20)
    max_freq = top20["Frecuencia"].max()

    for i, row in top20.iterrows():

        porcentaje = row["Frecuencia"] / max_freq

        st.progress(
            porcentaje,
            text=f"{i+1}. {row['Palabra']}  ·  {row['Frecuencia']}"
        )

with col2:

    st.dataframe(
        df_freq.head(30),
        use_container_width=True,
        height=500
    )

    st.download_button(
        "⬇️ Descargar CSV",
        data=df_freq.to_csv(index=False).encode("utf-8"),
        file_name="frecuencias.csv",
        mime="text/csv",
        use_container_width=True
    )


# ─────────────────────────────────────────────
# TEXTO PROCESADO
# ─────────────────────────────────────────────

with st.expander("🔍 Ver texto procesado"):

    st.write(
        texto_limpio[:2500]
        + ("..." if len(texto_limpio) > 2500 else "")
    )

plt.close("all")
