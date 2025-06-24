
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

st.set_page_config(page_title="Analisi Testuale per Paese e Scuola", layout="wide")
st.title("🧠 Analisi Testuale Interattiva")

uploaded_file = st.file_uploader("📤 Carica il file Excel con i dati NLP", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Filtri
    st.sidebar.header("🔎 Filtri")
    paesi = df["Paese"].dropna().unique().tolist()
    istituti = df["Nome istituzione/rappresentanza"].dropna().unique().tolist()
    domande = [c for c in df.columns if c.startswith("LEMMI_")]

    paese_sel = st.sidebar.multiselect("Paese", paesi, default=paesi)
    ist_sel = st.sidebar.multiselect("Scuola/Istituto", istituti, default=istituti)
    domanda_sel = st.sidebar.selectbox("Domanda", domande)

    df_filt = df[
        (df["Paese"].isin(paese_sel)) &
        (df["Nome istituzione/rappresentanza"].isin(ist_sel)) &
        (df[domanda_sel].notna())
    ]

    st.markdown(f"### 📄 Risposte filtrate – {domanda_sel.replace('LEMMI_', '')}")
    st.write(df_filt[[domanda_sel]].head())

    # Frequenze parole
    st.markdown("### 📊 Frequenze parole")
    corpus = df_filt[domanda_sel].dropna().apply(lambda x: ast.literal_eval(x))
    all_words = corpus.explode()
    freq_df = all_words.value_counts().reset_index()
    freq_df.columns = ["Lemma", "Frequenza"]
    st.dataframe(freq_df.head(30))

    # TF-IDF
    st.markdown("### 🧪 TF-IDF parole")
    docs = df_filt[domanda_sel].dropna().apply(lambda x: " ".join(ast.literal_eval(x)))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    sums = X.sum(axis=0)
    tfidf_scores = [(feature_names[i], sums[0, i]) for i in range(len(feature_names))]
    tfidf_df = pd.DataFrame(tfidf_scores, columns=["Lemma", "TF-IDF"]).sort_values(by="TF-IDF", ascending=False)
    st.dataframe(tfidf_df.head(30))

    # Word cloud
    st.markdown("### ☁️ Nuvola di parole")
    word_freq = dict(freq_df.head(100).values)
    wc = WordCloud(width=1000, height=500, background_color="white", colormap="Set2").generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
