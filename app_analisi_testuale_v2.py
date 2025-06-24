import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import networkx as nx
from itertools import tee
import seaborn as sns

st.set_page_config(page_title="Analisi Testuale", layout="wide")
st.title("🧠 Dashboard di Analisi Testuale")

uploaded_file = st.file_uploader("📤 Carica il file Excel con le risposte elaborate", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.sidebar.header("🔎 Filtri")
    domande = [c for c in df.columns if c.startswith("NORM_")]
    col_domanda = st.sidebar.selectbox("Domanda (testo)", domande)
    col_pos = "POS_" + col_domanda
    col_lemmi = "LEMMI_" + col_domanda

    paesi = df["Paese"].dropna().unique().tolist()
    paese_sel = st.sidebar.multiselect("Paesi", paesi, default=paesi)

    ruoli = df["Ruolo"].dropna().unique().tolist()
    ruolo_sel = st.sidebar.multiselect("Ruoli", ruoli, default=ruoli)

    df_filt = df[
        (df["Paese"].isin(paese_sel)) &
        (df["Ruolo"].isin(ruolo_sel)) &
        (df[col_domanda].notna())
    ]

    st.subheader("📄 Risposte filtrate")
    st.write(df_filt[[col_domanda]].head(5))

    # Word Cloud
    st.subheader("☁️ Nuvola di parole")
    corpus = " ".join(df_filt[col_domanda].astype(str))
    word_freq = {}
    for w in corpus.split():
        word_freq[w] = word_freq.get(w, 0) + 1
    wc = WordCloud(width=1000, height=500, background_color="white", colormap="Dark2").generate_from_frequencies(word_freq)
    fig_wc, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

    # Frequenze POS
    st.subheader("🔤 Frequenze per categoria grammaticale (POS)")
    all_pos = df_filt[col_pos].dropna().apply(lambda x: ast.literal_eval(x)).explode()
    pos_counts = all_pos.value_counts().sort_values(ascending=False)
    fig_pos, ax_pos = plt.subplots()
    sns.barplot(x=pos_counts.values, y=pos_counts.index, ax=ax_pos)
    ax_pos.set_title("Frequenze POS")
    ax_pos.set_xlabel("Frequenza")
    st.pyplot(fig_pos)

    # Co-occorrenze / rete semantica
    st.subheader("🌐 Rete di co-occorrenze (bigrammi di lemmi)")

    # Estrai bigrammi
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    edges = []
    for lista in df_filt[col_lemmi].dropna():
        try:
            lemmi = ast.literal_eval(lista)
            edges.extend(pairwise(lemmi))
        except:
            continue

    bigrammi = pd.Series(edges).value_counts().head(30)
    G = nx.Graph()
    for (n1, n2), weight in bigrammi.items():
        G.add_edge(n1, n2, weight=weight)

    fig_net, ax_net = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, width=[d["weight"]/2 for _,_,d in G.edges(data=True)])
    st.pyplot(fig_net)

    # Tabella parole
    st.subheader("📊 Tabella frequenze parole")
    freq_df = pd.DataFrame(word_freq.items(), columns=["Lemma", "Frequenza"]).sort_values(by="Frequenza", ascending=False)
    st.dataframe(freq_df.head(30))
