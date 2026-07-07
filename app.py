"""
RAG Document Intelligence — Dashboard Streamlit
Stack : LangChain + Mistral AI + ChromaDB

Features :
  - Upload multi-documents (PDF, DOCX, TXT)
  - Q&A intelligent avec sources citées
  - Résumé structuré automatique (JSON → affichage)
  - Comparaison de documents sur une même question
  - Historique de conversation
  - Métriques : chunks, tokens, latence
"""
import streamlit as st
import os
import time
import tempfile
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
.source-card {
    background: rgba(124,106,247,.08);
    border: 1px solid rgba(124,106,247,.25);
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.85rem;
}
.metric-card {
    background: #111; border: 1px solid #222;
    border-radius: 10px; padding: 20px; text-align: center;
}
.tag {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    margin: 2px;
}
.tag-green  { background: rgba(0,200,100,.15); color: #00c864; border: 1px solid rgba(0,200,100,.3); }
.tag-blue   { background: rgba(69,183,209,.15); color: #45b7d1; border: 1px solid rgba(69,183,209,.3); }
.tag-purple { background: rgba(124,106,247,.15); color: #a78bfa; border: 1px solid rgba(124,106,247,.3); }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'pipeline': None,
        'chat_history': [],
        'ingested_files': [],
        'summaries': {},
        'api_key_set': False,
        'total_queries': 0,
        'total_latency': 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline(api_key: str, chunk_size: int, chunk_overlap: int,
                 model: str, top_k: int):
    from core.rag_pipeline import RAGPipeline
    return RAGPipeline(
        api_key=api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model,
        top_k=top_k,
    )


def format_sources(sources: list) -> str:
    html = ""
    for s in sources:
        html += f"""
        <div class="source-card">
            📄 <b>{s['source']}</b> — chunk #{s['chunk_id']}<br/>
            <span style="color:#888;font-size:0.8rem">{s['content']}</span>
        </div>"""
    return html


def plot_chunks_distribution(ingested: list):
    if not ingested:
        return None
    df = pd.DataFrame(ingested)
    fig = px.bar(
        df, x='filename', y='n_chunks',
        color='n_chunks', color_continuous_scale='Viridis',
        labels={'filename': 'Document', 'n_chunks': 'Nombre de chunks'},
        title='Distribution des chunks par document'
    )
    fig.update_layout(
        template='plotly_dark', height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig


def plot_chat_activity(history: list):
    if len(history) < 2:
        return None
    times = [h['timestamp'] for h in history if h['role'] == 'assistant']
    latencies = [h.get('latency', 0) for h in history if h['role'] == 'assistant']
    if not times:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(latencies))), y=latencies,
        mode='lines+markers', fill='tozeroy',
        line=dict(color='#c8ff00', width=2),
        marker=dict(size=6),
        name='Latence (s)'
    ))
    fig.update_layout(
        template='plotly_dark', height=200,
        title='Latence des réponses (secondes)',
        xaxis_title='Question #', yaxis_title='Latence (s)',
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:900;margin-bottom:0'>
        🧠 RAG Document Intelligence
    </h1>
    <p style='color:#888;margin-top:4px'>
        Analyse, interroge et compare tes documents — LangChain · Mistral AI · ChromaDB
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key
        st.markdown("### 🔑 Mistral API Key")
        api_key = st.text_input(
            "Clé API Mistral",
            type="password",
            placeholder="sk-...",
            help="Gratuit sur console.mistral.ai — pas de carte bancaire requise"
        )
        if not api_key:
            st.warning("⚠️ Entrer une clé API Mistral pour commencer.")
            st.markdown("[Obtenir une clé gratuite →](https://console.mistral.ai)")
            st.stop()

        st.markdown("### 🤖 Modèle")
        model = st.selectbox(
            "Modèle Mistral",
            ["mistral-small-latest", "mistral-medium-latest", "open-mistral-7b"],
            help="mistral-small = rapide et gratuit"
        )

        st.markdown("### 📄 Chunking")
        chunk_size = st.slider("Taille des chunks (chars)", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chevauchement", 50, 400, 200, 50)

        st.markdown("### 🔍 Retrieval")
        top_k = st.slider("Chunks récupérés (top-k)", 2, 8, 4)

        st.divider()

        # Upload documents
        st.markdown("### 📂 Documents")
        uploaded_files = st.file_uploader(
            "Charger des documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="PDF, DOCX ou TXT — plusieurs fichiers acceptés"
        )

        if uploaded_files:
            pipeline = get_pipeline(api_key, chunk_size, chunk_overlap, model, top_k)
            for uf in uploaded_files:
                if uf.name not in [d['filename'] for d in st.session_state.ingested_files]:
                    with st.spinner(f"📥 Ingestion de {uf.name}..."):
                        try:
                            # Sauvegarder temporairement
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=Path(uf.name).suffix
                            ) as tmp:
                                tmp.write(uf.getvalue())
                                tmp_path = tmp.name

                            stats = pipeline.ingest(tmp_path, uf.getvalue())
                            os.unlink(tmp_path)

                            st.session_state.ingested_files.append(stats)
                            st.session_state.pipeline = pipeline
                            st.success(f"✅ {uf.name} — {stats['n_chunks']} chunks")
                        except Exception as e:
                            st.error(f"❌ {uf.name} : {e}")

        # Liste des fichiers ingérés
        if st.session_state.ingested_files:
            st.markdown(f"**{len(st.session_state.ingested_files)} document(s) chargé(s)**")
            for d in st.session_state.ingested_files:
                st.markdown(f"- 📄 `{d['filename']}` ({d['n_chunks']} chunks)")

        st.divider()

        # Reset
        if st.button("🗑️ Vider la base vectorielle", use_container_width=True):
            if st.session_state.pipeline:
                st.session_state.pipeline.reset()
            st.session_state.ingested_files = []
            st.session_state.chat_history = []
            st.session_state.summaries = {}
            get_pipeline.clear()
            st.rerun()

        st.markdown("""
        ---
        **Stack :**
        LangChain · Mistral AI · ChromaDB · Streamlit

        **Auteur :**
        [Oumou Kaltoum Sall](https://github.com/KalsoumDS)
        """)

    # ── VÉRIFICATION PIPELINE ─────────────────────────────────────────────────
    if not st.session_state.ingested_files:
        st.info("👈 Charger au moins un document dans la barre latérale pour commencer.")

        # Demo visuelle
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **📄 1. Upload**
            
            Charge tes documents PDF, DOCX ou TXT. 
            Le pipeline les découpe en chunks intelligents.
            """)
        with col2:
            st.markdown("""
            **🔍 2. Analyse**
            
            Résumé automatique, extraction des points clés, 
            détection de la langue et du domaine.
            """)
        with col3:
            st.markdown("""
            **💬 3. Q&A**
            
            Pose des questions en langage naturel. 
            Les sources sont citées avec leurs passages exacts.
            """)
        return

    pipeline = st.session_state.pipeline or get_pipeline(
        api_key, chunk_size, chunk_overlap, model, top_k
    )

    # ── ONGLETS ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Q&A Chat", "📋 Résumés", "⚖️ Comparaison", "📊 Métriques"
    ])

    # ── TAB 1 : Q&A CHAT ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### 💬 Questions & Réponses")
        st.caption(f"Base vectorielle : **{pipeline.total_chunks} chunks** de **{pipeline.n_documents} document(s)**")

        # Afficher l'historique
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role'], avatar='🧠' if msg['role'] == 'assistant' else '👤'):
                st.markdown(msg['content'], unsafe_allow_html=True)
                if msg.get('sources'):
                    with st.expander(f"📎 {len(msg['sources'])} source(s) utilisée(s)"):
                        st.markdown(format_sources(msg['sources']), unsafe_allow_html=True)
                if msg.get('latency'):
                    st.caption(f"⏱️ {msg['latency']:.2f}s · {msg.get('model', model)}")

        # Input
        if question := st.chat_input("Pose une question sur tes documents..."):
            # Afficher la question
            st.session_state.chat_history.append({
                'role': 'user', 'content': question,
                'timestamp': datetime.now().isoformat()
            })
            with st.chat_message('user', avatar='👤'):
                st.markdown(question)

            # Générer la réponse
            with st.chat_message('assistant', avatar='🧠'):
                with st.spinner("🔍 Recherche dans les documents..."):
                    t0 = time.time()
                    try:
                        result = pipeline.query(question)
                        latency = time.time() - t0
                        st.session_state.total_queries += 1
                        st.session_state.total_latency += latency

                        st.markdown(result['answer'])
                        if result.get('sources'):
                            with st.expander(f"📎 {result['n_sources']} source(s) utilisée(s)"):
                                st.markdown(format_sources(result['sources']), unsafe_allow_html=True)
                        st.caption(f"⏱️ {latency:.2f}s · {model} · {result['n_sources']} chunks utilisés")

                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result['answer'],
                            'sources': result.get('sources', []),
                            'latency': latency,
                            'model': model,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")

        # Questions suggérées
        if not st.session_state.chat_history:
            st.markdown("**💡 Questions suggérées :**")
            suggestions = [
                "Quel est le sujet principal de ce document ?",
                "Quelles sont les conclusions ou recommandations ?",
                "Explique les méthodes utilisées.",
                "Quels sont les résultats clés ?",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                if cols[i % 2].button(sug, use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user', 'content': sug,
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()

    # ── TAB 2 : RÉSUMÉS ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 📋 Résumés automatiques")

        for doc_info in st.session_state.ingested_files:
            fname = doc_info['filename']
            col_h, col_btn = st.columns([4, 1])
            with col_h:
                st.markdown(f"**📄 {fname}**")
            with col_btn:
                gen_btn = st.button(
                    "Générer" if fname not in st.session_state.summaries else "Regénérer",
                    key=f"sum_{fname}", use_container_width=True
                )

            if gen_btn:
                with st.spinner(f"📊 Analyse de {fname}..."):
                    try:
                        # Récupérer le texte depuis le vectorstore
                        docs = pipeline.vectorstore.similarity_search(
                            "résumé synthèse contenu principal", k=6,
                            filter={"source": fname}
                        )
                        text = "\n\n".join([d.page_content for d in docs])
                        summary = pipeline.summarize(text=text)
                        st.session_state.summaries[fname] = summary
                    except Exception as e:
                        st.error(f"❌ {e}")

            if fname in st.session_state.summaries:
                s = st.session_state.summaries[fname]
                with st.expander(f"📊 Résumé — {s.get('title', fname)}", expanded=True):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Résumé :**\n\n{s.get('summary', '')}")
                        st.markdown("**Points clés :**")
                        for pt in s.get('key_points', []):
                            st.markdown(f"- {pt}")
                    with col_b:
                        st.markdown(f"""
                        <div style='background:#111;border:1px solid #222;border-radius:8px;padding:16px'>
                        <div class='tag tag-blue'>{s.get('domain', '?')}</div><br/>
                        <div class='tag tag-green'>{s.get('language', '?')}</div>
                        <div class='tag tag-purple'>{s.get('complexity', '?')}</div><br/><br/>
                        📝 <b>{s.get('n_words', '?')} mots</b><br/>
                        📄 <b>{doc_info['n_chunks']} chunks</b>
                        </div>
                        """, unsafe_allow_html=True)
            st.divider()

    # ── TAB 3 : COMPARAISON ───────────────────────────────────────────────────
    with tab3:
        st.markdown("#### ⚖️ Comparaison de documents")
        if len(st.session_state.ingested_files) < 2:
            st.info("Charger au moins 2 documents pour utiliser la comparaison.")
        else:
            compare_q = st.text_input(
                "Question de comparaison",
                placeholder="Ex: Quelles sont les conclusions sur les performances ?",
                help="Les réponses seront générées séparément pour chaque document."
            )
            if st.button("⚖️ Comparer", type="primary") and compare_q:
                with st.spinner("Comparaison en cours..."):
                    try:
                        results = pipeline.compare_documents(compare_q)
                        st.markdown(f"**Question :** {compare_q}")
                        st.divider()
                        cols = st.columns(len(results))
                        for col, (fname, answer) in zip(cols, results.items()):
                            with col:
                                st.markdown(f"**📄 {fname}**")
                                st.markdown(answer)
                    except Exception as e:
                        st.error(f"❌ {e}")

    # ── TAB 4 : MÉTRIQUES ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### 📊 Métriques & Analyse")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", pipeline.n_documents)
        with col2:
            st.metric("🧩 Chunks total", pipeline.total_chunks)
        with col3:
            st.metric("💬 Requêtes", st.session_state.total_queries)
        with col4:
            avg_lat = (st.session_state.total_latency / max(st.session_state.total_queries, 1))
            st.metric("⏱️ Latence moy.", f"{avg_lat:.2f}s")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            fig_chunks = plot_chunks_distribution(st.session_state.ingested_files)
            if fig_chunks:
                st.plotly_chart(fig_chunks, use_container_width=True)

        with col_b:
            fig_lat = plot_chat_activity(st.session_state.chat_history)
            if fig_lat:
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.info("Pose des questions pour voir les métriques de latence.")

        # Détails des documents
        if st.session_state.ingested_files:
            st.markdown("#### 📋 Détails des documents")
            df = pd.DataFrame(st.session_state.ingested_files)
            df.columns = ['Fichier', 'Pages', 'Chunks', 'Chunk moy. (chars)', 'Total chars']
            st.dataframe(df, use_container_width=True)

        # Export de l'historique
        if st.session_state.chat_history:
            st.markdown("#### 💾 Export")
            history_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Télécharger l'historique (JSON)",
                history_json,
                file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )


if __name__ == '__main__':
    main()
