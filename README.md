# 🧠 RAG Document Intelligence

> Pipeline RAG complet pour l'analyse intelligente de documents techniques — LangChain · Mistral AI · ChromaDB · Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Mistral](https://img.shields.io/badge/Mistral_AI-API-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)

## 🎯 Objectif

Système de Retrieval-Augmented Generation (RAG) permettant d'interroger des documents techniques en langage naturel, avec citation des sources et résumé automatique structuré.

**Cas d'usage :**
- Analyser des rapports techniques volumineux
- Extraire des insights de publications scientifiques
- Comparer plusieurs documents sur une même question
- Générer des résumés exécutifs automatiques

## 🏗️ Architecture RAG

```
Documents (PDF/DOCX/TXT)
        ↓
  RecursiveTextSplitter
  (chunks de 1000 chars, overlap 200)
        ↓
  Mistral Embeddings (mistral-embed)
        ↓
  ChromaDB (vectorstore persistant)
        ↓
Question utilisateur
        ↓
  MMR Retrieval (Maximum Marginal Relevance)
  top-k chunks les plus pertinents et diversifiés
        ↓
  Mistral LLM (mistral-small-latest)
  + Prompt engineering (réponse sourcée)
        ↓
  Réponse + Sources citées
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-documents** | Upload simultané de plusieurs PDF/DOCX/TXT |
| **Q&A sourcé** | Réponses avec citation des passages exacts |
| **Résumé structuré** | Titre, résumé, points clés, domaine, complexité |
| **Comparaison** | Même question sur plusieurs documents en parallèle |
| **MMR Retrieval** | Diversité maximale des chunks récupérés |
| **Historique** | Conversation persistante + export JSON |
| **Métriques** | Distribution des chunks, latence, stats |

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🔑 Obtenir une clé Mistral (gratuit)

1. Aller sur [console.mistral.ai](https://console.mistral.ai)
2. Créer un compte (pas de carte bancaire)
3. Générer une clé API dans "API Keys"
4. Coller la clé dans la sidebar du dashboard

## 🏃 Lancement

```bash
streamlit run app.py
```

## 📁 Structure

```
rag-document-intelligence/
├── core/
│   ├── rag_pipeline.py     ← Pipeline RAG (LangChain + Mistral + ChromaDB)
│   └── __init__.py
├── chroma_db/              ← Vectorstore persistant (généré automatiquement)
├── app.py                  ← Dashboard Streamlit
├── requirements.txt
└── README.md
```

## 🛠️ Stack technique

- **LangChain** — Orchestration du pipeline RAG
- **Mistral AI** — Embeddings (`mistral-embed`) + LLM (`mistral-small`)
- **ChromaDB** — Base vectorielle persistante
- **MMR (Maximum Marginal Relevance)** — Retrieval avec diversité
- **Streamlit** — Dashboard interactif

## 👩‍💻 Auteur

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[GitHub](https://github.com/KalsoumDS) · [Email](mailto:s.sall@mundiapolis.ma)
