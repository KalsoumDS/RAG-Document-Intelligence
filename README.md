# 📄 Intelligence Documentaire RAG & Analyse de Politiques / ESG

> Démocratisation de l'accès aux réglementations, au droit du travail et aux rapports d'impact (ESG, Climat) via RAG Multimodal — LangChain · Mistral AI · ChromaDB · Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Mistral](https://img.shields.io/badge/Mistral_AI-API-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)

---

## 🎯 Problématique Métier & Impact Sociétal

Les ONG, citoyens et PME se perdent dans des milliers de pages de textes juridiques, rapports d'impact environnemental (RSE/ESG) ou réglementations administratives complexes.

### 💡 Solution & Valeur Ajoutée
Cette plateforme d'**Intelligence Documentaire RAG** permet d'interroger des corpus volumineux en langage naturel, en garantissant des **réponses 100% sourcées** avec citation exacte des pages et paragraphes d'origine.
- **Transparence & Zéro Hallucination** : Filtrage strict par Maximum Marginal Relevance (MMR) et citation des sources.
- **Accès Simplifié aux Textes de Loi & ESG** : Synthèse automatique des points clés et extraction d'insights complexes en quelques secondes.

---

## 🏗️ Architecture RAG

```
Documents complexes (PDF, DOCX, TXT - Textes de Loi / Rapports RSE)
                            ↓
                 RecursiveTextSplitter
               (chunks 1000 chars, overlap 200)
                            ↓
               Mistral Embeddings (mistral-embed)
                            ↓
               ChromaDB (Vectorstore persistant)
                            ↓
                    Question Utilisateur
                            ↓
       MMR Retrieval (Maximum Marginal Relevance - Top-K)
                            ↓
         Mistral LLM (mistral-small) + Prompt Engineering
                            ↓
                Réponse Structurée & Sourcée
```

---

## 🚀 Fonctionnalités Clés

- 📁 **Multi-Documents** : Ingestion simultanée de plusieurs rapports et textes réglementaires.
- 🎯 **Q&A 100% Sourcé** : Réponses précises accompagnées de la citation exacte des extraits sources.
- 📝 **Résumé Exécutif Automatique** : Extraction des points clés, du domaine et du niveau de complexité.
- 🔍 **Recherche MMR Vectorielle** : Maximisation de la diversité des passages récupérés pour éviter la redondance.
- 📊 **Métriques de Latence & Chunks** : Suivi de la distribution des embeddings et du temps de réponse.

---

## 🛠️ Installation & Lancement

```bash
# Cloner le dépôt
git clone https://github.com/KalsoumDS/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence

# Installer les dépendances
pip install -r requirements.txt

# Obtenir une clé API Mistral (Gratuit sur console.mistral.ai)
# Puis lancer l'application Streamlit
streamlit run app.py
```

---

## 🔬 Stack Technique

- **LangChain** — Orchestration globale du pipeline RAG
- **Mistral AI** — API Embeddings (`mistral-embed`) & LLM (`mistral-small`)
- **ChromaDB** — Base de données vectorielle persistante
- **Streamlit & Plotly** — Interface utilisateur & métriques d'analyse

---

## ✍️ Auteur

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall) · [GitHub](https://github.com/KalsoumDS) · [Portfolio](http://localhost:3001)
