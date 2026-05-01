<div align="center">

# 🏥 CareVision AI

### Privacy-Preserving Multimodal Clinical Assistant

*A research prototype demonstrating efficient data systems for AI with built-in user rights guarantees*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat-square)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GDPR](https://img.shields.io/badge/GDPR-Art.%2017%20Compliant-blue?style=flat-square)](https://gdpr-info.eu/art-17-gdpr/)

</div>

---

## Overview

CareVision is a full-stack multimodal AI pipeline built around a telemedicine use case. It processes consultation videos through speech recognition, computer vision, and an LLM agent, then provides a **verifiable, audited one-click data purge** implementing the GDPR Article 17 Right to be Forgotten.

Built as a demonstration prototype aligned with the research goals of developing efficient data systems for AI/ML that are easy to use while guaranteeing fundamental user rights.

---

## ✨ Key Features

| Feature | What it does |
|---|---|
| 🎙️ **Speech-to-Text** | OpenAI Whisper with language auto-detection and word-level timestamps |
| 👁️ **Vision Analysis** | OpenCV face detection + per-frame feature extraction (brightness, contrast, motion) |
| 🤖 **Anomaly Detection** | scikit-learn `IsolationForest` on per-frame feature matrix flags unusual events |
| 🦜 **LLM Orchestration** | LangChain LCEL chain: `ChatPromptTemplate \| Mistral \| JsonOutputParser` |
| 💬 **Chat Agent** | Multi-turn `ClinicalChatAgent` — domain experts query records in natural language |
| 🗄️ **Vector Store** | ChromaDB persistent client with cosine similarity (sentence-transformers) |
| 🔍 **Semantic Search** | Natural language queries across all stored transcripts |
| 🗑️ **Right to be Forgotten** | Audited GDPR Art. 17 pipeline — vectors, files, audio, frames all purged |
| 📋 **Audit Trail** | Tamper-evident timestamped JSON log of every data operation |

---

## 🏗️ Architecture

```
MP4 Video Upload
      │
      ├──► 🎙️  Whisper ASR
      │         └─ word-level transcript · language detection · persisted to disk
      │
      ├──► 👁️  OpenCV + scikit-learn
      │         └─ 1 FPS frame extraction
      │            per-frame: brightness · contrast · face count · motion blur
      │            IsolationForest anomaly detection on feature matrix
      │
      ├──► 🦜  LangChain LCEL Chain
      │         └─ ChatPromptTemplate | MistralAI | JsonOutputParser
      │            → { symptoms, observations, risk_flags, summary, followup }
      │
      ├──► 🗄️  ChromaDB Vector Store
      │         └─ sentence-transformers embeddings · cosine similarity
      │            semantic search across all patient records
      │
      └──► 🗑️  GDPR Art. 17 Deletion Pipeline
                └─ Step 1: delete ChromaDB vectors
                   Step 2: purge audio WAV · transcript TXT · video frames · source MP4
                   Step 3: write compliance entry to audit log
```

---

## 🖥️ Application Tabs

### 📹 Video Intake
Upload an MP4 consultation video. The full pipeline runs automatically — Whisper transcription, OpenCV frame analysis with scikit-learn anomaly scoring, LangChain summarisation, and ChromaDB storage. Results display in real-time with a colour-coded status tracker.

### 💬 Chat Agent
Ask free-form questions about the processed record in natural language. The `ClinicalChatAgent` maintains multi-turn conversation memory so clinicians can drill into symptoms, observations, or flag concerns interactively.

### 🔍 Semantic Search
Run cosine-similarity queries across all stored transcripts. Useful for finding similar past consultations or specific symptom patterns across a patient population.

### 🗑️ Data Deletion
One-click GDPR Art. 17 purge with a mandatory confirmation step. Every deletion is logged with a timestamp — the audit trail itself is preserved so compliance can be demonstrated to regulators.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/carevision-ai
cd carevision-ai

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Mistral API key (free at console.mistral.ai)
mkdir -p .streamlit
echo 'MISTRAL_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# 5. Launch
streamlit run app.py
```

> **Note** If No API key, the app runs in demo mode with a structured fallback output — all pipeline stages (Whisper, OpenCV, ChromaDB, deletion) still execute fully.

---

## 🔑  API Key

The LLM component uses **Mistral AI**:

1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Sign up with an email address
3. Navigate to **API Keys** → **Create new key**
4. Paste it into `.streamlit/secrets.toml`:

```toml
MISTRAL_API_KEY = "your_key_here"
```

---

## 📁 Project Structure

```
carevision/
├── app.py                    # Streamlit UI — 4 tabs, custom CSS, audit sidebar
├── requirements.txt
├── .streamlit/
│   └── secrets.toml          # API keys (git-ignored)
├── pipeline/
│   ├── __init__.py
│   ├── agent.py              # LangChain LCEL chain + ClinicalChatAgent
│   ├── audio.py              # Whisper ASR with word-level timestamps
│   ├── video.py              # OpenCV frame extraction + sklearn anomaly detection
│   ├── storage.py            # ChromaDB vector store + GDPR Art. 17 deletion
│   ├── validation.py         # Data quality gate (transcript · frames · audio)
│   └── logging_utils.py      # Timestamped JSON audit log
├── data/
│   ├── uploads/              # Source MP4 files (purged on deletion)
│   ├── transcripts/          # WAV audio + TXT transcript (purged on deletion)
│   ├── frames/               # Extracted JPEG frames (purged on deletion)
│   └── embeddings/           # ChromaDB on-disk vector store
└── logs/
    └── audit_log.json        # Persistent compliance audit trail
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **UI** | Streamlit + custom CSS | Rapid prototyping for research demos |
| **Speech** | OpenAI Whisper | State-of-the-art open-source ASR |
| **Vision** | OpenCV · scikit-learn | Lightweight frame analysis without heavy GPU models |
| **Orchestration** | LangChain LCEL | Composable, declarative AI pipelines |
| **LLM** | Mistral AI | Free tier · strong reasoning · JSON mode |
| **Vector DB** | ChromaDB | Local-first, zero-infrastructure vector search |
| **Embeddings** | sentence-transformers | High-quality semantic similarity, runs offline |

---

## 🔬 Research Relevance

This prototype directly addresses the DEEM Lab's core research themes:

**Data Engineering for AI/ML**
The pipeline implements a full ETL workflow for heterogeneous multimodal data (video → audio → frames → embeddings), with explicit data lineage and quality validation at each stage.

**Right to be Forgotten**
The deletion pipeline operates across every storage layer (filesystem, vector database, transcript store) with a timestamped audit log, demonstrating that complete erasure is both technically feasible and verifiable.

**Agentic Systems for Regulated Environments**
The `ClinicalChatAgent` demonstrates how LLM agents can assist domain experts (clinicians) in highly regulated environments, with explainable, loggable decisions and no hallucination of clinical data.

**Live Demo Capability**
The Streamlit interface is designed for presentations at events like *Lange Nacht der Wissenschaften* a single MP4 upload triggers the full pipeline in real-time with visible progress and immediate results.

---

## 🔭 Roadmap / Extensions

- [ ] Swap Whisper `base` for `medium` / `large-v3` for production accuracy
- [ ] Add **DSPy** optimised prompt signatures for automatic prompt tuning
- [ ] Integrate **PyTorch** MediaPipe pose estimation for clinical posture analysis
- [ ] **Voice interface** — real-time microphone input via `streamlit-webrtc`
- [ ] **Mobile view** — responsive layout for tablet use in clinical settings
- [ ] Replace Mistral API with **Ollama** (`ollama run mistral`) for fully air-gapped deployment

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ 
</div>