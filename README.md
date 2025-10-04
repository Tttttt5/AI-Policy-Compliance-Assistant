# AI Policy Compliance Assistant  
*A local GenAI-powered app using LangChain + Streamlit + Ollama to compare and analyze data protection laws.*

---

## Overview
This project uses **Retrieval-Augmented Generation (RAG)** to analyze and compare global data protection laws, such as:

- **GDPR (EU General Data Protection Regulation)**
- **India’s Digital Personal Data Protection Act (DPDP 2023)**
-  **UNESCO AI Ethics Guidelines**

It runs completely **offline** using local language models (via Ollama) and features a Streamlit web app interface.

---

## Features
Compare legal documents and AI ethics frameworks  
Retrieve and analyze relevant sections from PDFs  
Generate structured, sectioned responses (Overview, Similarities, Differences, Summary)  
Fully offline — no OpenAI API required  
Clean and interactive web UI with Streamlit  

---

##  Tech Stack
| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend Logic | Python + LangChain |
| Vector DB | Chroma |
| Embeddings | HuggingFace Sentence Transformers |
| Local LLM | Ollama (Phi3:mini / TinyLlama / Mistral) |
| PDF Parsing | PyPDF2 |

---

##  How to Run Locally
Follow these steps:

###  Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Policy-Compliance-Assistant.git
cd AI-Policy-Compliance-Assistant
