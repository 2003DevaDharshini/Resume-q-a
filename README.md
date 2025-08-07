# 📄 Resume-Based Q&A System using LLMs

This project is a **Resume Question Answering System** that uses **LlamaIndex**, **Gemini (Google's LLM)**, and **FastAPI** to allow users to ask natural language questions about their own resume and receive grounded responses.

---

## ✅ Task Objective

Build an application that:

- Uses your **own resume** as the input data source.
- Accepts questions as JSON payload via a **POST** request or web form.
- Uses **LlamaIndex** to embed and index your resume.
- Retrieves the most relevant content from the resume.
- Uses a **LLM (Gemini)** with a **structured prompt** to generate answers.
- Avoids hallucinations – only answers using the resume content.
- Provides a downloadable JSON output.

---

## 🧠 Tech Stack

- Python
- [LlamaIndex](https://www.llamaindex.ai/)
- Google Gemini API
- FastAPI
- Jinja2 (for HTML templating)
- HTML/CSS for front-end interface

---

## 📁 Project Structure

```bash
resume-qa/
├── main.py                  # FastAPI backend with LlamaIndex and Gemini
├── resume.txt               # Your resume in plain text format
├── storage/                 # LlamaIndex persistent storage
├── templates/
│   └── index.html           # Frontend HTML interface
├── .env                     # Contains GEMINI_API_KEY
└── requirements.txt         # All dependencies
