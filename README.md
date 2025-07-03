# PrivateGPT

PrivateGPT is a production-ready AI framework that enables you to ask questions about your documents using Large Language Models (LLMs), even in fully offline scenarios. It is designed for privacy, ensuring that no data leaves your execution environment. This is a fork of [PrivateGPT](https://github.com/zylon-ai/private-gpt) by Zylon, check official documentation for installation and further queries.

![Gradio UI](/fern/docs/assets/ui.png?raw=true)

> **Note:** For the latest updates, always refer to the [official documentation](https://docs.privategpt.dev/).

---

## 🎞️ Overview

PrivateGPT provides an **API** containing all the building blocks required to build private, context-aware AI applications. It follows and extends the [OpenAI API standard](https://openai.com/blog/openai-api), supporting both normal and streaming responses.

- **High-level API:** Abstracts the complexity of RAG (Retrieval Augmented Generation) pipelines, including document ingestion, context retrieval, and chat/completions.
- **Low-level API:** Allows advanced users to implement custom pipelines, embeddings, and chunk retrieval.

A working [Gradio UI](https://www.gradio.app/) client is provided for easy interaction and testing.

---

## 📄 Documentation

Full documentation on installation, configuration, running the server, deployment, ingestion, API details, and UI features is available at:  
https://docs.privategpt.dev/

---

## 🧩 Architecture

PrivateGPT is built using [FastAPI](https://fastapi.tiangolo.com/) and [LlamaIndex](https://www.llamaindex.ai/).

- **APIs:** Defined in `private_gpt/server/<api>`. Each API has a router and a service implementation.
- **Components:** Located in `private_gpt/components/<component>`, providing actual implementations for LLMs, embeddings, vector stores, etc.
- **Dependency Injection:** Decouples components and layers for flexibility.
- **Extensible:** Easily swap or extend LLMs, embeddings, and vector stores.

**Directory Structure:**
```
private_gpt/
  ├── components/
  ├── open_ai/
  ├── server/
  ├── settings/
  ├── ui/
  ├── utils/
  ├── main.py
  └── ...
```

---

## 🚀 Installation & Running

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
- (Optional) `make` for running scripts

### Clone the Repository

```sh
git clone https://github.com/zylon-ai/private-gpt.git
cd private-gpt
```

### Install Dependencies

Choose your setup and install the required extras. For example, for a local Ollama-powered setup:

```sh
poetry install --extras "ui llms-ollama embeddings-ollama vector-stores-qdrant"
```

Other setups (OpenAI, Azure OpenAI, Llama-CPP, etc.) are supported. See [installation docs](https://docs.privategpt.dev/installation) for details.

### Run the Application

Set your desired profile and run:

```sh
PGPT_PROFILES=ollama make run
```

The UI will be available at [http://localhost:8001](http://localhost:8001).

---

## 📦 Features

- **Private, offline RAG pipeline**
- **Multiple LLM backends:** Ollama, Llama-CPP, OpenAI, Azure OpenAI, Gemini, Sagemaker, etc.
- **Flexible embeddings and vector stores:** HuggingFace, Qdrant, PostgreSQL, Milvus, Clickhouse, etc.
- **Gradio UI** for easy interaction
- **Extensible API** following OpenAI standards
- **Bulk and API-based document ingestion**
- **Recipes** for common use cases (e.g., summarization)

---

## 🛠️ Development & Testing

- **Run tests:**  
  ```sh
  make test
  ```
- **Check formatting and types:**  
  ```sh
  make check
  ```
- **Format code:**  
  ```sh
  make format
  ```

---

## 💡 Contributing

Contributions are welcome! Please run `make check` before committing.  
See the [Project Board](https://github.com/users/imartinez/projects/3) for ideas.

---

## 💬 Community

- [Twitter (X)](https://twitter.com/PrivateGPT_AI)
- [Discord](https://discord.gg/bK6mRVpErU)

---

## 📖 Citation

If you use PrivateGPT in a paper, see [CITATION.cff](CITATION.cff) or use:

```bibtex
@software{Zylon_PrivateGPT_2023,
author = {Zylon by PrivateGPT},
license = {Apache-2.0},
month = may,
title = {{PrivateGPT}},
url = {https://github.com/zylon-ai/private-gpt},
year = {2023}
}
```

---

## 🤗 Partners & Supporters

- [Qdrant](https://qdrant.tech/)
- [Fern](https://buildwithfern.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [LangChain](https://github.com/hwchase17/langchain)
- [GPT4All](https://github.com/nomic-ai/gpt4all)
- [LlamaCpp](https://github.com/ggerganov/llama.cpp)
- [Chroma](https://www.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)

---
