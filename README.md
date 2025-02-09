<div align="center">

# 🚀 AdvancedRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/YanCotta/AdvancedRAG/wiki)

*A state-of-the-art implementation of Retrieval Augmented Generation with advanced techniques*

[Features](#key-features) • [Getting Started](#setup--usage) • [Documentation](#file-structure) • [Contributing](#contributing--license)

</div>

---

## 🎯 Overview
A sophisticated implementation of advanced Retrieval Augmented Generation (RAG) techniques, featuring multi-strategy retrieval, automated evaluation, and modular architecture.

## ✨ Key Features
<table>
<tr>
<td>

### 🔍 Multi-Strategy Retrieval Pipeline
- **AutoMerging Retrieval** with hierarchical node parsing
- **Sentence Window Retrieval** for granular context
- **Cross-encoder reranking** for enhanced relevance
- **Multi-hop reasoning** capabilities

</td>
<td>

### 📊 Advanced Evaluation Framework
- **Integrated TruLens** evaluation
- **Confidence scoring** and analysis
- **Automated groundedness** assessment
- **Performance metrics** dashboard

</td>
</tr>
</table>

## 📁 File Structure
```
AdvancedRAG/
├── AutoMergingRetrieval/
│   ├── utils.py              # Core utilities
│   └── AutoMergingRetrieval.py
├── AdvancedRAGPipeline/
│   ├── src/
│   │   ├── utils.py         # Pipeline utilities
│   │   └── pipeline.py      # RAG orchestration
│   └── data/                # Evaluation sets
└── data/                    # Shared resources
```

## 🛠️ Techniques and Methodologies

### AutoMerging Retrieval
> Utilizes hierarchical node parsing to merge document nodes across varying levels of granularity, resulting in more contextualized retrieval.

### Sentence Window Retrieval
> Extracts text in overlapping windows to capture granular context, enhancing retrieval precision.

### TruLens Evaluation
> Integrates feedback mechanisms that measure answer relevance and groundedness, ensuring high-quality responses.

## 🔧 Implementation Details

<details>
<summary><b>AutoMergingRetrieval</b></summary>

- Implements dynamic node size adjustment
- Uses similarity-based merging strategies
- Supports customizable merging thresholds
</details>

<details>
<summary><b>Advanced RAG Pipeline</b></summary>

- Integrates multiple retrieval strategies
- Features automated evaluation loops
- Provides detailed performance metrics
</details>

## 🚦 Setup & Usage

### Prerequisites
- Python 3.8+
- OpenAI API key
- HuggingFace API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YanCotta/AdvancedRAG.git
   cd AdvancedRAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   Create `.env` in project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

4. **Run the pipelines:**
   ```bash
   # For basic and auto-merging retrieval
   python src/run_retrieval.py

   # For full RAG pipeline with evaluations
   python AdvancedRAGPipeline/src/run_pipeline.py
   ```

## 📝 Contributing & License
We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

Licensed under the [MIT License](LICENSE).

---

<div align="center">
<p>Built with ❤️ by the AdvancedRAG Team</p>

[Report Bug](https://github.com/YanCotta/AdvancedRAG/issues) • [Request Feature](https://github.com/YanCotta/AdvancedRAG/issues)
</div>