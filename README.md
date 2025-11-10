# TúlioAI

<div align="center">

🎓 **Assistente de Estudos baseado em IA e RAG (Retrieval-Augmented Generation)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📋 Sobre

TúlioAI é um assistente de estudos inteligente que responde perguntas com base em documentos Markdown usando RAG (Retrieval-Augmented Generation).

## 🚀 Quick Start

```bash
# 1. Clone e entre no diretório
git clone https://github.com/thinkmadu/TutorAI.git && cd TutorAI

# 2. Execute o setup automático
python3 setup_env.py
```

O script configura ambiente virtual, dependências, FAISS e inicia a interface escolhida. 🎉

## ✨ Características

- 🔍 **Busca Vetorial**: FAISS para recuperação rápida e precisa
- 🧠 **LLM Local**: Modelos HuggingFace (TinyLlama, Mistral, etc.)
- 💻 **Dupla Interface**: CLI (terminal) e Streamlit (web)
- 🏗️ **Clean Architecture**: Domínio, serviços e infraestrutura separados
- 📚 **Base Personalizável**: Indexe seus próprios arquivos Markdown

## 🏗️ Arquitetura

```
tulioai/
├── main.py                    # Ponto de entrada
├── setup_env.py               # Configuração automática
├── create_faiss_index.py      # Criação do índice vetorial
├── core/                      # Domínio (entities, rules)
├── services/                  # Aplicação (rag_service)
├── infrastructure/            # Adapters (loaders, retrievers, generators)
├── interfaces/                # CLI e Streamlit
└── data/
    └── knowledge_base/        # Seus arquivos .md
```

## � Instalação

### Opção 1: Automática (Recomendado)

```bash
python3 setup_env.py
```

O script automaticamente:
- ✅ Cria ambiente virtual
- ✅ Instala dependências
- ✅ Configura `.env`
- ✅ Verifica/cria banco FAISS
- ✅ Inicia interface escolhida

### Opção 2: Manual

```bash
# Ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Dependências
pip install -r requirements.txt

# Configuração
cp .env.example .env  # Edite conforme necessário

# Índice FAISS
python create_faiss_index.py

# Iniciar
python main.py --interface cli
```

## 📖 Uso

### 1. Adicionar Documentos

```bash
# Adicione seus arquivos .md
cp seus_documentos/*.md data/knowledge_base/

# Reindexe
python create_faiss_index.py
```

### 2. Fazer Perguntas

**Interface CLI:**
```bash
python main.py --interface cli
```

**Interface Web:**
```bash
python main.py --interface streamlit
# Acesse http://localhost:8501
```

## ⚙️ Configuração

### Arquivo `.env`

```env
# Índice FAISS
FAISS_PATH=./models/faiss_index_tutorai

# Embeddings (HuggingFace)
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM (HuggingFace)
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Hardware
DEVICE=cpu  # ou cuda para GPU

# Geração
TEMPERATURE=0.1
MAX_NEW_TOKENS=512
TOP_K_RETRIEVAL=4

# Dados
KNOWLEDGE_BASE_PATH=./data/knowledge_base
```

### Modelos Recomendados

**Embeddings:**
- `sentence-transformers/all-MiniLM-L6-v2` (padrão, 80MB)
- `sentence-transformers/all-mpnet-base-v2` (melhor, 420MB)

**LLM:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (padrão, 1.1GB)
- `mistralai/Mistral-7B-Instruct-v0.2` (melhor, 7GB)

### GPU (CUDA)

```bash
# Instale PyTorch com CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Instale FAISS-GPU
pip uninstall faiss-cpu && pip install faiss-gpu

# Configure .env
DEVICE=cuda
```

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| Índice FAISS não encontrado | `python create_faiss_index.py` |
| Out of memory | Use modelo menor (TinyLlama) ou reduza `MAX_NEW_TOKENS` |
| Respostas ruins | Aumente `TOP_K_RETRIEVAL` ou melhore documentos |
| Indexação lenta | Use GPU ou reduza tamanho dos chunks |
| Erro de importação | `pip install --upgrade -r requirements.txt` |

## ❓ FAQ

<details>
<summary><b>Preciso rodar setup_env.py toda vez?</b></summary>

Não! Execute apenas na primeira instalação. Depois use `python main.py --interface cli/streamlit`.
</details>

<details>
<summary><b>Quando recriar o índice FAISS?</b></summary>

Sempre que adicionar/modificar documentos ou trocar modelo de embeddings:
```bash
python create_faiss_index.py
```
</details>

<details>
<summary><b>Posso usar GPU?</b></summary>

Sim! Se tem GPU NVIDIA:
1. Instale PyTorch com CUDA
2. Instale `faiss-gpu`
3. Configure `DEVICE=cuda` no `.env`
</details>

<details>
<summary><b>Qual modelo é melhor?</b></summary>

Depende do hardware:
- **TinyLlama** (1.1GB): Rápido, qualquer PC
- **Mistral-7B** (7GB): Melhor qualidade, precisa 16GB+ RAM
</details>

<details>
<summary><b>Como limpar instalação?</b></summary>

```bash
rm -rf .venv .env models/ logs/
# Mantenha data/ se tiver documentos importantes
```
</details>

## � Fluxo RAG

```
Pergunta → Embedding → Busca FAISS (Top-K) → Contexto → LLM → Resposta + Fontes
```

## 🛠️ Desenvolvimento

```bash
# Testes
pytest tests/

# Formatação
black . && isort . && flake8 .
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie branch: `git checkout -b feature/MinhaFeature`
3. Commit: `git commit -m 'Add MinhaFeature'`
4. Push: `git push origin feature/MinhaFeature`
5. Abra Pull Request

## 📝 Licença

MIT License - veja [LICENSE](LICENSE)

## 🙏 Agradecimentos

- [HuggingFace](https://huggingface.co/) - Modelos e Transformers
- [FAISS](https://github.com/facebookresearch/faiss) - Busca vetorial
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Streamlit](https://streamlit.io/) - Interface web

---

<div align="center">
Feito com ❤️ e ☕
</div>
