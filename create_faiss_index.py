#!/usr/bin/env python3
"""
create_faiss_index.py - Script para criar índice FAISS a partir de documentos Markdown

Este script:
1. Carrega documentos .md do diretório data/knowledge_base/
2. Divide em chunks
3. Gera embeddings
4. Cria índice FAISS
5. Salva índice para uso posterior
"""

import os
import sys
from pathlib import Path


def print_step(message):
    """Imprime passo atual"""
    print(f"► {message}")


def print_success(message):
    """Imprime mensagem de sucesso"""
    print(f"✓ {message}")


def print_error(message):
    """Imprime mensagem de erro"""
    print(f"✗ {message}")


def main():
    """Função principal"""
    print("\n" + "=" * 70)
    print("TulioAI - Criação de Índice FAISS".center(70))
    print("=" * 70 + "\n")
    
    try:
        # Carrega variáveis de ambiente
        from dotenv import load_dotenv
        load_dotenv()
        
        print_step("Importando bibliotecas...")
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # Configurações
        KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
        FAISS_PATH = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
        EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        print_success("Bibliotecas importadas")
        
        # Verifica se o diretório de documentos existe
        kb_path = Path(KNOWLEDGE_BASE_PATH)
        if not kb_path.exists():
            print_error(f"Diretório não encontrado: {KNOWLEDGE_BASE_PATH}")
            print(f"\nCrie o diretório e adicione arquivos .md:")
            print(f"  mkdir -p {KNOWLEDGE_BASE_PATH}")
            return 1
        
        # Verifica se há arquivos .md
        md_files = list(kb_path.glob("**/*.md"))
        if not md_files:
            print_error(f"Nenhum arquivo .md encontrado em {KNOWLEDGE_BASE_PATH}")
            print(f"\nAdicione arquivos Markdown ao diretório:")
            print(f"  {KNOWLEDGE_BASE_PATH}/")
            return 1
        
        print_success(f"Encontrados {len(md_files)} arquivo(s) .md")
        
        # Carrega documentos
        print_step("Carregando documentos...")
        loader = DirectoryLoader(
            KNOWLEDGE_BASE_PATH,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()
        print_success(f"{len(documents)} documento(s) carregado(s)")
        
        # Divide em chunks
        print_step("Dividindo documentos em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print_success(f"{len(chunks)} chunk(s) criado(s)")
        
        # Carrega modelo de embeddings
        print_step(f"Carregando modelo de embeddings: {EMBEDDINGS_MODEL}")
        print("(Isso pode demorar na primeira vez...)")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        print_success("Modelo de embeddings carregado")
        
        # Cria índice FAISS
        print_step("Criando índice FAISS...")
        print("(Gerando embeddings para todos os chunks...)")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print_success("Índice FAISS criado")
        
        # Salva índice
        print_step(f"Salvando índice em: {FAISS_PATH}")
        Path(FAISS_PATH).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(FAISS_PATH)
        print_success("Índice salvo com sucesso!")
        
        # Testa o índice
        print_step("Testando índice...")
        test_query = "teste"
        results = vectorstore.similarity_search(test_query, k=1)
        print_success(f"Teste OK - {len(results)} resultado(s) encontrado(s)")
        
        print("\n" + "=" * 70)
        print_success("Índice FAISS criado com sucesso!")
        print("=" * 70)
        print(f"\nEstatísticas:")
        print(f"  Arquivos processados: {len(md_files)}")
        print(f"  Documentos carregados: {len(documents)}")
        print(f"  Chunks gerados: {len(chunks)}")
        print(f"  Local do índice: {FAISS_PATH}")
        print(f"\nAgora você pode usar o TulioAI:")
        print(f"  python main.py --interface cli")
        print(f"  python main.py --interface streamlit")
        print()
        
        return 0
        
    except ImportError as e:
        print_error(f"Erro de importação: {e}")
        print("\nCertifique-se de que as dependências estão instaladas:")
        print("  pip install -r requirements.txt")
        print("\nOu execute primeiro:")
        print("  python setup_env.py")
        return 1
        
    except Exception as e:
        print_error(f"Erro ao criar índice: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
