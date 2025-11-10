#!/usr/bin/env python3
"""
TulioAI - Assistente de Estudos baseado em RAG
Ponto de entrada principal que permite escolher entre CLI e Streamlit
"""

import sys
import argparse
import os


def inicializar_rag_service():
    """Inicializa o RAGService com todos os componentes necessarios"""
    print("Inicializando componentes do sistema...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        from langchain_community.vectorstores import FAISS
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        
        from services.rag_service import RAGService
        from infrastructure.retrievers.faiss_retriever import FAISSRetriever
        from infrastructure.generators.llm_generator import LLMGenerator
        
        # Configuracoes
        FAISS_INDEX_PATH = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
        EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        LLM_MODEL = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # 1. Carrega embeddings
        print(f"Carregando embeddings: {EMBEDDINGS_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Carrega indice FAISS
        print(f"Carregando indice FAISS: {FAISS_INDEX_PATH}")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"Indice FAISS nao encontrado em {FAISS_INDEX_PATH}. "
                "Execute o script de indexacao primeiro."
            )
        
        faiss_index = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 3. Carrega modelo de linguagem
        print(f"Carregando modelo LLM: {LLM_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # 4. Cria adaptadores
        print("Criando adaptadores...")
        retriever = FAISSRetriever(faiss_index=faiss_index, embeddings=embeddings)
        generator = LLMGenerator(llm=llm)
        
        # 5. Cria RAGService
        print("Criando RAGService...")
        rag_service = RAGService(retriever=retriever, generator=generator)
        
        print("Sistema inicializado com sucesso!\n")
        return rag_service
        
    except ImportError as e:
        print(f"\nErro de importacao: {e}")
        print("Certifique-se de que todas as dependencias estao instaladas:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\nArquivo nao encontrado: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nErro ao inicializar sistema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def executar_cli():
    """Executa a interface CLI"""
    print("Iniciando interface CLI...\n")
    
    try:
        rag_service = inicializar_rag_service()
        from cli_interface import executar_cli
        executar_cli(rag_service)
        
    except KeyboardInterrupt:
        print("\n\nOperacao cancelada pelo usuario.\n")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nErro ao executar CLI: {e}\n")
        sys.exit(1)


def executar_streamlit():
    """Executa a interface Streamlit"""
    print("Iniciando interface Streamlit...\n")
    
    try:
        import subprocess
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_interface.py"
        ])
        
    except ImportError:
        print("Streamlit nao esta instalado.")
        print("Instale com: pip install streamlit\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"Erro ao executar Streamlit: {e}\n")
        sys.exit(1)


def main():
    """Funcao principal - ponto de entrada do TulioAI"""
    parser = argparse.ArgumentParser(
        description="TulioAI - Assistente de Estudos baseado em RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  python main.py --interface cli
  python main.py --interface streamlit

Variaveis de ambiente (opcionais):
  FAISS_PATH           Caminho para o indice FAISS
  EMBEDDINGS_MODEL     Modelo de embeddings
  LLM_MODEL            Modelo de linguagem
        """
    )
    
    parser.add_argument(
        "--interface",
        type=str,
        choices=["cli", "streamlit"],
        required=True,
        help="Interface a ser executada: 'cli' ou 'streamlit'"
    )
    
    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("TulioAI - Assistente de Estudos baseado em RAG")
    print("=" * 60 + "\n")
    
    try:
        if args.interface == "cli":
            executar_cli()
        elif args.interface == "streamlit":
            executar_streamlit()
    
    except KeyboardInterrupt:
        print("\n\nOperacao cancelada pelo usuario.\n")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nErro fatal: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
