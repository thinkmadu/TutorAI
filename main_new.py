#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TúlioAI - Assistente de Estudos baseado em RAG

Ponto de entrada principal que permite escolher entre CLI e Streamlit
"""

import sys
import argparse


def main():
    """Função principal - ponto de entrada do TúlioAI"""
    parser = argparse.ArgumentParser(
        description="TúlioAI - Assistente de Estudos baseado em RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

# Executar interface CLI
python main.py --cli

# Executar interface Streamlit
python main.py --streamlit

# Executar CLI com uma única pergunta
python main.py --cli --pergunta "O que é Python?"

# Indexar nova base de conhecimento
python main.py --cli --indexar data/knowledge_base --indice data/faiss_index

Para mais opções da CLI, use:
python main.py --cli --help
        """
    )
    
    # Grupo mutuamente exclusivo para interface
    interface_group = parser.add_mutually_exclusive_group(required=True)
    
    interface_group.add_argument(
        "--cli",
        action="store_true",
        help="Executar interface de linha de comando (CLI)"
    )
    
    interface_group.add_argument(
        "--streamlit",
        action="store_true",
        help="Executar interface web com Streamlit"
    )
    
    # Argumentos opcionais (apenas para modo CLI)
    parser.add_argument(
        "--indexar",
        type=str,
        help="[CLI] Caminho para diretório com arquivos .md para indexar"
    )
    
    parser.add_argument(
        "--indice",
        type=str,
        default="data/faiss_index",
        help="[CLI] Caminho para salvar/carregar o índice FAISS (padrão: data/faiss_index)"
    )
    
    parser.add_argument(
        "--pergunta",
        type=str,
        help="[CLI] Fazer uma única pergunta e sair"
    )
    
    parser.add_argument(
        "--modo",
        type=str,
        choices=["answer", "explain", "summarize"],
        default="answer",
        help="[CLI] Modo de resposta (padrão: answer)"
    )
    
    parser.add_argument(
        "--modelo",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="[CLI] Modelo HuggingFace a usar (padrão: TinyLlama)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="[CLI] Número de chunks a recuperar (padrão: 4)"
    )
    
    # Parse argumentos
    args = parser.parse_args()
    
    # Executa interface apropriada
    if args.cli:
        print("🚀 Iniciando TúlioAI em modo CLI...")
        from cli_interface import main as cli_main
        
        # Reconstrói sys.argv para passar para CLI
        sys.argv = ["cli_interface.py"]
        
        if args.indexar:
            sys.argv.extend(["--indexar", args.indexar])
        
        if args.indice:
            sys.argv.extend(["--indice", args.indice])
        
        if args.pergunta:
            sys.argv.extend(["--pergunta", args.pergunta])
        
        if args.modo:
            sys.argv.extend(["--modo", args.modo])
        
        if args.modelo:
            sys.argv.extend(["--modelo", args.modelo])
        
        if args.top_k:
            sys.argv.extend(["--top-k", str(args.top_k)])
        
        cli_main()
    
    elif args.streamlit:
        print("🚀 Iniciando TúlioAI em modo Streamlit...")
        
        try:
            import subprocess
            # Executa Streamlit
            subprocess.run([
                sys.executable, 
                "-m", 
                "streamlit", 
                "run", 
                "streamlit_interface.py"
            ])
        except ImportError:
            print("❌ Erro: Streamlit não está instalado.")
            print("Instale com: pip install streamlit")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Erro ao executar Streamlit: {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operação cancelada pelo usuário\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
