"""
CLI Interface - Interface de linha de comando para TúlioAI
Permite interação via terminal com o assistente de estudos
"""

from services.rag_service import RAGService
from core.entities import Resposta


def executar_cli(rag_service: RAGService) -> None:
    """
    Executa a interface de linha de comando interativa
    
    Args:
        rag_service: Instância configurada do RAGService
    """
    # Exibe banner de boas-vindas
    print("\n" + "=" * 60)
    print("🎓 TúlioAI - Assistente de Estudos")
    print("=" * 60)
    print("\nDigite suas perguntas ou 'sair'/'exit' para encerrar.\n")
    
    # Loop principal da CLI
    while True:
        try:
            # Lê entrada do usuário
            entrada = input("TúlioAI> ").strip()
            
            # Verifica se usuário quer sair
            if entrada.lower() in ['sair', 'exit', 'quit']:
                print("\n👋 Até logo! Bons estudos!\n")
                break
            
            # Ignora entradas vazias
            if not entrada:
                continue
            
            # Chama o RAGService para gerar resposta
            print("\n🤔 Processando sua pergunta...\n")
            resposta = rag_service.gerar_resposta(pergunta_texto=entrada)
            
            # Exibe a resposta formatada
            exibir_resposta(resposta)
            
        except KeyboardInterrupt:
            # Permite sair com Ctrl+C
            print("\n\n👋 Até logo! Bons estudos!\n")
            break
            
        except Exception as e:
            # Trata erros inesperados
            print(f"\n❌ Erro: {str(e)}\n")
            print("Por favor, tente novamente.\n")


def exibir_resposta(resposta: Resposta) -> None:
    """
    Exibe a resposta formatada no terminal
    
    Args:
        resposta: Objeto Resposta gerado pelo RAGService
    """
    # Exibe o texto da resposta
    print("💡 Resposta:")
    print("-" * 60)
    print(resposta.texto)
    print("-" * 60)
    
    # Exibe as fontes
    if resposta.fontes:
        print(f"\n📚 Fontes ({len(resposta.fontes)}):")
        for i, fonte in enumerate(resposta.fontes, 1):
            print(f"  [{i}] {fonte.caminho}")
    else:
        print("\n📚 Fontes: Nenhuma fonte encontrada")
    
    print()  # Linha em branco para separação


if __name__ == "__main__":
    # Este bloco permite executar a CLI diretamente se necessário
    # Normalmente, a CLI será chamada de main.py
    print("⚠️  Execute este módulo através do main.py")
    print("Exemplo: python main.py")
