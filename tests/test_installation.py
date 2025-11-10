"""
Script de teste simples para verificar a instalação

Nota: Este projeto usa duas nomenclaturas:
- TutorAI: nome do repositório (referências técnicas, imports)
- TúlioAI: nome do produto (interfaces de usuário)
"""

def test_imports():
    """Testa se todas as bibliotecas necessárias estão instaladas"""
    print("🔍 Verificando dependências...\n")
    
    erros = []
    
    # Core
    try:
        from core.entities import Pergunta, Resposta, Documento, Fonte
        print("✅ core.entities")
    except ImportError as e:
        erros.append(f"❌ core.entities: {e}")
        print(f"❌ core.entities: {e}")
    
    try:
        from core.rules import RegrasDeDominio
        print("✅ core.rules")
    except ImportError as e:
        erros.append(f"❌ core.rules: {e}")
        print(f"❌ core.rules: {e}")
    
    # Services
    try:
        from services.rag_service import RAGService
        print("✅ services.rag_service")
    except ImportError as e:
        erros.append(f"❌ services.rag_service: {e}")
        print(f"❌ services.rag_service: {e}")
    
    # Infrastructure
    try:
        from infrastructure.loaders.markdown_loader import MarkdownLoader, TextSplitter
        print("✅ infrastructure.loaders.markdown_loader")
    except ImportError as e:
        erros.append(f"❌ infrastructure.loaders.markdown_loader: {e}")
        print(f"❌ infrastructure.loaders.markdown_loader: {e}")
    
    # Bibliotecas externas
    try:
        import numpy as np
        print(f"✅ numpy (versão {np.__version__})")
    except ImportError as e:
        erros.append(f"❌ numpy: {e}")
        print(f"❌ numpy: {e}")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers")
    except ImportError as e:
        erros.append(f"❌ sentence-transformers: {e}")
        print(f"❌ sentence-transformers: {e}")
    
    try:
        import transformers
        print(f"✅ transformers (versão {transformers.__version__})")
    except ImportError as e:
        erros.append(f"❌ transformers: {e}")
        print(f"❌ transformers: {e}")
    
    try:
        import torch
        print(f"✅ torch (versão {torch.__version__})")
        if torch.cuda.is_available():
            print(f"   🎮 GPU CUDA disponível: {torch.cuda.get_device_name(0)}")
        else:
            print("   💻 Usando CPU")
    except ImportError as e:
        erros.append(f"❌ torch: {e}")
        print(f"❌ torch: {e}")
    
    try:
        import faiss
        print("✅ faiss")
    except ImportError as e:
        erros.append(f"❌ faiss: {e}")
        print(f"❌ faiss: {e}")
    
    # Opcional
    try:
        import streamlit
        print(f"✅ streamlit (versão {streamlit.__version__}) [opcional]")
    except ImportError:
        print("⚠️  streamlit não instalado [opcional]")
    
    print("\n" + "="*60)
    
    if erros:
        print(f"\n❌ {len(erros)} erro(s) encontrado(s):")
        for erro in erros:
            print(f"  {erro}")
        print("\nInstale as dependências faltantes com:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas as dependências necessárias estão instaladas!")
        print("\nVocê pode começar a usar o TúlioAI:")
        print("  python main.py --cli")
        return True


def test_entities():
    """Testa criação de entidades básicas"""
    print("\n🧪 Testando entidades...\n")
    
    from core.entities import Pergunta, Resposta, Documento, Fonte
    
    try:
        # Teste Fonte
        fonte = Fonte(
            caminho="test.md",
            titulo="Teste",
            relevancia_score=0.95
        )
        print(f"✅ Fonte criada: {fonte}")
        
        # Teste Documento
        doc = Documento(
            conteudo="Conteúdo de teste",
            caminho="test.md",
            titulo="Documento de Teste"
        )
        print(f"✅ Documento criado: {doc.titulo}")
        
        # Teste Pergunta
        pergunta = Pergunta(
            texto="O que é Python?",
            modo="answer"
        )
        print(f"✅ Pergunta criada: {pergunta.texto}")
        
        # Teste Resposta
        resposta = Resposta(
            texto="Python é uma linguagem de programação.",
            fontes=[fonte],
            confianca=0.9
        )
        print(f"✅ Resposta criada: confiança {resposta.confianca:.2%}")
        
        print("\n✅ Todas as entidades funcionam corretamente!")
        return True
    
    except Exception as e:
        print(f"\n❌ Erro ao testar entidades: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TúlioAI - Teste de Instalação")
    print("="*60 + "\n")
    
    imports_ok = test_imports()
    
    if imports_ok:
        entities_ok = test_entities()
        
        if entities_ok:
            print("\n" + "="*60)
            print("🎉 Sistema pronto para uso!")
            print("="*60 + "\n")
        else:
            print("\n⚠️  Alguns testes falharam")
    else:
        print("\n⚠️  Instale as dependências primeiro")
