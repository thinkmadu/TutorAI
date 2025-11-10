"""
Streamlit Interface - Interface web para TúlioAI
Permite interação via navegador com o assistente de estudos
"""

import streamlit as st
import os
from services.rag_service import RAGService
from infrastructure.retrievers.faiss_retriever import FAISSRetriever
from infrastructure.generators.llm_generator import LLMGenerator
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# Configuração de caminhos (ajuste conforme necessário)
FAISS_INDEX_PATH = os.getenv("FAISS_PATH", "./data/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@st.cache_resource
def carregar_embeddings():
    """Carrega o modelo de embeddings (cached)"""
    with st.spinner("Carregando modelo de embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'}
        )
    return embeddings


@st.cache_resource
def carregar_faiss(_embeddings):
    """Carrega o índice FAISS (cached)"""
    with st.spinner("Carregando índice FAISS..."):
        try:
            faiss_index = FAISS.load_local(
                FAISS_INDEX_PATH,
                _embeddings,
                allow_dangerous_deserialization=True
            )
            return faiss_index
        except Exception as e:
            st.error(f"Erro ao carregar índice FAISS: {e}")
            st.info(f"Verifique se o índice existe em: {FAISS_INDEX_PATH}")
            return None


@st.cache_resource
def carregar_llm():
    """Carrega o modelo de linguagem (cached)"""
    with st.spinner(f"Carregando modelo {LLM_MODEL}..."):
        try:
            # Carrega tokenizer
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
            
            # Carrega modelo
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Cria pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Cria HuggingFacePipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
            
        except Exception as e:
            st.error(f"Erro ao carregar modelo LLM: {e}")
            return None


def inicializar_rag_service():
    """Inicializa o RAGService com todos os componentes"""
    # Carrega componentes
    embeddings = carregar_embeddings()
    faiss_index = carregar_faiss(embeddings)
    llm = carregar_llm()
    
    # Verifica se todos foram carregados
    if faiss_index is None or llm is None:
        st.error("❌ Não foi possível inicializar o sistema. Verifique os componentes.")
        st.stop()
    
    # Cria adaptadores
    retriever = FAISSRetriever(faiss_index=faiss_index, embeddings=embeddings)
    generator = LLMGenerator(llm=llm)
    
    # Cria e retorna RAGService
    rag_service = RAGService(retriever=retriever, generator=generator)
    return rag_service


def inicializar_session_state():
    """Inicializa variáveis de estado da sessão"""
    if 'historico' not in st.session_state:
        st.session_state.historico = []
    
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = inicializar_rag_service()


def exibir_historico():
    """Exibe o histórico de conversa"""
    for mensagem in st.session_state.historico:
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])
            
            # Se for resposta do assistente, exibe fontes
            if mensagem["role"] == "assistant" and "fontes" in mensagem:
                if mensagem["fontes"]:
                    with st.expander("📚 Ver fontes"):
                        for i, fonte in enumerate(mensagem["fontes"], 1):
                            st.markdown(f"**[{i}]** `{fonte}`")


def processar_pergunta(pergunta: str):
    """Processa a pergunta do usuário"""
    # Adiciona pergunta ao histórico
    st.session_state.historico.append({
        "role": "user",
        "content": pergunta
    })
    
    # Exibe pergunta do usuário
    with st.chat_message("user"):
        st.write(pergunta)
    
    # Processa com spinner
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            try:
                # Chama RAGService
                resposta = st.session_state.rag_service.gerar_resposta(pergunta_texto=pergunta)
                
                # Exibe resposta
                st.write(resposta.texto)
                
                # Exibe fontes
                if resposta.fontes:
                    with st.expander("📚 Ver fontes"):
                        for i, fonte in enumerate(resposta.fontes, 1):
                            st.markdown(f"**[{i}]** `{fonte.caminho}`")
                
                # Adiciona resposta ao histórico
                st.session_state.historico.append({
                    "role": "assistant",
                    "content": resposta.texto,
                    "fontes": [fonte.caminho for fonte in resposta.fontes]
                })
                
            except Exception as e:
                st.error(f"❌ Erro ao processar pergunta: {str(e)}")
                st.session_state.historico.append({
                    "role": "assistant",
                    "content": f"Desculpe, ocorreu um erro: {str(e)}"
                })


def main():
    """Função principal da interface Streamlit"""
    # Configuração da página
    st.set_page_config(
        page_title="TúlioAI",
        page_icon="🎓",
        layout="wide"
    )
    
    # Inicializa session state
    inicializar_session_state()
    
    # Título
    st.title("🎓 TúlioAI - Assistente de Estudos")
    st.markdown("---")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Sobre")
        st.info(
            "TúlioAI é um assistente de estudos baseado em RAG "
            "(Retrieval-Augmented Generation) que responde perguntas "
            "com base em uma base de conhecimento em Markdown."
        )
        
        st.header("⚙️ Configurações")
        st.text(f"Modelo LLM: {LLM_MODEL}")
        st.text(f"Embeddings: {EMBEDDINGS_MODEL}")
        st.text(f"Índice FAISS: {FAISS_INDEX_PATH}")
        
        if st.button("🗑️ Limpar histórico"):
            st.session_state.historico = []
            st.rerun()
    
    # Área de conversa
    st.subheader("💬 Conversa")
    
    # Container para histórico
    historico_container = st.container()
    
    with historico_container:
        exibir_historico()
    
    # Campo de entrada
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        pergunta = st.text_input(
            "Digite sua pergunta:",
            key="input_pergunta",
            placeholder="Ex: O que é Python?"
        )
    
    with col2:
        enviar = st.button("📤 Enviar", use_container_width=True)
    
    # Processa pergunta quando enviada
    if enviar and pergunta:
        processar_pergunta(pergunta)
        # Limpa o campo de entrada
        st.rerun()
    
    # Também aceita Enter no campo de texto
    if pergunta and not enviar:
        # Se o usuário pressionou Enter, a pergunta é processada
        # (Streamlit detecta automaticamente)
        pass


if __name__ == "__main__":
    main()
