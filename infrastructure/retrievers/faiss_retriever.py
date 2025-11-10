"""
FAISS Retriever - Adaptador de recuperação usando FAISS
Encapsula a lógica de recuperação de documentos usando FAISS e LangChain
"""

from typing import List
from langchain_community.vectorstores import FAISS
from core.entities import Documento


class FAISSRetriever:
    """Adaptador que encapsula a recuperação de documentos com FAISS"""
    
    def __init__(self, faiss_index: FAISS, embeddings):
        """
        Inicializa o retriever FAISS
        
        Args:
            faiss_index: Objeto FAISS (índice carregado)
            embeddings: Modelo de embeddings usado para criar o índice
        """
        self._faiss_index = faiss_index
        self._embeddings = embeddings
    
    def recuperar(self, pergunta: str, k: int = 4) -> List[Documento]:
        """
        Recupera documentos relevantes para a pergunta usando FAISS
        
        Args:
            pergunta: Texto da pergunta
            k: Número de documentos a recuperar (padrão: 4)
            
        Returns:
            Lista de entidades Documento relevantes
            
        Raises:
            Exception: Se houver erro durante a recuperação
        """
        try:
            # Realiza busca de similaridade usando FAISS
            langchain_docs = self._faiss_index.similarity_search(query=pergunta, k=k)
            
            # Converte documentos LangChain para entidades Documento
            documentos = []
            for lc_doc in langchain_docs:
                # Extrai conteúdo e metadados do documento LangChain
                conteudo = lc_doc.page_content
                metadados = lc_doc.metadata.copy() if hasattr(lc_doc, 'metadata') else {}
                
                # Adiciona caminho aos metadados se existir no metadata
                if 'source' in metadados and 'caminho' not in metadados:
                    metadados['caminho'] = metadados['source']
                
                # Cria entidade Documento
                documento = Documento(
                    conteudo=conteudo,
                    metadados=metadados
                )
                
                documentos.append(documento)
            
            return documentos
            
        except Exception as e:
            # Loga e relança exceção com contexto
            print(f"Erro ao recuperar documentos: {e}")
            raise Exception(f"Falha na recuperação de documentos: {str(e)}") from e
