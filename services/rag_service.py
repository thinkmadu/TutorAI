"""
RAG Service - Serviço principal do sistema RAG
Coordena o fluxo de Recuperação e Geração Aumentada
"""

from typing import Protocol, List
from core.entities import Pergunta, Resposta, Documento, Fonte


class RetrieverProtocol(Protocol):
    """Protocolo/Interface para Adaptadores de Recuperação"""
    
    def recuperar(self, pergunta: str, k: int) -> List[Documento]:
        """
        Recupera documentos relevantes para a pergunta
        
        Args:
            pergunta: Texto da pergunta
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos relevantes
        """
        ...


class GeneratorProtocol(Protocol):
    """Protocolo/Interface para Adaptadores de Geração"""
    
    def gerar(self, pergunta: str, contexto: List[Documento]) -> str:
        """
        Gera resposta baseada na pergunta e contexto
        
        Args:
            pergunta: Texto da pergunta
            contexto: Lista de documentos recuperados
            
        Returns:
            Texto da resposta gerada
        """
        ...


class RAGService:
    """Serviço que coordena o fluxo principal do RAG"""
    
    def __init__(self, retriever: RetrieverProtocol, generator: GeneratorProtocol):
        """
        Inicializa o serviço RAG
        
        Args:
            retriever: Adaptador de Recuperação
            generator: Adaptador de Geração
        """
        self._retriever = retriever
        self._generator = generator
    
    def gerar_resposta(self, pergunta_texto: str) -> Resposta:
        """
        Gera uma resposta para a pergunta usando RAG
        
        Args:
            pergunta_texto: Texto da pergunta do usuário
            
        Returns:
            Entidade Resposta com texto gerado e fontes
        """
        # 1. Instancia entidade Pergunta
        
        # 2. Recupera documentos relevantes usando o retriever
        documentos_recuperados = self._retriever.recuperar(pergunta=pergunta_texto, k=4)
        
        # 3. Gera resposta usando o generator
        texto_gerado = self._generator.gerar(
            pergunta=pergunta_texto,
            contexto=documentos_recuperados
        )
        
        # 4. Extrai fontes dos documentos recuperados
        fontes = []
        for doc in documentos_recuperados:
            # Assumindo que Documento tem atributo 'caminho' nos metadados
            caminho = doc.metadados.get('caminho', 'Desconhecido')
            fonte = Fonte(caminho=caminho)
            fontes.append(fonte)
        
        # 5. Cria e retorna entidade Resposta
        resposta = Resposta(
            texto=texto_gerado,
            fontes=fontes
        )
        
        return resposta
