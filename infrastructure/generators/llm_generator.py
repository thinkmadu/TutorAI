"""
LLM Generator - Adaptador de geração de texto
Encapsula a lógica de geração usando modelos de linguagem HuggingFace
"""

from typing import Any, List, Union
from langchain_huggingface import HuggingFacePipeline
from core.entities import Documento


class LLMGenerator:
    """Adaptador que encapsula a geração de texto com LLM"""
    
    def __init__(self, llm: Union[HuggingFacePipeline, Any]):
        """
        Inicializa o generator com um modelo de linguagem
        
        Args:
            llm: Objeto HuggingFacePipeline ou modelo carregado
        """
        self._llm = llm
    
    def gerar(self, pergunta: str, contexto: List[Documento]) -> str:
        """
        Gera resposta baseada na pergunta e contexto
        
        Args:
            pergunta: Texto da pergunta
            contexto: Lista de documentos recuperados
            
        Returns:
            Texto da resposta gerada
            
        Raises:
            Exception: Se houver erro durante a geração
        """
        try:
            # Formata o contexto concatenando o conteúdo dos documentos
            contexto_texto = self._formatar_contexto(contexto)
            
            # Cria o prompt formatado
            prompt = self._criar_prompt(pergunta, contexto_texto)
            
            # Gera resposta usando o LLM
            # HuggingFacePipeline e outros modelos LangChain usam invoke()
            resposta = self._llm.invoke(prompt)
            
            # Limpa a resposta (remove o prompt se estiver incluído)
            resposta_limpa = self._limpar_resposta(resposta, prompt)
            
            return resposta_limpa
            
        except Exception as e:
            # Loga e relança exceção com contexto
            print(f"Erro ao gerar resposta: {e}")
            raise Exception(f"Falha na geração de resposta: {str(e)}") from e
    
    def _formatar_contexto(self, contexto: List[Documento]) -> str:
        """
        Formata o contexto a partir dos documentos
        
        Args:
            contexto: Lista de documentos
            
        Returns:
            String com contexto formatado
        """
        if not contexto:
            return "Nenhum contexto disponível."
        
        partes_contexto = []
        for i, doc in enumerate(contexto, 1):
            partes_contexto.append(f"[Documento {i}]\n{doc.conteudo}\n")
        
        return "\n".join(partes_contexto)
    
    def _criar_prompt(self, pergunta: str, contexto_texto: str) -> str:
        """
        Cria prompt formatado para o modelo
        
        Args:
            pergunta: Pergunta do usuário
            contexto_texto: Contexto formatado
            
        Returns:
            Prompt completo
        """
        # Template de prompt otimizado para modelos de chat
        prompt = f"""<|system|>
Você é um assistente de estudos especializado. Use o contexto fornecido para responder à pergunta de forma clara e precisa.
Se você não souber a resposta com base no contexto, diga que não sabe. Não invente informações.
<|end|>
<|user|>
Contexto:
{contexto_texto}

Pergunta: {pergunta}
<|end|>
<|assistant|>"""
        
        return prompt
    
    def _limpar_resposta(self, resposta: str, prompt: str) -> str:
        """
        Limpa a resposta removendo o prompt se estiver incluído
        
        Args:
            resposta: Resposta gerada pelo modelo
            prompt: Prompt original
            
        Returns:
            Resposta limpa
        """
        # Remove o prompt se estiver no início da resposta
        if resposta.startswith(prompt):
            resposta = resposta[len(prompt):].strip()
        
        # Remove marcadores de template se presentes
        resposta = resposta.replace("<|system|>", "")
        resposta = resposta.replace("<|user|>", "")
        resposta = resposta.replace("<|assistant|>", "")
        resposta = resposta.replace("<|end|>", "")
        
        return resposta.strip()
