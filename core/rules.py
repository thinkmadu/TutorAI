"""
Regras de Negócio Puras do TúlioAI
Lógica de domínio independente de frameworks
"""

from typing import List
from .entities import Fonte, Resposta


class RegrasDeDominio:
    """Regras de negócio aplicáveis ao sistema"""
    
    @staticmethod
    def validar_pergunta(texto: str) -> bool:
        """Valida se uma pergunta é válida"""
        if not texto or not texto.strip():
            return False
        if len(texto.strip()) < 3:
            return False
        return True
    
    @staticmethod
    def filtrar_fontes_relevantes(
        fontes: List[Fonte], 
        threshold: float = 0.5
    ) -> List[Fonte]:
        """Filtra fontes com base em um threshold de relevância"""
        return [f for f in fontes if f.relevancia_score >= threshold]
    
    @staticmethod
    def ordenar_fontes_por_relevancia(fontes: List[Fonte]) -> List[Fonte]:
        """Ordena fontes por relevância (score decrescente)"""
        return sorted(fontes, key=lambda f: f.relevancia_score, reverse=True)
    
    @staticmethod
    def calcular_confianca_resposta(fontes: List[Fonte]) -> float:
        """
        Calcula nível de confiança da resposta baseado nas fontes
        Retorna valor entre 0.0 e 1.0
        """
        if not fontes:
            return 0.0
        
        # Média dos scores de relevância das fontes
        media_relevancia = sum(f.relevancia_score for f in fontes) / len(fontes)
        
        # Penaliza se houver poucas fontes
        fator_quantidade = min(len(fontes) / 3, 1.0)
        
        confianca = media_relevancia * (0.7 + 0.3 * fator_quantidade)
        return min(confianca, 1.0)
    
    @staticmethod
    def deve_indicar_incerteza(resposta: Resposta) -> bool:
        """Determina se a resposta deve indicar incerteza ao usuário"""
        return resposta.confianca < 0.6 or len(resposta.fontes) < 2
    
    @staticmethod
    def limitar_tamanho_resposta(texto: str, max_tokens: int = 512) -> str:
        """
        Limita o tamanho da resposta (aproximação por palavras)
        1 token ≈ 0.75 palavras
        """
        max_palavras = int(max_tokens * 0.75)
        palavras = texto.split()
        
        if len(palavras) <= max_palavras:
            return texto
        
        return ' '.join(palavras[:max_palavras]) + "..."
