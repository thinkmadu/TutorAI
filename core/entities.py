"""
Entidades de Domínio do TúlioAI
Define as estruturas de dados fundamentais do sistema
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Pergunta:
    """Representa uma pergunta feita pelo usuário"""
    
    texto: str


@dataclass
class Documento:
    """Representa um documento carregado da base de conhecimento"""
    
    conteudo: str
    metadados: dict = field(default_factory=dict)


@dataclass
class Fonte:
    """Representa uma fonte de informação (caminho do arquivo .md original)"""
    
    caminho: str
    relevancia_score: Optional[float] = None


@dataclass
class Resposta:
    """Representa uma resposta gerada pelo sistema"""
    
    texto: str
    fontes: List[Fonte] = field(default_factory=list)
