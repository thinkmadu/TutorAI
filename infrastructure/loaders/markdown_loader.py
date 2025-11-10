"""
Markdown Loader - Carrega e processa arquivos Markdown
"""

from pathlib import Path
from typing import List, Dict, Optional
import re
from core.entities import Documento


class MarkdownLoader:
    """Carrega arquivos Markdown e extrai metadados"""
    
    def __init__(self, base_path: str):
        """
        Inicializa o loader
        
        Args:
            base_path: Caminho base para buscar arquivos .md
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {base_path}")
    
    def carregar_arquivo(self, caminho: str) -> Documento:
        """
        Carrega um único arquivo Markdown
        
        Args:
            caminho: Caminho do arquivo
            
        Returns:
            Documento carregado
        """
        caminho_completo = Path(caminho)
        
        if not caminho_completo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        
        with open(caminho_completo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        metadados = self._extrair_metadados(conteudo, caminho_completo)
        
        return Documento(
            conteudo=conteudo,
            caminho=str(caminho_completo),
            metadados=metadados,
            titulo=metadados.get('titulo')
        )
    
    def carregar_todos(self, recursivo: bool = True) -> List[Documento]:
        """
        Carrega todos os arquivos .md do diretório base
        
        Args:
            recursivo: Se True, busca recursivamente em subdiretórios
            
        Returns:
            Lista de documentos carregados
        """
        documentos = []
        
        if recursivo:
            arquivos = self.base_path.rglob("*.md")
        else:
            arquivos = self.base_path.glob("*.md")
        
        for arquivo in arquivos:
            try:
                doc = self.carregar_arquivo(str(arquivo))
                documentos.append(doc)
            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {e}")
                continue
        
        return documentos
    
    def _extrair_metadados(
        self, 
        conteudo: str, 
        caminho: Path
    ) -> Dict[str, any]:
        """
        Extrai metadados do arquivo Markdown
        
        Args:
            conteudo: Conteúdo do arquivo
            caminho: Caminho do arquivo
            
        Returns:
            Dicionário com metadados
        """
        metadados = {
            'nome_arquivo': caminho.name,
            'caminho_relativo': str(caminho.relative_to(self.base_path)),
            'tamanho': len(conteudo),
        }
        
        # Extrai primeiro heading como título
        titulo_match = re.search(r'^#\s+(.+)$', conteudo, re.MULTILINE)
        if titulo_match:
            metadados['titulo'] = titulo_match.group(1).strip()
        else:
            metadados['titulo'] = caminho.stem
        
        # Extrai todos os headings
        headings = re.findall(r'^#{1,6}\s+(.+)$', conteudo, re.MULTILINE)
        metadados['headings'] = headings
        
        # Conta elementos
        metadados['num_linhas'] = conteudo.count('\n') + 1
        metadados['num_palavras'] = len(conteudo.split())
        
        return metadados
    
    @staticmethod
    def normalizar_texto(texto: str) -> str:
        """
        Normaliza texto removendo formatação Markdown excessiva
        
        Args:
            texto: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        # Remove múltiplas linhas vazias
        texto = re.sub(r'\n{3,}', '\n\n', texto)
        
        # Remove espaços extras
        texto = re.sub(r' {2,}', ' ', texto)
        
        # Remove formatação de código inline excessiva
        # (mantém código mas limpa formatação)
        texto = re.sub(r'`{3,}', '```', texto)
        
        return texto.strip()


class TextSplitter:
    """Divide texto em chunks para processamento"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separadores: Optional[List[str]] = None
    ):
        """
        Inicializa o splitter
        
        Args:
            chunk_size: Tamanho máximo de cada chunk (em caracteres)
            chunk_overlap: Sobreposição entre chunks
            separadores: Lista de separadores (padrão: paragráfos, linhas, espaços)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separadores = separadores or ["\n\n", "\n", " ", ""]
    
    def dividir(self, texto: str) -> List[str]:
        """
        Divide texto em chunks
        
        Args:
            texto: Texto a dividir
            
        Returns:
            Lista de chunks
        """
        if len(texto) <= self.chunk_size:
            return [texto]
        
        chunks = []
        chunks = self._dividir_recursivo(texto, self.separadores)
        
        return chunks
    
    def _dividir_recursivo(
        self, 
        texto: str, 
        separadores: List[str]
    ) -> List[str]:
        """Divide texto recursivamente usando diferentes separadores"""
        if not separadores:
            return self._dividir_por_tamanho(texto)
        
        separador = separadores[0]
        separadores_restantes = separadores[1:]
        
        if separador:
            partes = texto.split(separador)
        else:
            partes = list(texto)
        
        chunks = []
        chunk_atual = ""
        
        for parte in partes:
            if len(chunk_atual) + len(parte) + len(separador) <= self.chunk_size:
                if chunk_atual:
                    chunk_atual += separador + parte
                else:
                    chunk_atual = parte
            else:
                if chunk_atual:
                    chunks.append(chunk_atual)
                
                if len(parte) > self.chunk_size:
                    # Parte muito grande, dividir com próximo separador
                    sub_chunks = self._dividir_recursivo(parte, separadores_restantes)
                    chunks.extend(sub_chunks)
                    chunk_atual = ""
                else:
                    chunk_atual = parte
        
        if chunk_atual:
            chunks.append(chunk_atual)
        
        return chunks
    
    def _dividir_por_tamanho(self, texto: str) -> List[str]:
        """Divide texto forçadamente por tamanho quando não há separadores"""
        chunks = []
        inicio = 0
        
        while inicio < len(texto):
            fim = inicio + self.chunk_size
            chunks.append(texto[inicio:fim])
            inicio = fim - self.chunk_overlap
        
        return chunks
    
    def dividir_documentos(self, documentos: List[Documento]) -> List[Documento]:
        """
        Divide múltiplos documentos em chunks
        
        Args:
            documentos: Lista de documentos
            
        Returns:
            Lista de documentos com chunks preenchidos
        """
        for doc in documentos:
            doc.chunks = self.dividir(doc.conteudo)
        
        return documentos
