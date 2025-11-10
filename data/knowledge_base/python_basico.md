# Python - Guia Completo

## O que é Python?

Python é uma linguagem de programação de alto nível, interpretada, de tipagem dinâmica e multiparadigma. Foi criada por Guido van Rossum e lançada em 1991.

### Características Principais

- **Sintaxe Clara**: Python enfatiza a legibilidade do código
- **Interpretada**: Não requer compilação prévia
- **Multiparadigma**: Suporta programação procedural, orientada a objetos e funcional
- **Tipagem Dinâmica**: Tipos são determinados em tempo de execução
- **Grande Biblioteca Padrão**: Vem com muitos módulos úteis

## Conceitos Básicos

### Variáveis

Em Python, variáveis são criadas quando você atribui um valor a elas. Não é necessário declarar o tipo.

```python
nome = "João"
idade = 25
altura = 1.75
ativo = True
```

### Tipos de Dados

Python possui vários tipos de dados built-in:

- **int**: Números inteiros
- **float**: Números decimais
- **str**: Strings (texto)
- **bool**: Booleanos (True/False)
- **list**: Listas mutáveis
- **tuple**: Tuplas imutáveis
- **dict**: Dicionários (chave-valor)
- **set**: Conjuntos

### Estruturas de Controle

#### If/Else

```python
if idade >= 18:
    print("Maior de idade")
else:
    print("Menor de idade")
```

#### Loops

**For Loop:**
```python
for i in range(5):
    print(i)
```

**While Loop:**
```python
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

### Funções

Funções são definidas usando a palavra-chave `def`:

```python
def saudacao(nome):
    return f"Olá, {nome}!"

mensagem = saudacao("Maria")
print(mensagem)
```

### Classes e Objetos

Python suporta programação orientada a objetos:

```python
class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade
    
    def apresentar(self):
        return f"Meu nome é {self.nome} e tenho {self.idade} anos"

pessoa = Pessoa("Carlos", 30)
print(pessoa.apresentar())
```

## Bibliotecas Populares

### NumPy

Biblioteca para computação numérica e arrays multidimensionais.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())
```

### Pandas

Biblioteca para análise e manipulação de dados.

```python
import pandas as pd

df = pd.DataFrame({
    'nome': ['Ana', 'Bruno', 'Carlos'],
    'idade': [25, 30, 35]
})
```

### Requests

Biblioteca para fazer requisições HTTP.

```python
import requests

response = requests.get('https://api.example.com/data')
data = response.json()
```

## Boas Práticas

### PEP 8

PEP 8 é o guia de estilo oficial para código Python. Principais recomendações:

- Use 4 espaços para indentação
- Limite linhas a 79 caracteres
- Use snake_case para funções e variáveis
- Use PascalCase para classes
- Adicione espaços ao redor de operadores

### Docstrings

Documente suas funções e classes:

```python
def calcular_area_circulo(raio):
    """
    Calcula a área de um círculo.
    
    Args:
        raio (float): O raio do círculo
    
    Returns:
        float: A área do círculo
    """
    import math
    return math.pi * raio ** 2
```

### Type Hints

Use type hints para melhorar a legibilidade:

```python
def soma(a: int, b: int) -> int:
    return a + b
```

## Virtual Environments

Ambientes virtuais isolam dependências de projetos:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate

# Instalar pacotes
pip install requests

# Salvar dependências
pip freeze > requirements.txt
```

## Exceções

Trate erros com try/except:

```python
try:
    resultado = 10 / 0
except ZeroDivisionError:
    print("Erro: Divisão por zero")
except Exception as e:
    print(f"Erro inesperado: {e}")
finally:
    print("Bloco finally sempre executa")
```

## List Comprehensions

Forma concisa de criar listas:

```python
# Tradicional
quadrados = []
for i in range(10):
    quadrados.append(i ** 2)

# List comprehension
quadrados = [i ** 2 for i in range(10)]

# Com condição
pares = [i for i in range(10) if i % 2 == 0]
```

## Decorators

Decorators modificam o comportamento de funções:

```python
def meu_decorator(func):
    def wrapper():
        print("Antes da função")
        func()
        print("Depois da função")
    return wrapper

@meu_decorator
def dizer_ola():
    print("Olá!")

dizer_ola()
```

## Módulos e Packages

Organize código em módulos e pacotes:

```
meu_projeto/
│
├── main.py
└── utilitarios/
    ├── __init__.py
    ├── matematica.py
    └── texto.py
```

Importar:
```python
from utilitarios.matematica import somar
from utilitarios import texto
```

## Recursos para Aprender

- Documentação oficial: https://docs.python.org
- Tutorial interativo: https://www.learnpython.org
- Exercícios: https://exercism.org/tracks/python
- Comunidade: https://www.python.org/community
