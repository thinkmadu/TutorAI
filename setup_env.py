#!/usr/bin/env python3
"""
setup_env.py - Script de configuração automática do ambiente TulioAI

Este script automatiza:
- Criação de ambiente virtual
- Instalação de dependências
- Criação/carregamento de arquivo .env
- Preparação dos componentes RAG
- Inicialização da interface escolhida
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


# Cores para output no terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message):
    """Imprime cabeçalho formatado"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_step(message):
    """Imprime passo atual"""
    print(f"{Colors.OKCYAN}► {message}{Colors.ENDC}")


def print_success(message):
    """Imprime mensagem de sucesso"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Imprime mensagem de aviso"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Imprime mensagem de erro"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def get_venv_path():
    """Retorna o caminho do ambiente virtual"""
    # Verifica possíveis nomes de venv
    possible_venvs = ['.venv', 'venv', 'env']
    for venv_name in possible_venvs:
        venv_path = Path(venv_name)
        if venv_path.exists():
            return venv_path
    # Retorna o padrão se nenhum existir
    return Path('.venv')


def check_venv_exists():
    """Verifica se o ambiente virtual existe"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    
    return venv_path.exists() and python_path.exists()


def create_venv():
    """Cria um ambiente virtual"""
    print_step("Verificando ambiente virtual...")
    
    if check_venv_exists():
        venv_path = get_venv_path()
        print_success(f"Ambiente virtual já existe em '{venv_path}'")
        return True
    
    print_step("Criando ambiente virtual...")
    venv_path = Path('.venv')
    
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print_success(f"Ambiente virtual criado em '{venv_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Erro ao criar ambiente virtual: {e.stderr}")
        return False


def get_venv_python():
    """Retorna o caminho do executável Python no venv"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")


def get_venv_pip():
    """Retorna o caminho do executável pip no venv"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")


def upgrade_pip():
    """Atualiza o pip no ambiente virtual"""
    print_step("Atualizando pip...")
    
    try:
        pip_path = get_venv_pip()
        subprocess.run(
            [pip_path, "install", "--upgrade", "pip", "--quiet"],
            check=True,
            capture_output=True
        )
        print_success("Pip atualizado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print_warning(f"Aviso ao atualizar pip: {e}")
        return True  # Não é crítico


def create_env_file():
    """Cria arquivo .env com configurações padrão"""
    print_step("Verificando arquivo .env...")
    
    env_path = Path('.env')
    
    if env_path.exists():
        print_success("Arquivo .env já existe")
        return True
    
    print_step("Criando arquivo .env com configurações padrão...")
    
    default_env = """# TulioAI - Configurações de Ambiente

# Caminho para o índice FAISS
FAISS_PATH=./models/faiss_index_tutorai

# Modelo de Embeddings
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Modelo de Linguagem
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Caminho para modelo local (opcional - deixe vazio para baixar do HuggingFace)
MODEL_PATH=

# Dispositivo para execução (cpu ou cuda)
DEVICE=cpu

# Parâmetros de Geração
TEMPERATURE=0.1
MAX_NEW_TOKENS=512

# Parâmetros de Retrieval
TOP_K_RETRIEVAL=4

# Diretório com documentos Markdown
KNOWLEDGE_BASE_PATH=./data/knowledge_base

# ID da pasta do Google Drive contendo o banco FAISS (opcional)
GOOGLE_DRIVE_FAISS_FOLDER_ID=
"""
    
    try:
        env_path.write_text(default_env)
        print_success("Arquivo .env criado com sucesso")
        return True
    except Exception as e:
        print_error(f"Erro ao criar .env: {e}")
        return False


def load_env_file():
    """Carrega variáveis do arquivo .env"""
    print_step("Carregando variáveis de ambiente...")
    
    try:
        # Instala python-dotenv se necessário
        pip_path = get_venv_pip()
        venv_python = get_venv_python()
        
        # Instala python-dotenv no venv
        print_step("Instalando python-dotenv...")
        subprocess.run(
            [pip_path, "install", "python-dotenv", "--quiet"],
            check=True,
            capture_output=True
        )
        
        # Carrega .env usando o Python do venv
        load_script = """
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv()
        print("✓ Variáveis de ambiente carregadas do .env")
    else:
        print("⚠ Arquivo .env não encontrado")
except Exception as e:
    print(f"✗ Erro: {e}")
    import sys
    sys.exit(1)
"""
        
        result = subprocess.run(
            [venv_python, "-c", load_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_error(f"Erro ao carregar .env: {result.stderr}")
            return False
        
        print_success("Variáveis de ambiente carregadas")
        return True
        
    except Exception as e:
        print_error(f"Erro ao carregar .env: {e}")
        return False


def install_requirements():
    """Instala dependências do requirements.txt"""
    print_step("Verificando dependências...")
    
    req_path = Path('requirements.txt')
    
    if not req_path.exists():
        print_warning("Arquivo requirements.txt não encontrado")
        print_step("Criando requirements.txt com dependências padrão...")
        
        default_requirements = """# TulioAI - Dependências
langchain>=0.1.0
langchain-community>=0.0.20
langchain-text-splitters>=0.0.1
transformers>=4.36.0
torch>=2.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
huggingface-hub>=0.20.0
streamlit>=1.29.0
python-dotenv>=1.0.0
gdown>=4.7.1
unstructured[md]>=0.10.0
"""
        try:
            req_path.write_text(default_requirements)
            print_success("requirements.txt criado")
        except Exception as e:
            print_error(f"Erro ao criar requirements.txt: {e}")
            return False
    
    print_step("Instalando dependências (isso pode demorar alguns minutos)...")
    
    try:
        pip_path = get_venv_pip()
        result = subprocess.run(
            [pip_path, "install", "-r", "requirements.txt", "--quiet"],
            check=True,
            capture_output=True,
            text=True
        )
        print_success("Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print_error("Erro ao instalar dependências")
        print(f"{Colors.FAIL}{e.stderr}{Colors.ENDC}")
        
        # Tenta instalação sem --quiet para ver erros
        print_step("Tentando instalação detalhada...")
        try:
            subprocess.run(
                [pip_path, "install", "-r", "requirements.txt"],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


def create_directories():
    """Cria diretórios necessários"""
    print_step("Criando estrutura de diretórios...")
    
    directories = [
        'data/knowledge_base',
        'models/faiss_index_tutorai',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print_success("Diretórios criados")
    return True


def check_faiss_exists():
    """Verifica se o banco FAISS existe"""
    faiss_path = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
    
    if not os.path.exists(faiss_path):
        return False
    
    # Verifica arquivos principais do FAISS
    index_file = os.path.join(faiss_path, "index.faiss")
    pkl_file = os.path.join(faiss_path, "index.pkl")
    
    return os.path.exists(index_file) and os.path.exists(pkl_file)


def create_test_faiss_from_md():
    """Cria um banco FAISS de teste a partir de um arquivo .md"""
    print_step("Criando banco de dados de teste...")
    
    # Solicita caminho do arquivo
    while True:
        md_path = input(f"\n{Colors.BOLD}Caminho do arquivo .md de teste: {Colors.ENDC}").strip()
        
        if os.path.exists(md_path) and md_path.endswith('.md'):
            break
        else:
            print_warning("Arquivo não encontrado ou não é .md. Tente novamente.")
    
    try:
        venv_python = get_venv_python()
        faiss_path = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
        embeddings_model = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        print_step(f"Processando arquivo: {md_path}")
        
        # Script para criar o índice
        create_script = f"""
import os
import sys
from pathlib import Path

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    print("Carregando arquivo Markdown...")
    loader = UnstructuredMarkdownLoader("{md_path}")
    documents = loader.load()
    print(f"✓ Arquivo carregado: {{len(documents)}} documento(s)")
    
    print("Dividindo em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ {{len(chunks)}} chunk(s) criado(s)")
    
    print("Carregando modelo de embeddings: {embeddings_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name="{embeddings_model}",
        model_kwargs={{'device': 'cpu'}}
    )
    print("✓ Modelo de embeddings carregado")
    
    print("Criando índice FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✓ Índice FAISS criado")
    
    print("Salvando índice em: {faiss_path}")
    Path("{faiss_path}").parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local("{faiss_path}")
    print("✓ Índice salvo com sucesso")
    
    print("\\n✓ Banco de dados de teste criado com sucesso!")
    
except Exception as e:
    print(f"✗ Erro ao criar banco: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        result = subprocess.run(
            [venv_python, "-c", create_script],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Erro ao criar banco de teste:")
            print(result.stderr)
            return False
        
        print_success(f"Banco de dados de teste criado em '{faiss_path}'")
        return True
        
    except Exception as e:
        print_error(f"Erro ao criar banco de teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_faiss_from_drive():
    """Baixa o banco FAISS do Google Drive"""
    print_step("Baixando banco de dados do Google Drive...")
    
    drive_folder_id = os.getenv("GOOGLE_DRIVE_FAISS_FOLDER_ID", "")
    
    if not drive_folder_id:
        print_error("GOOGLE_DRIVE_FAISS_FOLDER_ID não está definido no .env")
        print("Por favor, adicione o ID da pasta do Google Drive ao arquivo .env")
        print("Exemplo: GOOGLE_DRIVE_FAISS_FOLDER_ID=1a2b3c4d5e6f7g8h9i0j")
        return False
    
    try:
        venv_python = get_venv_python()
        faiss_path = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
        
        # Script para baixar do Drive
        download_script = f"""
import os
import sys
from pathlib import Path

try:
    import gdown
    
    print("Baixando pasta do Google Drive...")
    print(f"ID da pasta: {drive_folder_id}")
    print(f"Destino: {faiss_path}")
    
    # Cria diretório de destino
    Path("{faiss_path}").mkdir(parents=True, exist_ok=True)
    
    # Baixa a pasta
    url = f"https://drive.google.com/drive/folders/{drive_folder_id}"
    gdown.download_folder(url, output="{faiss_path}", quiet=False, use_cookies=False)
    
    # Verifica se os arquivos foram baixados
    index_file = os.path.join("{faiss_path}", "index.faiss")
    pkl_file = os.path.join("{faiss_path}", "index.pkl")
    
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print("\\n✓ Banco de dados baixado com sucesso!")
    else:
        print("\\n✗ Arquivos FAISS não encontrados após download")
        sys.exit(1)
    
except ImportError:
    print("✗ Biblioteca gdown não está instalada")
    print("Isso não deveria acontecer pois gdown já foi instalado no venv.")
    print("Tente reinstalar as dependências: .venv/bin/pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Erro ao baixar do Drive: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        result = subprocess.run(
            [venv_python, "-c", download_script],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Erro ao baixar do Google Drive:")
            print(result.stderr)
            return False
        
        print_success(f"Banco de dados baixado para '{faiss_path}'")
        return True
        
    except Exception as e:
        print_error(f"Erro ao baixar do Drive: {e}")
        import traceback
        traceback.print_exc()
        return False


def ask_faiss_creation_method():
    """Pergunta ao usuário como criar o banco FAISS"""
    print_header("Banco de Dados Vetorial Não Encontrado")
    
    print("O banco de dados vetorial FAISS não foi encontrado.")
    print("Deseja:")
    print(f"  {Colors.OKBLUE}1{Colors.ENDC} - Criar um banco de teste a partir de um arquivo .md fornecido")
    print(f"  {Colors.OKBLUE}2{Colors.ENDC} - Baixar o banco de dados da pasta do Google Drive")
    
    while True:
        choice = input(f"\n{Colors.BOLD}Escolha (1/2): {Colors.ENDC}").strip()
        
        if choice in ['1', '2']:
            return choice
        else:
            print_warning("Opção inválida. Digite 1 ou 2.")


def load_rag_components():
    """Carrega e inicializa componentes do RAG"""
    print_header("Inicializando Componentes RAG")
    
    try:
        venv_python = get_venv_python()
        
        # 1. Verifica se o banco FAISS existe
        # Primeiro carrega o .env usando subprocess para pegar o FAISS_PATH
        check_script = """
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("FAISS_PATH", "./models/faiss_index_tutorai"))
"""
        
        result = subprocess.run(
            [venv_python, "-c", check_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            faiss_path = result.stdout.strip()
            os.environ["FAISS_PATH"] = faiss_path
        else:
            faiss_path = "./models/faiss_index_tutorai"
            
        print_step("Verificando banco de dados FAISS...")
        
        if not check_faiss_exists():
            print_warning("Banco FAISS não encontrado!")
            
            # Pergunta o método de criação
            choice = ask_faiss_creation_method()
            
            if choice == '1':
                # Criar banco de teste
                if not create_test_faiss_from_md():
                    print_error("Falha ao criar banco de teste")
                    return False
            elif choice == '2':
                # Baixar do Drive
                if not download_faiss_from_drive():
                    print_error("Falha ao baixar banco do Drive")
                    return False
            
            # Verifica novamente se o banco foi criado/baixado
            if not check_faiss_exists():
                print_error("Banco FAISS ainda não está disponível")
                return False
        else:
            print_success("Banco FAISS encontrado!")
        
        # 2. Agora carrega os componentes RAG efetivamente
        print_step("Carregando componentes RAG...")
        
        load_script = """
import sys
import os
from dotenv import load_dotenv

load_dotenv()

print("Verificando imports...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    print("✓ Imports OK")
except ImportError as e:
    print(f"✗ Erro de import: {e}")
    sys.exit(1)

# Configurações
FAISS_PATH = os.getenv("FAISS_PATH", "./models/faiss_index_tutorai")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(f"\\nConfigurações:")
print(f"  FAISS_PATH: {FAISS_PATH}")
print(f"  EMBEDDINGS_MODEL: {EMBEDDINGS_MODEL}")
print(f"  MODEL_NAME: {MODEL_NAME}")

# Carrega embeddings
print(f"\\nCarregando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={'device': 'cpu'}
)
print("✓ Embeddings carregados")

# Carrega índice FAISS
print(f"\\nCarregando índice FAISS de: {FAISS_PATH}")
db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
print("✓ Índice FAISS carregado")

# Testa o índice
print("\\nTestando índice FAISS...")
results = db.similarity_search("teste", k=1)
print(f"✓ Teste OK - {len(results)} resultado(s)")

print("\\n✓ Componentes RAG carregados com sucesso!")
print("\\nO sistema está pronto para uso!")
"""
        
        result = subprocess.run(
            [venv_python, "-c", load_script],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print_error("Erro ao carregar componentes RAG:")
            print(result.stderr)
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Erro ao carregar componentes: {e}")
        import traceback
        traceback.print_exc()
        return False


def ask_interface_choice():
    """Pergunta ao usuário qual interface deseja usar"""
    print_header("Escolha a Interface")
    
    print("Como você deseja iniciar o TulioAI?")
    print(f"  {Colors.OKBLUE}1{Colors.ENDC} - Interface CLI (Terminal)")
    print(f"  {Colors.OKBLUE}2{Colors.ENDC} - Interface Streamlit (Web)")
    print(f"  {Colors.OKBLUE}3{Colors.ENDC} - Apenas configurar (sair)")
    
    while True:
        choice = input(f"\n{Colors.BOLD}Escolha (1/2/3): {Colors.ENDC}").strip()
        
        if choice in ['1', '2', '3']:
            return choice
        else:
            print_warning("Opção inválida. Digite 1, 2 ou 3.")


def launch_interface(choice):
    """Inicia a interface escolhida"""
    venv_python = get_venv_python()
    
    if choice == '1':
        print_header("Iniciando Interface CLI")
        print_step("Carregando sistema...")
        
        try:
            subprocess.run(
                [venv_python, "main.py", "--interface", "cli"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print_error(f"Erro ao executar CLI: {e}")
            return False
        except KeyboardInterrupt:
            print("\n")
            print_success("CLI encerrado pelo usuário")
        
    elif choice == '2':
        print_header("Iniciando Interface Streamlit")
        print_step("Abrindo aplicação web...")
        print_warning("Pressione Ctrl+C para encerrar o servidor Streamlit")
        
        try:
            subprocess.run(
                [venv_python, "main.py", "--interface", "streamlit"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print_error(f"Erro ao executar Streamlit: {e}")
            return False
        except KeyboardInterrupt:
            print("\n")
            print_success("Streamlit encerrado pelo usuário")
    
    elif choice == '3':
        print_success("Configuração concluída!")
        print("\nPara usar o TulioAI posteriormente, execute:")
        print(f"  {Colors.OKBLUE}python main.py --interface cli{Colors.ENDC}")
        print(f"  {Colors.OKBLUE}python main.py --interface streamlit{Colors.ENDC}")
    
    return True


def main():
    """Função principal do script de setup"""
    print_header("TulioAI - Setup Automático do Ambiente")
    
    try:
        # 1. Criar/verificar ambiente virtual
        if not create_venv():
            print_error("Falha ao criar ambiente virtual")
            return 1
        
        # 2. Atualizar pip
        upgrade_pip()
        
        # 3. Criar arquivo .env
        if not create_env_file():
            print_error("Falha ao criar arquivo .env")
            return 1
        
        # 4. Carregar .env
        if not load_env_file():
            print_error("Falha ao carregar .env")
            return 1
        
        # 5. Criar diretórios
        if not create_directories():
            print_error("Falha ao criar diretórios")
            return 1
        
        # 6. Instalar dependências
        if not install_requirements():
            print_error("Falha ao instalar dependências")
            return 1
        
        # 7. Verificar componentes RAG
        if not load_rag_components():
            print_warning("Componentes RAG não puderam ser totalmente verificados")
            print_warning("Você pode precisar criar o índice FAISS manualmente")
        
        # 8. Perguntar interface
        choice = ask_interface_choice()
        
        # 9. Iniciar interface escolhida
        launch_interface(choice)
        
        print("\n")
        print_success("Setup concluído com sucesso!")
        return 0
        
    except KeyboardInterrupt:
        print("\n")
        print_warning("Setup cancelado pelo usuário")
        return 130
    except Exception as e:
        print_error(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
