# TutorAI (ou TúlioAI)

Um assistente de estudos baseado em IA que responde perguntas com base em uma base de conhecimento em arquivos Markdown.

## Tecnologias utilizadas

*   Python
*   LangChain
*   FAISS
*   Gradio
*   Transformers
*   Sentence Transformers
*   Hugging Face Models (ex: TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## Como rodar

1.  Clone este repositório.
2.  Navegue até o diretório do projeto.
3.  (Opcional, mas recomendado) Crie um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
4.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Importante:** Este projeto espera que você tenha:
    *   Um banco de dados vetorial FAISS (já processado) salvo em algum lugar.
    *   O caminho para esse banco de dados seja ajustado no código (`caminho_faiss_index_completo`).
    *   O modelo de linguagem (ex: TinyLlama) seja baixado automaticamente pelo `transformers` ou já esteja disponível localmente.
6.  Execute o script:
    ```bash
    python main.py
    ```
    Um link público será gerado para acessar a interface web.

## Observações

*   Este projeto foi inicialmente desenvolvido no Google Colab Free.
*   O desempenho pode variar dependendo do modelo escolhido e dos recursos da máquina onde for executado.