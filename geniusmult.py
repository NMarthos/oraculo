import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
        'chat': ChatOpenAI
    }
}

MEMORIA = ConversationBufferMemory()


def carrega_arquivos(tipo_arquivo, arquivos):
    documentos = []

    if tipo_arquivo in ['Site', 'Youtube']:
        for item in arquivos:
            if tipo_arquivo == 'Site':
                documentos.append(carrega_site(item))
            if tipo_arquivo == 'Youtube':
                documentos.append(carrega_youtube(item))

    else:  # Pdf, Csv, Txt
        for arquivo in arquivos:
            ext = {
                'Pdf': '.pdf',
                'Csv': '.csv',
                'Txt': '.txt'
            }[tipo_arquivo]

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp:
                temp.write(arquivo.read())
                nome_temp = temp.name

            if tipo_arquivo == 'Pdf':
                documentos.append(carrega_pdf(nome_temp))
            elif tipo_arquivo == 'Csv':
                documentos.append(carrega_csv(nome_temp))
            elif tipo_arquivo == 'Txt':
                documentos.append(carrega_txt(nome_temp))

    return "\n".join(documentos)


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivos):
    documentos = carrega_arquivos(tipo_arquivo, arquivos)

    system_message = f'''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas de um documento(s) do tipo {tipo_arquivo}:

    ####
    {documentos}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue", 
    sugira ao usuário carregar novamente o Oráculo!'''

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain


def pagina_chat():
    st.header('🤖 Bem-vindo ao Oráculo', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o oráculo')
    if input_usuario:
        st.chat_message('human').markdown(input_usuario)
        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
        }))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria


def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)

        arquivos = []
        if tipo_arquivo in ['Site', 'Youtube']:
            entrada = st.text_area(f'Digite uma ou mais URLs de {tipo_arquivo}, uma por linha')
            arquivos = [linha.strip() for linha in entrada.splitlines() if linha.strip()]
        else:
            extensoes = {
                'Pdf': ['.pdf'],
                'Csv': ['.csv'],
                'Txt': ['.txt']
            }[tipo_arquivo]
            arquivos = st.file_uploader('Faça upload de um ou mais arquivos', type=extensoes, accept_multiple_files=True)

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}', '')
        )
        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Oráculo', use_container_width=True):
        if not arquivos:
            st.warning('Adicione pelo menos um arquivo ou URL antes de continuar.')
        else:
            carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivos)

    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
