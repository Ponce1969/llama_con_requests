import logging
import sys
from colorama import init, Fore, Style
import textwrap
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import config  # Importar el archivo de configuración

# Inicializa colorama
init(autoreset=True)

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter(config.LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

def handle_error(error, exit=True):
    logger.error(f"Error: {error}")
    print(f"Error: {error}")
    if exit:
        sys.exit(1)

def format_response(response):
    # Ajustar el texto a un ancho de 90 caracteres y agregar colores
    wrapped_response = textwrap.fill(response,replace_whitespace=False, width=100)
    formatted_response = f"{Fore.GREEN}{Style.BRIGHT}Respuesta de LLama 3.1:{Style.RESET_ALL}\n\n{Fore.CYAN}{wrapped_response}{Style.RESET_ALL}"
    return formatted_response

def main():
    """
    Función principal que inicializa un chatbot utilizando el modelo LLama 3.1 e interactúa con el usuario.
    La función realiza los siguientes pasos:
    1. Lee la clave de la API desde un archivo.
    2. Crea una instancia de la clase ChatGroq con la clave de la API y el nombre del modelo.
    3. Configura el sistema de inicio y la memoria conversacional.
    4. Entra en un bucle para recibir preguntas del usuario y generar respuestas utilizando LLama 3.1.
    5. Imprime la respuesta en la consola.
    Parámetros:
        Ninguno
    Retorna:
        Ninguno
    """
    
    try:
        with open(config.API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        handle_error("La API key no fue encontrada en el archivo especificado.")
        return

    groq_chat = ChatGroq(
        groq_api_key=api_key, 
        model_name=config.MODEL_NAME
    )

    system_prompt = "Eres un experto en Python y un ingeniero en software con un gran conocimiento en este lenguaje, estas trabajando en un proyecto de desarrollo de software en python y debes entrenar a los juniors que consultan contigo."
    
    memory = ConversationBufferWindowMemory(k=config.CONVERSATIONAL_MEMORY_LENGTH, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("¿Qué quieres preguntarle a LLama 3? (escribe 'salir' para terminar): ")
        if user_question.lower() in ['salir', 'exit']:
            print("Saliendo del chat. ¡Hasta luego!")
            break

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        response = conversation.predict(human_input=user_question)
        formatted_response = format_response(response)
        print(formatted_response, end='\n\n')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        handle_error(f"Error inesperado: {e}") 