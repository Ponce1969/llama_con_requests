import requests
import json
import logging
import sys
from colorama import init, Fore, Style
import textwrap

# Inicializa colorama
init(autoreset=True)

# Configuración de la API key
api_key_file = "api_key.txt"

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define un manejo de errores personalizado
def handle_error(error, exit=True):
    logger.error(f"Error: {error}")
    print(f"Error: {error}")
    if exit:
        sys.exit(1)

# Define una función para validar la respuesta de la API
def validate_response(response):
    if response.status_code == 200:
        objeto = response.json()
        return objeto["choices"][0]["message"]["content"]
    else:
        handle_error(f"Error: {response.status_code} - {response.text}")
        return None
    

# Main function
def main():
    try:
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        handle_error("La API key no fue encontrada en el archivo especificado.")
        return

    # Texto de entrada para LLama 3.1
    input_text = input("¿Que quieres preguntarle a LLama 3? : ")

    # Convierte el texto de entrada a JSON
    input_data = {
        "messages": [
            {
                "role": "system",
                "content": "Eres un experto en Python y un ingeniero en software con un gran conocimiento en este lenguaje, estas trabajando en un proyecto de desarrollo de software en python y debes entrenar a los juniors que consultan contigo."
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        "model": "llama-3.1-70b-versatile"
    }
   
    # Encabezados de la solicitud API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Realizar la solicitud API
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', json=input_data, headers=headers)

        # Validate the response
        response_json = validate_response(response)

        if response_json:
            # Formatear la respuesta con Markdown y colorama
            wrapped_response = textwrap.fill(response_json, width=90)
            formatted_response = f"{Fore.GREEN}{Style.BRIGHT}Respuesta de LLama 3.1:{Style.RESET_ALL}\n\n{Fore.CYAN}{wrapped_response}{Style.RESET_ALL}"
            print(formatted_response, end='\n')
        else:
            handle_error("Error: La respuesta no es válida")

    except requests.exceptions.RequestException as e:
        handle_error(f"Error al realizar la solicitud API: {e}")

if __name__ == '__main__':
    main()