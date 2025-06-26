import logging
import json
import requests
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ollama_api_base = "http://localhost:11434/api"

def main():

    # Warm up the model to ensure it is ready for use
    logger.info("Starting the Ollama server...")
    start_time = time.time()
    response = requests.post(
        f"{ollama_api_base}/chat",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3.2",
            "messages": []
        }
    )
    end_time = time.time()
    logger.info("Model is warm up. Took %.4f seconds", end_time - start_time)


    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = requests.post(
        f"{ollama_api_base}/chat",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3.2",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "stream": False
        }
    )
    end_time = time.time()
    print(response.json()["message"]["content"])
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
    response = requests.post(
        f"{ollama_api_base}/chat",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3.2",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "stream": True
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            chunk_response = json.loads(line.decode('utf-8'))
            print(chunk_response["message"]["content"], end='', flush=True)
    end_time = time.time()
    print()
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


if __name__ == "__main__":
    main()