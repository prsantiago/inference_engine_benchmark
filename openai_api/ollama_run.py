import logging
import time
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ollama_api_key = "empty"
ollama_api_base = "http://localhost:11434/v1"

def main():

    ollama_client = OpenAI(
        api_key=ollama_api_key,
        base_url=ollama_api_base
    )

    # Load the model to ensure it is ready for use
    logger.info("Starting the Ollama server...")
    start_time = time.time()
    response = ollama_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": ""
            }
        ]
    )
    end_time = time.time()
    logger.info("Model is warm up. Took %.4f seconds", end_time - start_time)


    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = ollama_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    end_time = time.time()
    print(response.choices[0].message.content)
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
    for response in ollama_client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        stream=True
    ):
        if response.choices and response.choices[0].delta.content is not None:
            print(response.choices[0].delta.content, end='', flush=True)
    end_time = time.time()
    print()  # Ensure the output ends with a newline
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


if __name__ == "__main__":
    main()