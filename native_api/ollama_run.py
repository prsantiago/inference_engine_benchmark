import logging
from ollama import chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():

    logger.info("Starting the Ollama server...")

    # Warm-up the model to ensure it is ready for use
    _ = chat(model="llama3.2", messages=[], keep_alive=-1)
    
    logger.info("Model is warmed up and ready for use.")

    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    )
    print(response["message"]["content"])

    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    for response_stream in chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        stream=True
    ):
        print(response_stream["message"]["content"], end="", flush=True)



if __name__ == "__main__":
    main()
