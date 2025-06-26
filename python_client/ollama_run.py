import logging
import time
from ollama import chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():

    # Warm-up the model to ensure it is ready for use
    logger.info("Starting the Ollama server...")
    start_time = time.time()
    _ = chat(model="llama3.2", messages=[], keep_alive=1)
    end_time = time.time()
    logger.info("Model is warmed up and ready for use. Took %.4f seconds to warm up", end_time - start_time)


    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    )
    end_time = time.time()
    print(response["message"]["content"])
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
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
    end_time = time.time()
    print()  # Ensure the output ends with a newline
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


    # TODO: Simple chat completion from batch


if __name__ == "__main__":
    main()
