import logging
import requests
import time
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

vllm_api_key = "empty"
vllm_api_base = "http://localhost:8000"

def main():

    vllm_client = OpenAI(
        api_key=vllm_api_key,
        base_url=f"{vllm_api_base}/v1"
    )

    # Load the model to ensure it is ready for use
    logger.info("Starting the vLLM server...")
    start_time = time.time()
    response = requests.get(
        f"{vllm_api_base}/health",
    )
    end_time = time.time()
    if response.status_code != 200:
        logger.error("Failed to start the vLLM server. Status code: %d", response.status_code)
        return
    logger.info("vLLM server is running. Took %.4f seconds", end_time - start_time)


    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = vllm_client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        max_tokens=50
    )
    end_time = time.time()
    print(response.choices[0].message.content)
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
    for response in vllm_client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        max_tokens=50,
        stream=True
    ):
        if response.choices and response.choices[0].delta.content is not None:
            print(response.choices[0].delta.content, end="", flush=True)
    end_time = time.time()
    print()
    logger.info("Response generated in %.4f seconds", end_time - start_time)


if __name__ == "__main__":
    main()