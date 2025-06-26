import json
import requests
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

litgpt_api_base = "http://localhost:8000"

def main():

    # Simple chat completion
    logger.info("Sending 'hello, how are you?' to the model...")
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "prompt": "Hello, how are you?",
        },
        stream = False
    )
    for line in response.iter_lines(decode_unicode=True):
        if line:
            chunck_response = json.loads(line)
            print(chunck_response["output"], end="", flush=True)
    end_time = time.time()
    print()
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "prompt": "Hello, how are you?",
        },
        stream = True
    )
    for line in response.iter_lines(decode_unicode=True):
        if line:
            chunck_response = json.loads(line)
            print(chunck_response["output"], end="", flush=True)
    end_time = time.time()
    print()
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


if __name__ == "__main__":
    main()