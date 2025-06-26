import logging
import time
from litgpt import LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():

    # Load the model
    # FIXME: cannot load the model with quantization
    logger.info("Starting the LitGPT server...")
    start_time = time.time()
    llm = LLM.load("meta-llama/Llama-3.2-3B-Instruct")
    # llm.distribute(quantize="bnb.nf4", precision="16-true")
    end_time = time.time()
    logger.info("Model is loaded and ready for use. Took %.4f seconds loading", end_time - start_time)


    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = llm.generate(
        "Hello, how are you?",
    )
    end_time = time.time()
    print(response)
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    start_time = time.time()
    for response_stream in llm.generate(
        "Hello, how are you?",
        stream=True
    ):
        print(response_stream, end="", flush=True)
    end_time = time.time()
    print()  # Ensure the output ends with a newline
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


    # TODO: Simple chat completion from batch


if __name__ == "__main__":
    main()