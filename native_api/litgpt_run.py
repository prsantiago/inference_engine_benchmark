import logging
from litgpt import LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():

    logger.info("Staring the LitGPT server...")

    # Load the model
    llm = LLM.load("meta-llama/Llama-3.2-3B")

    logger.info("Model is loaded and ready for use.")

    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    response = llm.generate(
        "Hello, how are you?",
    )
    print(response)

    # Simple chat completion streaming
    logger.info("Streaming response for: 'Hello, how are you?'")
    for response_stream in llm.generate(
        "Hello, how are you?",
        stream=True
    ):
        print(response_stream, end="", flush=True)

if __name__ == "__main__":
    main()