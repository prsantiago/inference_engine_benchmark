import logging
import os
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():

    logger.info("Starting the vLLM server...")

    # Warm-up the model to ensure it is ready for use
    llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", quantization="awq", dtype="float16")
    llm.chat([])
    
    logger.info("Model is warmed up and ready for use.")

    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    response = llm.generate(["Hello, how are you?"],
    )
    print(response[0].outputs[0].text)

    ## TODO: # Simple chat completion streaming



if __name__ == "__main__":
    main()
