import logging
import time
from vllm import LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():

    # Warm-up the model to ensure it is ready for use
    # finding a model with quantization support is a bit tricky
    logger.info("Starting the vLLM server...")
    start_time = time.time()
    llm = LLM(model="meta-llama/Llama-3.2-3B", quantization="bitsandbytes", max_model_len=4096, gpu_memory_utilization=0.7)
    llm.chat([])
    end_time = time.time()
    logger.info("Model is warmed up and ready for use. Took %.4f seconds", end_time - start_time)
    

    # Simple chat completion
    logger.info("Sending: 'Hello, how are you?' to the model...")
    start_time = time.time()
    response = llm.generate(["Hello, how are you?"],
    )
    end_time = time.time()
    print(response[0].outputs[0].text)
    logger.info("Response generated in %.4f seconds", end_time - start_time)


    # TODO: # Simple chat completion streaming


    # FIXME: Simple chat completion from batch
    # logger.info("Sending batch of messages to the model...")
    # messages = ["capital city of France", "capital city of spain?", "capital city of mexico?"]
    # responses = llm.generate(messages)
    # for output in responses:
    #     print(output.outputs[0].text)



if __name__ == "__main__":
    main()
