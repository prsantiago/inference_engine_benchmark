import logging
import time
from litgpt import LLM
from litgpt.api import benchmark_dict_to_markdown_table

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
    # llm.distribute(quantize="bnb.nf4", precision="16-true", fixed_kv_cache_size=250)
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
    print()
    logger.info("Streaming response generated in %.4f seconds", end_time - start_time)


    # TODO: Simple chat completion from batch


    # Perform speed and resource usage tests
    logger.info("Starting speed and resource usage tests...")
    text, bench_d = llm.benchmark(prompt="Hello, how are you?", num_iterations=10, top_k=1, stream=True)

    print(f"Benchmark text: {text}")
    print(benchmark_dict_to_markdown_table(bench_d))


if __name__ == "__main__":
    main()