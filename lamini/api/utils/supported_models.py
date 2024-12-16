# This file list all of tested huggingface models on Lamini Platform.
# The variable names are shortened for easier recognition.


LLAMA_32_1B_INST = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_32_3B_INST = "meta-llama/Llama-3.2-3B-Instruct"
TS_1M = "roneneldan/TinyStories-1M"
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
FALCON_7B = "tiiuae/falcon-7b"
FALCON_7B_INST = "tiiuae/falcon-7b-instruct"
FALCON_11B = "tiiuae/falcon-11B"
STAR_CODER_2_7B = "bigcode/starcoder2-7b"
LLAMA_31_8B_INST = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MISTRAL_7B_INST_V03 = "mistralai/Mistral-7B-Instruct-v0.3"
TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"

ALL = [
    LLAMA_31_8B_INST,
    LLAMA_32_1B_INST,
    LLAMA_32_3B_INST,
    TS_1M,
    TINY_GPT2,
    FALCON_7B,
    FALCON_7B_INST,
    FALCON_11B,
    STAR_CODER_2_7B,
    MISTRAL_7B_INST_V03,
    TINY_MISTRAL,
]

PROD = [TINY_MISTRAL, LLAMA_31_8B_INST, LLAMA_32_3B_INST, MISTRAL_7B_INST_V03]

SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_MODELS = [SENTENCE_TRANSFORMERS]
