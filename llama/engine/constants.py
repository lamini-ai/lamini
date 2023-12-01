import openai

# TODO: confirm what cue works best for lists
# LIST_CUE = '['
# LIST_CUE = '1.'
LIST_CUE = "\n-"
ESC_LIST_CUE = "\\n-"
SEPARATOR = "\n"
CHUNK_SEPARATOR = " (continuing input) "
EXAMPLE_SEPARATOR = "END"
STOP_TOKEN = "\nEND"
FIXED_TEMPERATURE = 0.0
RANDOM_TEMPERATURE = 0.7
MAX_INPUT_TOKENS = 3872
MAX_OUTPUT_TOKENS = 256
EMBEDDING_MODEL = lambda prompt: openai.Embedding.create(
    input=[prompt], model="text-embedding-ada-002"
)["data"][0]["embedding"]