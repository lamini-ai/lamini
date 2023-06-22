from llama import Type, Context, LLM
from llama.error.error import APIError as LlamaAPIError

from datetime import datetime
import jsonlines
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Lamini", description="Filters data for LLM instruction tuning"
    )

    parser.add_argument(
        "-c", "--count", default=100, help="The number of examples to filter."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=10,
        help="The number of examples to filter in a batch.",
    )

    arguments = vars(parser.parse_args())

    total_examples = int(arguments["count"])
    batch_size = int(arguments["batch_size"])

    start = datetime.now()
    print(f"Start Time: {start}")
    for count in range(0, total_examples, batch_size):
        if count + batch_size > total_examples:
            batch_size = total_examples - count
        print(f"Processing index {count} out of {total_examples} using batch size {batch_size}")
        filter_data(start_index=count, batch_size=batch_size)
    end = datetime.now()
    print(f"End Time: {end}")
    print(f"Total Time: {end - start}")


class Question(Type):
    question: str = Context("a question")


class IsValid(Type):
    is_valid_question: str = Context("Whether or not the above question is valid, True or False")


def filter_data(start_index, batch_size):

    with open("data/filtered_1_questions.jsonl", "a") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        llm = LLM(name="generate-lamini")

        dataset = list(load_questions(path="data/1_questions.jsonl"))

        for index in range(start_index, start_index + batch_size):
            datum = dataset[index]
            print(f"====== Datum =====\n{datum}")
            is_valid = filter_datum(llm, datum)
            print(f"Is Valid: {is_valid}")

            if process_bool(is_valid.is_valid_question):
                writer.write(datum.dict())


def filter_datum(llm, datum):

    attempts = 5

    for _ in range(attempts):
        try:
            return llm(
                input=datum,
                output_type=IsValid,
                model_name="lamini/open",
                max_tokens=32,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")


def process_bool(llm_output):
    processed_llm_output = llm_output.split('\n')[0]
    if processed_llm_output.lower() == 'true':
        return True
    else:
        return False


def load_questions(path, key="question"):
    with open(path) as questions_file:
        reader = jsonlines.Reader(questions_file)

        for _, line in enumerate(reader):
            yield Question(
                question=line[key],
            )


main()
