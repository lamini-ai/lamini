import llama

from llama import Type, Context, LLM

from llama.error.error import APIError as LlamaAPIError

from datetime import datetime
import jsonlines
import random
import argparse

import os


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
    # parser.add_argument(
    #     "-r", "--response_data", default=False, help="Whether to use seed data as examples for data filtering.", type=bool
    # )

    arguments = vars(parser.parse_args())

    total_examples = int(arguments["count"])
    batch_size = int(arguments["batch_size"])
    # add_response_data = int(arguments["response_data"])

    start = datetime.now()
    print(f"Start Time: {start}")
    for count in range(0, total_examples, batch_size):
    # for count in range(142, total_examples, batch_size):
        if count + batch_size > total_examples:
            batch_size = total_examples - count
        print(f"Processing index {count} out of {total_examples} using batch size {batch_size}")
        filter_data(start_index=count, batch_size=batch_size)
        # filter_data(start_index=count, batch_size=batch_size, add_response_data=add_response_data)
    end = datetime.now()
    print(f"End Time: {end}")
    print(f"Total Time: {end - start}")


class Question(Type):
    question: str = Context("a question")


class IsValid(Type):
    # is_valid: str = Context("a boolean, 'True' if this is a valid question and 'False' otherwise")
    # is_valid: str = Context("True if this is a valid question and False otherwise")
    # is_valid_question: str = Context("Whether or not the above question is valid, 'True' or 'False'")
    is_valid_question: str = Context("Whether or not the above question is valid, True or False")


class NovelQuestion(Type):
    question: str = Context("a novel question, with a radically different subject")


def filter_data(start_index, batch_size):
# def filter_data(start_index, batch_size, add_response_data=False):

    # with open("data/filtered_deduped_lamini_dataset.jsonl", "w") as questions_file:
    with open("data/filtered_1_questions.jsonl", "a") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        llm = LLM(name="generate-lamini")

        # if add_response_data:
        #     seed_instructions = list(load_seed_instances())
        #     llm.add_data(make_valid_pairs(seed_instructions))

        # dataset = load_dataset()
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


def make_valid_pairs(seed_instructions):
    pairs = []
    for seed in seed_instructions:
        pairs.append([seed[0], IsValid(is_valid_question='True')])
        pairs.append([Question(question=seed[1].response), IsValid(is_valid_question='False')])

    return pairs


def make_pairs(seed_instructions):
    pairs = []
    for seed in seed_instructions:
        other = random.sample(seed_instructions, 1)[0]

        pairs.append([seed, NovelQuestion(question=other.question)])

    return pairs


def parse(string):
    # position = string.find("\n")

    # string = string[position + 1 :]

    position = string.find("\n", 10)
    if position > 0:
        string = string[:position]

    position = string.find(".", 10)
    if position > 0:
        string = string[:position]

    return string


def load_seed_dataset():
    return load_questions(path="seed_tasks.jsonl", key="instruction")


def load_questions(path, key="question"):
    with open(path) as questions_file:
        reader = jsonlines.Reader(questions_file)

        for _, line in enumerate(reader):
            yield Question(
                question=line[key],
            )


class Response(Type):
    response: str = Context("the response to the question")


class QuestionAndResponse(Type):
    question: str = Context("a question")
    response: str = Context("the response to the question")


def load_seed_instances():
    return load_questions_and_answers("seed_tasks.jsonl")

def load_questions_and_answers(path, question_key="instruction", answer_key="output"):
    questions_and_answers = []
    with open(path) as questions_file:
        reader = jsonlines.Reader(questions_file)

        for _, line in enumerate(reader):
            questions_and_answers.append([Question(
                question=line[question_key],
            ), Response(
                response=line["instances"][0][answer_key],
            )])
    return questions_and_answers

def generate_responses(index, batch_size, add_response_data=False):
    questions = list(load_questions(path="data/questions.jsonl"))

    with open("data/dataset.jsonl", "a") as dataset_file:
        writer = jsonlines.Writer(dataset_file, flush=True)

        llm = LLM(name="generate-lamini-reponse")

        if add_response_data:
            llm.add_data(load_seed_instances())

        for question in questions[index:index+batch_size]:
            print("====== Question =====\n", question)
            response = get_response(llm, question)

            response.response = parse_response(response.response)
            print("===== Response =====\n", response)
            question_and_response = QuestionAndResponse(
                question=question.question, response=response.response
            )
            writer.write(question_and_response.dict())

def get_response(llm, question):

    attempts = 5

    for i in range(attempts):
        try:
            return llm(
                input=question,
                output_type=Response,
                temperature=0.0,
                model_name="lamini/instruct",
                max_tokens=128,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")


def parse_response(string):
    #break_point = string.find("\n\n")

    #if break_point >= 0:
    #    string = string[:break_point]

    return string.strip()


main()
