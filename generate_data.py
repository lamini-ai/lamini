import llama

from llama import Type, Context, LLM

from llama.error.error import APIError as LlamaAPIError

import time
import jsonlines
import random
import argparse

import os


def main():
    parser = argparse.ArgumentParser(
        prog="Lamini", description="Generates data for LLM instruction tuning"
    )

    parser.add_argument(
        "-c", "--count", default=100, help="The number of examples to generate."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=10,
        help="The number of examples to generate in a batch.",
    )

    arguments = vars(parser.parse_args())

    total_examples = int(arguments["count"])
    batch_size = int(arguments["batch_size"])

    for count in range(0, total_examples, batch_size):
        print(f"Processing index {count} out of {total_examples} using batch size {batch_size}")
        generate_questions(start_index=count, batch_size=batch_size)
        generate_responses(index=count, batch_size=batch_size)


class Question(Type):
    question: str = Context("a question")


class NovelQuestion(Type):
    question: str = Context("a novel question, with a radically different subject")


def generate_questions(start_index, batch_size):

    with open("data/questions.jsonl", "a") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        llm = LLM(name="generate-lamini")

        seed_instructions = list(load_seed_dataset())

        llm.add_data(make_pairs(seed_instructions))

        for index in range(start_index, start_index + batch_size):
            instruction = seed_instructions[index % len(seed_instructions)]
            print("====== Seed Question =====\n", instruction)
            novel_question = get_question(llm, instruction)

            novel_question.question = parse(novel_question.question)
            print("===== Novel Question =====\n", novel_question)
            writer.write(novel_question.dict())

def get_question(llm, instruction):

    attempts = 5

    for i in range(attempts):
        try:
            return llm(
                input=instruction,
                output_type=NovelQuestion,
                temperature=0.7,
                model_name="lamini/open",
                max_tokens=32,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")



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

        for index, line in enumerate(reader):
            if index > 2000:
                break
            yield Question(
                question=line[key],
            )


class Response(Type):
    response: str = Context("the response to the question")


class QuestionAndResponse(Type):
    question: str = Context("a question")
    response: str = Context("the response to the question")


def generate_responses(index, batch_size):
    questions = list(load_questions(path="data/questions.jsonl"))

    with open("data/dataset.jsonl", "a") as dataset_file:
        writer = jsonlines.Writer(dataset_file, flush=True)

        llm = LLM(name="generate-lamini-reponse")

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
