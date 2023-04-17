import llama

from llama import Type, Context, LLM

import time
import jsonlines
import random

import os


def main():
    generate_questions()
    generate_responses()


def generate_questions():

    with open("data/questions.jsonl", "w") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        generate_questions_sequential(writer)


class Question(Type):
    question: str = Context("a question")


class NovelQuestion(Type):
    question: str = Context(
        "a novel question, a questions with a radically different subject"
    )


def generate_questions_sequential(writer):

    llm = LLM(name="generate-lamini")

    seed_instructions = list(load_seed_dataset())

    llm.add_data(make_pairs(seed_instructions))

    for instruction in seed_instructions:
        print("====== Seed Question =====\n", instruction)
        novel_question = llm(
            input=instruction,
            output_type=NovelQuestion,
            temperature=0.7,
            model_name="lamini/open",
            max_tokens=32,
        )
        novel_question.question = parse(novel_question.question)
        print("===== Novel Question =====\n", novel_question)
        writer.write(novel_question.dict())


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


def generate_responses():
    questions = load_questions(path="data/1_questions.jsonl")

    responses = generate_responses_for_questions(questions)

    with open("data/dataset.jsonl", "w") as dataset_file:
        writer = jsonlines.Writer(dataset_file)

        for response in responses:
            writer.write(response.dict())


class Response(Type):
    response: str = Context("the response to the question")


class QuestionAndResponse(Type):
    question: str = Context("a question")
    response: str = Context("the response to the question")


def generate_responses_for_questions(questions):

    llm = LLM(name="generate-lamini-reponse")

    responses = []

    for question in questions:
        print("====== Question =====\n", question)
        response = llm(
            input=question,
            output_type=Response,
            temperature=0,
            model_name="lamini/instruct",
            max_tokens=128,
        )
        response.response = parse_response(response.response)
        print("===== Response =====\n", response)
        responses.append(
            QuestionAndResponse(question=question.question, response=response.response)
        )

    return responses


def parse_response(string):
    break_point = string.find("\n\n")

    if break_point >= 0:
        string = string[:break_point]

    return string.strip()


main()
