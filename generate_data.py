from llama import Type, Context, LLM
from llama.error.error import APIError as LlamaAPIError

import jsonlines
import random
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Lamini", description="Generates data for LLM instruction tuning"
    )

    parser.add_argument(
        "-c", "--count", default=100, help="The number of examples to generate; default is 100."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=10,
        help="The number of examples to generate in a batch; default is 10.",
    )
    parser.add_argument(
        "-r", "--response_data", default=False, help="Whether to use seed data as examples for response generation; default is False.", type=bool
    )
    parser.add_argument(
        "-s", "--seed_file", default="seed_tasks.jsonl", help="The path to the seed dataset file; default is 'seed_tasks.jsonl'."
    )
    parser.add_argument(
        "-q", "--question_key", default="question", help="The key for each question in the seed dataset; default is 'question'."
    )
    parser.add_argument(
        "-a", "--answer_key", default="answer", help="The key for each answer in the seed dataset; default is 'answer'."
    )

    arguments = vars(parser.parse_args())

    total_examples = int(arguments["count"])
    batch_size = int(arguments["batch_size"])
    add_response_data = bool(arguments["response_data"])
    seed_file = arguments["seed_file"]
    question_key = arguments["question_key"]
    answer_key = arguments["answer_key"]

    for count in range(0, total_examples, batch_size):
        if count + batch_size > total_examples:
            batch_size = total_examples - count
        print(f"Processing index {count} out of {total_examples} using batch size {batch_size}")
        generate_questions(index=count, batch_size=batch_size, path=seed_file, key=question_key)
        generate_responses(index=count, batch_size=batch_size, path=seed_file, question_key=question_key, answer_key=answer_key, add_response_data=add_response_data)


class Question(Type):
    question: str = Context("a question")


class NovelQuestion(Type):
    question: str = Context("a novel question, with a radically different subject")


def generate_questions(index, batch_size, path, key):

    with open("data/questions.jsonl", "a") as questions_file:
        writer = jsonlines.Writer(questions_file, flush=True)

        llm = LLM(name="generate-lamini")

        seed_questions = list(load_questions(path, key))

        llm.add_data(make_pairs(seed_questions))

        for index in range(index, index + batch_size):
            question = seed_questions[index % len(seed_questions)]
            print("====== Seed Question =====\n", question)
            novel_question = get_question(llm, question)

            novel_question.question = parse(novel_question.question)
            print("===== Novel Question =====\n", novel_question)
            writer.write(novel_question.dict())

def get_question(llm, question):

    attempts = 5

    for _ in range(attempts):
        try:
            return llm(
                input=question,
                output_type=NovelQuestion,
                temperature=0.7,
                model_name="lamini/open",
                max_tokens=32,
            )
        except LlamaAPIError as e:
            print("Lamini API error {i}, retrying")

    raise RuntimeError("Too many Lamini API errors.")



def make_pairs(seed_questions):
    pairs = []
    for question in seed_questions:
        other = random.sample(seed_questions, 1)[0]

        pairs.append([question, NovelQuestion(question=other.question)])

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


def load_questions(path, key='question'):
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


def load_questions_and_answers(path, question_key, answer_key):
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

def generate_responses(index, batch_size, path, question_key, answer_key, add_response_data):
    questions = list(load_questions(path="data/questions.jsonl"))

    with open("data/dataset.jsonl", "a") as dataset_file:
        writer = jsonlines.Writer(dataset_file, flush=True)

        llm = LLM(name="generate-lamini-reponse")

        if add_response_data:
            llm.add_data(load_questions_and_answers(path, question_key, answer_key))

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

    for _ in range(attempts):
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
