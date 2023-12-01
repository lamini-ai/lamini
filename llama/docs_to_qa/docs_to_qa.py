from llama import BasicModelRunner

import pandas as pd
from tqdm import tqdm
import os
import re
import json
from collections import defaultdict
from random import sample

PROMPT_SEP = "\n"


class DocsToQA:
    def __init__(
        self,
        docs_dirpath=None,
        qa_path=None,
        model_name="meta-llama/Llama-2-13b-chat-hf",
    ):
        self.docs = {}  # { "doc_id": "doc_text" }
        self.embedded_docs = {}  # { "doc_id": "doc_embedding" }
        self.docs_dirpath = docs_dirpath
        if self.docs_dirpath:
            self._chunk_docs()
        self.question_examples = [] # [ "question", ... ]
        self.answer_examples = [] # [ "answer", ... ]
        # self.qa_examples = []  # [ ["question", "answer"], ... ]
        if qa_path:
            self._get_question_examples(qa_path)
            self._get_answer_examples(qa_path)
            # self._get_qa_examples(qa_path)

        self.question_system_prompt = "You are a focused assistant who only asks questions, no chit chat. Always ask questions as helpfully as possible, while being safe. You only ask factually coherent questions about the reference text. Do not repeat the request and do not express thanks, just start asking questions and only ask questions."
        self.answer_system_prompt = "You are an expert. You answer questions factually, grounded in given reference material. Answer concisely."
        self.qa_system_prompt = "You are an assistant who answers questions and holds a conversation. You are helpful and friendly."

        self.question_prompt_suffix = "Write 5 questions about the above:"
        self.answer_prompt_suffix = (
            "Answer the above question, based solely on the reference material above:"
        )

        self.prompt_sep = PROMPT_SEP

        self.question_llm = BasicModelRunner(model_name=model_name)
        self.answer_llm = BasicModelRunner(model_name=model_name)
        self.qa_llm = BasicModelRunner(model_name=model_name)
        self.llama_prompt_template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""

        self.questions = defaultdict(
            list
        )  # { "doc_id": ["question1", "question2", ...], ... }
        self.qa = defaultdict(list)  # { "doc_id": ["question", "answer"], ... }

    def _get_question_examples(self, qa_path):
        df = pd.read_csv(qa_path)
        if "Question" in df.columns:
            for _, row in df.iterrows():
                question = row["Question"]
                self.question_examples.append(question)

    def _get_question_prompt(self, num_examples=3):
        prompt_question_examples = sample(self.question_examples, min(num_examples, len(self.question_examples)))
        prompt = "Example Questions:\n"
        for question in prompt_question_examples:
            prompt += f"{question}\n"
        prompt += "Task:\n"
        return prompt

    def _get_answer_examples(self, qa_path):
        df = pd.read_csv(qa_path)
        if "Answer" in df.columns:
            for _, row in df.iterrows():
                answer = row["Answer"]
                self.answer_examples.append(answer)

    def _get_answer_prompt(self, num_examples=3):
        prompt_answer_examples = sample(self.answer_examples, min(num_examples, len(self.answer_examples)))
        prompt = "Example Answers:\n"
        for answer in prompt_answer_examples:
            prompt += f"{answer}\n"
        prompt += "Task:\n"
        return prompt

    # def _get_qa_examples(self, qa_path):
    #     df = pd.read_csv(qa_path)
    #     for _, row in df.iterrows():
    #         question = row["Question"]
    #         answer = row["Answer"]
    #         self.qa_examples.append([question, answer])

    # def _get_qa_prompt(self, num_examples=3):
    #     prompt_qa_examples = sample(self.qa_examples, min(num_examples, len(self.qa_examples)))
    #     prompt = "Examples of some or all task items:\n"
    #     for qa in prompt_qa_examples:
    #         question, answer = qa
    #         prompt += f"{question}{self.prompt_sep}{self.answer_prompt_suffix} [/INST] {answer}\n[INST] "
    #     prompt += "Task:\n"
    #     return prompt

    def _add_doc_to_question(self, question, doc_id):
        doc = self.docs[doc_id]
        question_with_doc = f"{doc}{self.prompt_sep}{question}"
        return question_with_doc

    def train(self, is_public=False, use_retrieval=True):
        # Create dataframe with columns: "question", "answer"
        rows = []
        for doc_id, qas in self.qa.items():
            for qa in qas:
                question, answer = qa
                prompt = self._make_prompt(self.qa_system_prompt, question)
                rows.append([prompt, answer])

                if use_retrieval:
                    # Include examples with doc ("retrieval") context
                    question_with_doc = self._add_doc_to_question(question, doc_id)
                    prompt_with_doc = self._make_prompt(self.qa_system_prompt, question_with_doc)
                    rows.append([prompt_with_doc, answer])

        df = pd.DataFrame(rows, columns=["input", "output"])
        self.qa_llm.clear_data()
        self.qa_llm.load_data_from_dataframe(df)
        self.qa_llm.train(is_public=is_public)
        return self.qa_llm.model_name

    def run(self, user_input, doc_id=None, verbose=False):
        if doc_id is not None:
            user_input = self._add_doc_to_question(user_input, doc_id)
        prompt = self._make_prompt(self.qa_system_prompt, user_input)
        if verbose:
            print("=============PROMPT================")
            print(prompt)
        output = self.qa_llm(prompt)
        output = self._parse_llm_output(output)
        return output

    def _get_chunks(self, text, char_chunk_size, start_index=0):
        # chunks = []
        for i in range(0, len(text), char_chunk_size):
            # chunks.append(text[i:i+char_chunk_size])
            self.docs[start_index] = text[i:i+char_chunk_size]
            start_index += 1
        # return chunks
        return start_index

    def _get_type(self, f):
        return os.path.splitext(f)[1]

    def _chunk_docs(self, char_chunk_size=5000):
        """Default chunk size is 5000"""
        doc_index = 0
        for dirpath, _, filenames in os.walk(self.docs_dirpath):
            for filename in filenames:
                docs_path = dirpath + os.sep + filename
                docs_path_type = self._get_type(docs_path)
                if docs_path_type == ".csv":
                    df = pd.read_csv(docs_path, nrows=100) # size limit
                    for _, row in df.iterrows():
                        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        doc_index = self._get_chunks(text, char_chunk_size, doc_index)
                elif docs_path_type == ".jsonl" or docs_path_type == ".jsonlines" or docs_path_type == ".json":
                    df = pd.read_json(docs_path, lines=True, nrows=100) # size limit
                    for _, row in df.iterrows():
                        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        doc_index = self._get_chunks(text, char_chunk_size, doc_index)
                elif docs_path_type == ".txt":
                    text = open(docs_path).read(1000000) # size limit
                    doc_index = self._get_chunks(text, char_chunk_size, doc_index)
                # else:
                #     raise ValueError(f"Unsupported file type: {docs_path_type}")

    def _make_prompt(self, system_prompt, user_prompt, cue=None):
        llama_prompt = self.llama_prompt_template.format(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        if cue:
            llama_prompt += cue
        return llama_prompt

    def _parse_llm_output(self, output):
        parsed_output = re.sub(r"(\W)(?=\1)", "", output)  # remove repeated punctuation
        parsed_output = re.sub(
            r"\b(\w+)( \1\b)+", r"\1", parsed_output
        )  # remove repeated words
        return parsed_output

    def _parse_enumerated_list(self, text):
        pattern = r"\d+\.\s*(.*)"
        matches = re.findall(pattern, text)
        return matches

    def _run_questions(
        self,
        start_index=0,
        save=False,
        verbose=False,
    ):
        dirpath = None
        for docs_id, chunk in tqdm(list(self.docs.items())[start_index:]):
            prompt = f"{chunk}{self.prompt_sep}{self.question_prompt_suffix}" if self.question_prompt_suffix else chunk
            # if self.qa_examples:
                # prompt = self._get_qa_prompt() + prompt
            if self.question_examples:
                prompt = self._get_question_prompt() + prompt
            prompt = self._make_prompt(self.question_system_prompt, prompt, cue="1.")
            output = self.question_llm(prompt)
            output = self._parse_llm_output(output)
            try:
                if not output.startswith("1."):
                    output = f"1. {output}"
                output = self._parse_enumerated_list(output)
            except:
                output = [output]
            self.questions[docs_id] = output
            if verbose:
                print("=============PROMPT================")
                print(prompt)
                print("============SYSTEM PROMPT=================")
                # print(question_llm.system_prompt)
                print(self.question_system_prompt)
                print("============GENERATED QUESTION================")
                print(output)
                print("=============================")
            if save:
                if dirpath:
                    self._save_questions(dirpath=dirpath, verbose=verbose)
                else:
                    dirpath = self._save_questions(verbose=verbose)
        if save:
            print(f"Saved questions to {dirpath}/questions.json")

    def load_questions(self, dirpath):
        questions_path = f"{dirpath}/questions.json"
        questions_prompt_path = f"{dirpath}/questions_prompt.json"
        self.questions = json.load(open(questions_path))
        self.questions = {
            int(k) if k.isdigit() else k: v for k, v in self.questions.items()
        }
        questions_prompt = json.load(open(questions_prompt_path))
        self.question_system_prompt = questions_prompt["system_prompt"]
        self.question_prompt_suffix = questions_prompt["prompt_suffix"]
        if not self.docs:
            # docs_path = f"{dirpath}/docs.json"
            # self.docs = json.load(open(docs_path))
            self.docs_dirpath = open(f"{dirpath}/docs_dirpath.txt").read()
            self._chunk_docs()

    def _save_questions(self, dirpath=None, verbose=True):
        if dirpath is None:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            dirpath = f"outputs/questions_{ts}"
        os.makedirs(dirpath, exist_ok=True)

        filepath = f"{dirpath}/questions.json"
        with open(filepath, "w") as file:
            json.dump(self.questions, file)
        if verbose:
            print(
                f"Saved {len([question for question in self.questions.values()])} questions to {filepath}"
            )

        prompt_filepath = f"{dirpath}/questions_prompt.json"
        with open(prompt_filepath, "w") as file:
            json.dump(
                {
                    "system_prompt": self.question_system_prompt,
                    "prompt_suffix": self.question_prompt_suffix,
                },
                file,
            )
        if verbose:
            print(f"Saved question prompt to {prompt_filepath}")

        # docs_filepath = f"{dirpath}/docs.json"
        # with open(docs_filepath, "w") as file:
        #     json.dump(self.docs, file)
        # if verbose:
        #     print(f"Saved documents to {docs_filepath}")
        docs_filepath = f"{dirpath}/docs_dirpath.txt"
        with open(docs_filepath, "w") as file:
            file.write(self.docs_dirpath)
        if verbose:
            print(f"Saved documents path to {docs_filepath}")
        return dirpath

    def prompt_engineer_questions(
        self,
        system_prompt=None,
        prompt_suffix=None,
        start_index=0,
        save=False,
        verbose=True,
    ):
        if system_prompt is not None:
            self.question_system_prompt = system_prompt
        if prompt_suffix is not None:
            self.question_prompt_suffix = prompt_suffix

        self._run_questions(start_index=start_index, save=save, verbose=verbose)

        return self.questions

    def _run_answers(
        self,
        start_index=0,
        batch_size=1,
        save=False,
        verbose=False,
    ):
        if save:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            dirpath = f"outputs/qa_{ts}"
            os.makedirs(dirpath, exist_ok=True)

            self._save_questions(dirpath=dirpath)

            prompt_filepath = f"{dirpath}/answers_prompt.json"
            with open(prompt_filepath, "w") as file:
                json.dump(
                    {
                        "system_prompt": self.answer_system_prompt,
                        "prompt_suffix": self.answer_prompt_suffix,
                    },
                    file,
                )
            print(f"Saved answers prompt to {prompt_filepath}")

            # docs_filepath = f"{dirpath}/docs.json"
            # with open(docs_filepath, "w") as file:
            #     json.dump(self.docs, file)
            # if verbose:
            #     print(f"Saved documents to {docs_filepath}")
            docs_filepath = f"{dirpath}/docs_dirpath.txt"
            with open(docs_filepath, "w") as file:
                file.write(self.docs_dirpath)
            if verbose:
                print(f"Saved documents path to {docs_filepath}")

        doc_ids_in_questions = [int(k) for k in self.questions.keys()]
        prompts_list = []
        questions_list = []
        docs_ids_list = []
        for docs_id, chunk in self.docs.items():
            if docs_id not in doc_ids_in_questions:
                continue
            question_list = self.questions[docs_id]
            for question in question_list:
                prompt = f"{chunk}{self.prompt_sep}{question}{self.prompt_sep}{self.answer_prompt_suffix}" if self.answer_prompt_suffix else f"{chunk}{self.prompt_sep}{question}"
                # if self.qa_examples:
                    # prompt = self._get_qa_prompt() + prompt
                if self.answer_examples:
                    prompt = self._get_answer_prompt() + prompt
                prompt = self._make_prompt(self.answer_system_prompt, prompt)

                prompts_list.append(prompt)
                questions_list.append(question)
                docs_ids_list.append(docs_id)

        # Iterate through prompts in batches
        for i in tqdm(range(start_index, len(prompts_list), batch_size)):
            batch_prompts = prompts_list[i : i + batch_size]
            batch_answers = self.answer_llm(batch_prompts)
            batch_questions = questions_list[i : i + batch_size]
            batch_docs_ids = docs_ids_list[i : i + batch_size]
            for j in range(len(batch_questions)):
                question = batch_questions[j]
                prompt = batch_prompts[j]
                answer = batch_answers[j]["output"]
                answer = self._parse_llm_output(answer)
                docs_id = batch_docs_ids[j]

                self.qa[docs_id].append([question, answer])
                if verbose:
                    print("=============PROMPT================")
                    print(prompt)
                    print("============SYSTEM PROMPT=================")
                    print(self.answer_system_prompt)
                    print("=============GENERATED ANSWER================")
                    print(answer)
                    print("=============================")
            if save:
                filepath = f"{dirpath}/qa.json"
                with open(filepath, "w") as file:
                    json.dump(self.qa, file)
                if verbose:
                    print(f"Saved {i + batch_size} answers to {filepath}")
        if save:
            print(f"Saved answers to {filepath}")

    def load_qa(self, dirpath):
        qa_path = f"{dirpath}/qa.json"
        answers_prompt_path = f"{dirpath}/answers_prompt.json"

        self.qa = json.load(open(qa_path))
        self.qa = {int(k) if k.isdigit() else k: v for k, v in self.qa.items()}
        answers_prompt = json.load(open(answers_prompt_path))
        self.answer_system_prompt = answers_prompt["system_prompt"]
        self.answer_prompt_suffix = answers_prompt["prompt_suffix"]

        questions_path = f"{dirpath}/questions.json"
        questions_prompt_path = f"{dirpath}/questions_prompt.json"

        self.questions = json.load(open(questions_path))
        self.questions = {
            int(k) if k.isdigit() else k: v for k, v in self.questions.items()
        }
        questions_prompt = json.load(open(questions_prompt_path))
        self.question_system_prompt = questions_prompt["system_prompt"]
        self.question_prompt_suffix = questions_prompt["prompt_suffix"]
        if not self.docs:
            # docs_path = f"{dirpath}/docs.json"
            # self.docs = json.load(open(docs_path))
            self.docs_dirpath = open(f"{dirpath}/docs_dirpath.txt").read()
            self._chunk_docs()

    def prompt_engineer_answers(
        self,
        system_prompt=None,
        prompt_suffix=None,
        start_index=0,
        questions=None,
        save=False,
        verbose=True,
    ):
        if system_prompt is not None:
            self.answer_system_prompt = system_prompt
        if prompt_suffix is not None:
            self.answer_prompt_suffix = prompt_suffix
        if questions is None:
            assert (
                self.questions
            ), "You must set questions first, or pass questions in as {doc_id: question}"
            questions = self.questions

        self._run_answers(start_index=start_index, save=save, verbose=verbose)

        return self.qa


def load_model(docs_dirpath=None, qa_path=None, model_name=None):
    """
    Load DocsToQA model with specified
    docs_dirpath - default None
    qa_path (Optional) - default None
    and model_name - default "meta-llama/Llama-2-13b-chat-hf"
    """
    if model_name is None:
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    llm = DocsToQA(docs_dirpath, qa_path, model_name)
    return llm


def run_prompt_engineer_questions(
        docs_dirpath,
        qa_path=None,
        model_name=None,
        system_prompt=None,
        prompt_suffix=None,
        start_index=0,
        save=True,
        verbose=True,
    ):
    """
    Generates questions for each document in docs_dirpath using an LLM.

    Input:
        Params for loading an LLM:
        docs_dirpath (str): path to docs directory
        qa_path (str): path to qa csv or jsonl file
        model_name (str): name of model

        Params for generating prompt engineered questions:
        system_prompt (str): system prompt
        prompt_suffix (str): prompt suffix
        start_index (int): start index
        save (bool): whether to save
        verbose (bool): whether to print

    Output:
        questions (list): list of questions

    """
    llm = load_model(docs_dirpath, qa_path, model_name)
    questions = llm.prompt_engineer_questions(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )

    return questions


# Note: For good quality finetuned model, manually edit answers in answers file
# TODO: create LLM pipeline to filter answers
def run_prompt_engineer_answers(
    questions_dirpath,
    docs_dirpath=None,
    qa_path=None,
    model_name=None,
    system_prompt=None,
    prompt_suffix=None,
    start_index=0,
    save=True,
    verbose=True,
):
    """
    Generates answers for each question in questions_dirpath using an LLM.

    Input:
        Params for loading an LLM:
        docs_dirpath (str, optional): path to docs directory
        qa_path (str, optional): path to qa csv or jsonl file
        model_name (str, optional): name of model

        Params for generating prompt engineered answers:
        questions_dirpath (str): path to questions directory
        system_prompt (str, optional): system prompt
        prompt_suffix (str, optional): prompt suffix
        start_index (int, optional): start index
        save (bool, optional): whether to save
        verbose (bool, optional): whether to print

    Output:
        answers (list): list of answers
    """
    llm = load_model(docs_dirpath, qa_path, model_name)
    llm.load_questions(dirpath=questions_dirpath)
    answers = llm.prompt_engineer_answers(
        system_prompt=system_prompt,
        prompt_suffix=prompt_suffix,
        start_index=start_index,
        save=save,
        verbose=verbose,
    )

    return answers


def finetune_qa(qa_dirpath, docs_dirpath=None, model_name=None, is_public=False, use_retrieval=True):
    """
    Finetunes an LLM on a set of questions and answers.

    Input:
        qa_dirpath (str): path to qa directory
        docs_dirpath (str, optional): path to docs directory
        model_name (str, optional): name of model
        is_public (bool, optional): whether to use public or private model

    Output:
        llm (DocsToQA): finetuned LLM
    """

    llm = load_model(docs_dirpath, model_name=model_name)
    llm.load_qa(dirpath=qa_dirpath)
    model_name = llm.train(is_public=is_public, use_retrieval=use_retrieval)
    return model_name


def run_model(question, docs_dirpath=None, model_name=None, doc_id=None, verbose=False):
    """
    Generates an answer for a question using an LLM.

    Input:
        question (str): question
        docs_dirpath (str, optional): path to docs directory
        model_name (str, optional): name of model
        doc_id (str, optional): id of document
        verbose (bool, optional): whether to print

    Output:
        output (str): answer
    """
    llm = load_model(docs_dirpath, model_name=model_name)
    output = llm.run(question, doc_id, verbose=verbose)
    if verbose:
        print("============MODEL ANSWER================")
        print(output)
    return output
