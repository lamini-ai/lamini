from tqdm import tqdm
from llama import Type, Context, LLMEngine
from llama.error.error import APIError as LlamaAPIError
from random import sample


def get_attributes(value):
    return [attribute for attribute, _ in value.__fields__.items()]


# class NovelQuestion(Type):
#     question: str = Context("a novel question, with a radically different subject")


# TODO: figure out how to work with more arbitrary Types
class AugmenterOutput(Type):
    # new: str = Context("a novel one, with a different subject")
    new: str = Context("a novel question, with a related subject")


def augment_question_answer_pairs(
    seed_data,
    n=10,
    question_model_name="lamini/open",
    answer_model_name="lamini/instruct",
    question_prompt="",
    answer_prompt="",
    related_data=None,
    schema=None,
    verbose=False,
    config={},
):
    """Augment a seed data with new generated pairs of questions and answers."""
    # Enforce paired data, error with good message if not
    if not isinstance(seed_data[0], list) or not len(seed_data[0]) == 2:
        raise ValueError(
            "Seed data for question-answer pairs must be a list of lists, where each inner list is a question-answer pair (length 2 list)."
        )
    augmenter = QuestionAnswerAugmenter(
        seed_data,
        question_model_name=question_model_name,
        answer_model_name=answer_model_name,
        question_prompt=question_prompt,
        answer_prompt=answer_prompt,
        related_data=related_data,
        schema=schema,
        config=config,
    )
    return augmenter.run(n, verbose=verbose)


def augment(
    seed_data,
    n=10,
    model_name="lamini/open",
    prompt="",
    related_data=None,
    schema=None,
    verbose=False,
    config={},
):
    """Augments a dataset with more of the same Type (e.g. Question)"""
    if isinstance(seed_data[0], list):
        print(
            "Warning: augment_questions() called with paired data, but only questions will be augmented."
        )
    augmenter = Augmenter(
        seed_data,
        model_name=model_name,
        prompt=prompt,
        related_data=related_data,
        schema=schema,
        config=config,
    )
    return augmenter.run(n, verbose=verbose)


def augment_with_answers(
    questions,
    output_type=None,
    model_name="lamini/instruct",
    prompt="",
    related_data=None,
    schema=None,
    verbose=False,
    config={},
):
    augmenter = AnswerAugmenter(
        questions,
        output_type=output_type,
        model_name=model_name,
        prompt=prompt,
        related_data=related_data,
        schema=schema,
        config=config,
    )
    return augmenter.run(verbose=verbose)


def is_batch_model(model_name):
    return True


class Augmenter:
    def __init__(
        self,
        seed_data,
        model_name="lamini/open",
        prompt=None,
        related_data=None,
        schema=None,
        config={},
    ):
        self.seed_data = seed_data
        self.seed_inputs = self.__get_inputs()

        self.model_name = model_name
        self.batch = is_batch_model(self.model_name)
        self.llm = LLMEngine(
            id="lamini-augmenter", model_name=self.model_name, config=config
        )
        self.llm.delete_data()
        self.input_type = type(self.seed_inputs[0])
        self.first_attribute = self.__get_first_attribute()
        data = self.__make_pairs()
        if related_data:
            data += related_data
        self.llm.save_data(data)
        self.schema = schema
        self.prompt = prompt

        self.augmentations = []

    def __get_first_attribute(self):
        return get_attributes(self.input_type)[0]

    def __get_inputs(self):
        if type(self.seed_data[0]) is list:
            return [datum[0] for datum in self.seed_data]
        return self.seed_data

    def __make_pairs(self):
        pairs = []
        # for seed_input in seed_inputs:
        #     other = sample(seed_inputs, 1)[0]
        for i, seed_input in enumerate(self.seed_inputs):
            other = sample(self.seed_inputs[:i] + self.seed_inputs[i + 1 :], 1)[0]

            # pairs.append([seed, NovelQuestion(question=other.question)])
            pairs.append([seed_input, AugmenterOutput(new=other[self.first_attribute])])

        return pairs

    def run(self, n, verbose=False):
        if self.batch:
            batch_size = 20
            for i in tqdm(range(0, n, batch_size)):
                # Go through all the seed data once, then repeat if n is larger than the seed data
                data_to_llm = [
                    self.seed_inputs[j % len(self.seed_inputs)]
                    for j in range(i, min(i + batch_size, n))
                ]

                attempts = 5
                outs = None
                for _ in range(attempts):
                    try:
                        outs = self.llm(
                            data_to_llm,
                            AugmenterOutput,
                            temperature=0.7,
                            schema=self.schema,
                            task=self.prompt,
                        )
                    except LlamaAPIError as e:
                        print(f"Lamini API error {e}, retrying")
                if not outs:
                    raise RuntimeError("Too many Lamini API errors.")
                outs = [
                    self.input_type(**{self.first_attribute: out.new}) for out in outs
                ]
                self.augmentations.extend(outs)

                if verbose:
                    print("Augmenter Input", data_to_llm)
                    print("Augmenter Output", outs)

            return self.augmentations
        else:
            for i in tqdm(range(n)):
                # Go through all the seed data once, then repeat if n is larger than the seed data
                datum_to_llm = self.seed_inputs[i % len(self.seed_inputs)]

                attempts = 5
                out = None
                for _ in range(attempts):
                    try:
                        out = self.llm(
                            datum_to_llm,
                            AugmenterOutput,
                            temperature=0.7,
                            schema=self.schema,
                            task=self.prompt,
                        )
                    except LlamaAPIError as e:
                        print(f"Lamini API error {e}, retrying")
                if not out:
                    raise RuntimeError("Too many Lamini API errors.")
                out = self.input_type(**{self.first_attribute: out.new})
                self.augmentations.append(out)

                if verbose:
                    print("Augmenter Input", datum_to_llm)
                    print("Augmenter Output", out)

            return self.augmentations


class AnswerAugmenter:
    def __init__(
        self,
        questions,
        output_type=None,
        model_name="lamini/instruct",
        prompt="",
        seed_data=None,
        related_data=None,
        schema=None,
        config={},
    ):
        assert (
            isinstance(questions[0], list) or output_type is not None
        ), "Must provide output_type if questions are not paired with an Answer type."
        self.questions = questions  # e.g. questions

        # If there are multiple elements, then assume the last one is Answer type
        self.output_type = (
            output_type if output_type is not None else type(questions[0][-1])
        )  # e.g. Answer type

        self.model_name = model_name
        self.batch = is_batch_model(self.model_name)
        self.llm = LLMEngine(
            id="lamini-answer-augmenter", model_name=self.model_name, config=config
        )
        self.llm.delete_data()
        data = []
        if seed_data:
            data += seed_data
        if related_data:
            data += related_data
        self.llm.save_data(data)
        self.schema = schema
        self.prompt = prompt

        self.augmentations = []

    def run(self, verbose=False):
        if self.batch:
            # Iterate through all questions and generate answers
            batch_size = 20
            for i in tqdm(range(0, len(self.questions), batch_size)):
                # If there are multiple elements, then assume the first one is the input, e.g. Question
                data_to_llm = [
                    datum[0] if isinstance(datum, list) else datum
                    for datum in self.questions[i : i + batch_size]
                ]
                attempts = 5
                answers = None
                for _ in range(attempts):
                    try:
                        answers = self.llm(
                            data_to_llm,
                            self.output_type,
                            schema=self.schema,
                            task=self.prompt,
                        )
                    except LlamaAPIError as e:
                        print(f"Lamini API error {e}, retrying")
                if not answers:
                    raise RuntimeError("Too many Lamini API errors.")
                self.augmentations.extend(answers)
                if verbose:
                    print("AnswerAugmenter Input", data_to_llm)
                    print("AnswerAugmenter Output", answers)
            return self.augmentations
        else:
            # Iterate through all questions and generate answers
            for datum in tqdm(self.questions):
                # If there are multiple elements, then assume the first one is the input, e.g. Question
                datum_to_llm = datum[0] if isinstance(datum, list) else datum
                attempts = 5
                answer = None
                for _ in range(attempts):
                    try:
                        answer = self.llm(
                            datum_to_llm,
                            self.output_type,
                            schema=self.schema,
                            task=self.prompt,
                        )
                    except LlamaAPIError as e:
                        print(f"Lamini API error {e}, retrying")
                if not answer:
                    raise RuntimeError("Too many Lamini API errors.")
                self.augmentations.append(answer)
                if verbose:
                    print("AnswerAugmenter Input", datum_to_llm)
                    print("AnswerAugmenter Output", answer)
            return self.augmentations


class QuestionAnswerAugmenter:  # TODO: eventually inherit from generic DataAugmenter class
    """Build a dataset from seed data of question-answer pairs and augment them with new generated data."""

    def __init__(
        self,
        seed_data,
        question_model_name="lamini/open",
        answer_model_name="lamini/instruct",
        question_prompt="",
        answer_prompt="",
        related_data=None,
        schema=None,
        config={},
    ):
        assert isinstance(
            seed_data[0], list
        ), "Seed data must be a list of lists, where each inner list is a question-answer pair (length 2 list)."

        self.seed_data = seed_data

        self.question_type = type(seed_data[0][0])
        self.answer_type = type(seed_data[0][1])

        self.question_model_name = question_model_name
        self.answer_model_name = answer_model_name
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt

        # self.question_llm = LLMEngine(id="lamini-q-augmenter", model_name=self.question_model_name)
        # self.answer_llm = LLMEngine(id="lamini-a-augmenter", model_name=self.answer_model_name)
        self.related_data = related_data
        self.schema = schema
        self.config = config

        self.augmentations = []  # Question-Answer pairs

    def run(self, n, verbose=False):
        augmented_questions = Augmenter(
            self.seed_data,
            model_name=self.question_model_name,
            prompt=self.question_prompt,
            related_data=self.related_data,
            schema=self.schema,
            config=self.config,
        ).run(n, verbose=verbose)
        augmented_answers = AnswerAugmenter(
            augmented_questions,
            output_type=self.answer_type,
            prompt=self.answer_prompt,
            model_name=self.answer_model_name,
            seed_data=self.seed_data,
            related_data=self.related_data,
            schema=self.schema,
            config=self.config,
        ).run(verbose=verbose)
        self.augmentations = list(zip(augmented_questions, augmented_answers))
        return self.augmentations
