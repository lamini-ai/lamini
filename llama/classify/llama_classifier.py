from typing import List
from llama.program.util.run_ai import query_run_embedding

from llama import LlamaV2Runner, Type, Context

from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

from itertools import chain

import os
import re
import random
import pickle
import jsonlines

import logging

logger = logging.getLogger(__name__)


class BinaryLaminiClassifier:
    """A zero shot binary classifier that uses the Lamini LlamaV2Runner to generate
    examples from prompts and then uses a logistic regression to classify
    the examples.
    """

    def __init__(
        self,
        config: dict = {},
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        augmented_example_count: int = 10,
        generator_from_prompt=None,
        example_modifier=None,
        example_expander=None,
    ):
        self.classifier = LaminiClassifier(
            config=config,
            model_name=model_name,
            augmented_example_count=augmented_example_count,
            generator_from_prompt=generator_from_prompt,
            example_modifier=example_modifier,
            example_expander=example_expander,
        )

    def add_positive_examples(self, examples):
        self.classifier.add_data_to_class("positive", examples)

    def add_negative_examples(self, examples):
        self.classifier.add_data_to_class("negative", examples)

    def get_augmented_positive_examples(self):
        return self.classifier.get_data()["positive"]

    def get_augmented_negative_examples(self):
        return self.classifier.get_data()["negative"]

    def prompt_train(self, positive_prompt, negative_prompt):
        self.classifier.prompt_train(
            {"positive": positive_prompt, "negative": negative_prompt}
        )

    def predict_proba(self, text: List[str]):
        return [
            [
                probs[self.classifier.class_names_to_ids["negative"]],
                probs[self.classifier.class_names_to_ids["positive"]],
            ]
            for probs in self.classifier.predict_proba(text)
        ]

    def predict(self, text: List[str]):
        return [1 if prob[1] > prob[0] else 0 for prob in self.predict_proba(text)]

    def classify(self, text: List[str], top_n=None, threshold=None, metadata=False):
        return self.classifier.classify(text, top_n, threshold, metadata)

    def dumps(self):
        return pickle.dumps(self)

    @staticmethod
    def loads(data):
        return pickle.loads(data)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


class LaminiClassifier:
    """A zero shot classifier that uses the Lamini LlamaV2Runner to generate
    examples from prompts and then trains a final logistic regression on top
    of an LLM to classify the examples.
    """

    def __init__(
        self,
        config: dict = {},
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        augmented_example_count: int = 10,
        batch_size: int = 10,
        threads: int = 1,
        saved_examples_path: str = "/tmp/saved_examples.jsonl",
        generator_from_prompt=None,
        example_modifier=None,
        example_expander=None,
    ):
        self.config = config
        self.model_name = model_name
        self.augmented_example_count = augmented_example_count
        self.batch_size = batch_size
        self.threads = threads
        self.saved_examples_path = saved_examples_path

        if generator_from_prompt is None:
            generator_from_prompt = DefaultExampleGenerator
        self.generator_from_prompt = generator_from_prompt

        if example_modifier is None:
            example_modifier = DefaultExampleModifier
        self.example_modifier = example_modifier

        if example_expander is None:
            example_expander = DefaultExampleExpander
        self.example_expander = example_expander

        self.class_ids_to_metadata = {}
        self.class_names_to_ids = {}

        # Examples is a dict of examples, where each row is a different
        # example class, followed by examples of that class
        self.examples = self.load_examples()

    def prompt_train(self, prompts: dict):
        """Trains the classifier using prompts for each class.

        First, augment the examples for each class using the prompts.
        """
        with ThreadPoolExecutor(max_workers=self.threads) as thread_pool:
            # Generate examples from prompts
            generation_tasks = []

            for class_name, prompt in prompts.items():
                logger.info(
                    f"Generating examples for class '{class_name}' from prompt {prompt}"
                )
                self.add_class(class_name)

                # submit the generation task to the thread pool
                generated_examples = thread_pool.submit(
                    self.generate_examples_from_prompt,
                    class_name,
                    prompt,
                    self.examples.get(class_name, []),
                )

                # save the future
                generation_tasks.append(generated_examples)

            # Wait for all the generation tasks to finish
            for class_name, generated_examples in zip(
                prompts.keys(), as_completed(generation_tasks)
            ):
                try:
                    self.examples[class_name] = generated_examples.result()

                    # Save partial progress
                    self.save_examples()
                except Exception as e:
                    logger.error(
                        f"Failed to generate examples for class '{class_name}'"
                    )
                    logger.error(e)
                    logger.error(generated_examples.exception())
                    logger.error(
                        "Consider rerunning the generation task if the error is transient, e.g. 500"
                    )

        self.train()

    def train(self):
        # Form the embeddings
        X = []
        y = []

        for class_name, examples in tqdm(self.examples.items()):
            index = self.class_names_to_ids[class_name]
            y += [index] * len(examples)
            class_embeddings = self.get_embeddings(examples)
            X += class_embeddings

        # Train the classifier
        self.logistic_regression = LogisticRegression(random_state=0).fit(X, y)

    def add_data_to_class(self, class_name, examples):
        if not isinstance(examples, list):
            examples = [examples]

        self.add_class(class_name)

        if not class_name in self.examples:
            self.examples[class_name] = []
        self.examples[class_name] += examples

    def add_class(self, class_name):
        if not class_name in self.class_names_to_ids:
            class_id = len(self.class_names_to_ids)
            self.class_names_to_ids[class_name] = class_id
            self.class_ids_to_metadata[class_id] = {"class_name": class_name}

    def add_metadata_to_class(self, class_name, metadata):
        self.class_ids_to_metadata[self.class_names_to_ids[class_name]][
            "metadata"
        ] = metadata

    def get_data(self):
        return self.examples

    def get_embeddings(self, examples):
        if isinstance(examples, str):
            examples = [examples]
        embeddings = query_run_embedding(examples, config=self.config)

        return [embedding[0] for embedding in embeddings]

    def predict_proba(self, text):
        return self.logistic_regression.predict_proba(self.get_embeddings(text))

    def predict(self, text):
        if not isinstance(text, list):
            raise Exception("Text to predict must be a list of string(s)")

        probs = self.predict_proba(text)

        # select the class with the highest probability, note that text and
        # probs are lists of arbitrary length
        winning_classes = [
            max(enumerate(prob), key=lambda x: x[1])[0] for prob in probs
        ]

        # convert the class ids to class names
        return [
            list(self.class_names_to_ids.keys())[class_id]
            for class_id in winning_classes
        ]

    def classify(self, text, top_n=None, threshold=None, metadata=False):
        is_singleton = True if isinstance(text, str) else False

        batch_probs = self.predict_proba(text)

        batch_final_probs = []
        for probs in batch_probs:
            final_probs = []
            for class_id, prob in enumerate(probs):
                if threshold is None or prob > threshold:
                    # Include the metadata if requested
                    class_name = self.class_ids_to_metadata[class_id]["class_name"]
                    final_prob = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "prob": prob,
                    }
                    if metadata:
                        metadata = self.class_ids_to_metadata[class_id]
                        final_prob["metadata"] = metadata
                    final_probs.append(final_prob)

            # Sort the final_probs, a list of dicts each with a key "prob"
            sorted_probs = sorted(final_probs, key=lambda x: x["prob"], reverse=True)

            if top_n is not None:
                sorted_probs = sorted_probs[:top_n]
            batch_final_probs.append(sorted_probs)

        return batch_final_probs if not is_singleton else batch_final_probs[0]

    def dumps(self):
        return pickle.dumps(self)

    @staticmethod
    def loads(data):
        return pickle.loads(data)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return LaminiClassifier.loads(f.read())

    def create_new_example_generator(self, prompt, original_examples):
        example_generator = self.generator_from_prompt(
            prompt,
            config=self.config,
            model_name=self.model_name,
            batch_size=self.batch_size // 5,
        )
        example_modifier = self.example_modifier(
            config=self.config,
            model_name=self.model_name,
            batch_size=self.batch_size // 5,
        )
        example_expander = self.example_expander(
            prompt, config=self.config, model_name=self.model_name
        )

        examples = original_examples.copy()

        index = len(examples)

        while True:
            # Phase 1: Generate example types from prompt
            compressed_example_features = example_generator.generate_examples(
                seed=index, examples=examples
            )

            # Phase 2: Modify the features to be more diverse
            different_example_features = example_modifier.modify_examples(
                compressed_example_features
            )

            different_example_features = chain(
                different_example_features, compressed_example_features
            )

            different_example_features_batches = self.batchify(
                different_example_features
            )

            # Phase 3: Expand examples from features
            for features_batches in different_example_features_batches:
                expanded_example_batch = example_expander.expand_example(
                    features_batches
                )

                for expanded_example in expanded_example_batch:
                    logger.debug(
                        f"Generated example number {index} out of {self.augmented_example_count}"
                    )

                    index += 1
                    examples.append(expanded_example)
                    yield expanded_example

                    if index >= self.augmented_example_count:
                        return

    def batchify(self, examples):
        batches = []
        # handle batches that are smaller than batch_size
        for example in examples:
            if len(batches) == 0 or len(batches[-1]) == self.batch_size:
                batches.append([])
            batches[-1].append(example)

        return batches

    def generate_examples_from_prompt(self, class_name, prompt, original_examples):
        examples = []
        if isinstance(original_examples, str):
            original_examples = [original_examples]

        # No need to generate more examples if we already have enough
        if len(original_examples) >= self.augmented_example_count:
            logger.debug(
                f"Already have enough examples ({len(original_examples)}) for class '{class_name}', not generating more"
            )
            return original_examples

        for example in tqdm(
            self.create_new_example_generator(prompt, original_examples),
            total=self.augmented_example_count,
        ):
            examples.append(example)

            if len(examples) >= self.augmented_example_count:
                break

        return examples + original_examples

    def load_examples(self):
        filename = self.saved_examples_path
        if not os.path.exists(filename):
            return {}

        # load the examples from the jsonl file using the jsonlines library
        with jsonlines.open(filename) as reader:
            examples = {}
            for row in reader:
                class_name = row["class_name"]
                example = row["examples"]
                self.add_class(class_name)
                examples[class_name] = example

        return examples

    def save_examples(self):
        filename = self.saved_examples_path

        # save the examples as a jsonl file using the jsonlines library
        with jsonlines.open(filename, "w") as writer:
            for class_name, example in self.examples.items():
                row = {
                    "class_name": class_name,
                    "examples": example,
                }
                writer.write(row)


class DefaultExampleGenerator:
    def __init__(
        self,
        prompt,
        config=None,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        batch_size=10,
    ):
        self.prompt = prompt
        self.config = config
        self.example_count = 5
        self.model_name = model_name
        self.batch_size = batch_size

        self.max_history = 2

    def generate_examples(self, seed, examples):
        prompt_batch, system_prompt = self.get_prompt_and_system_prompt_batch(
            seed=seed, examples=examples
        )

        class FiveOutputs(Type):
            example_1: str = Context("")
            example_2: str = Context("")
            example_3: str = Context("")
            example_4: str = Context("")
            example_5: str = Context("")

        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        results = runner(
            inputs=prompt_batch,
            system_prompt=system_prompt,
            output_type=FiveOutputs,
        )

        logger.debug("+++++++ Default Example Generator Result ++++++++")
        logger.debug(results)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(results)

        for example in examples:
            yield example

    def get_prompt_and_system_prompt_batch(self, seed, examples):
        batch_size = min(len(examples) + 1, self.batch_size)

        prompts = []

        for i in range(batch_size):
            prompt, system_prompt = self.get_prompt_and_system_prompt(
                seed=seed, examples=examples
            )

            prompts.append(prompt)

        return prompts, system_prompt

    def get_prompt_and_system_prompt(self, seed, examples):
        prompt, system_prompt = self.get_prompt(seed=seed, examples=examples)

        logger.debug("+++++++ Default Example Generator Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Generator System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return prompt, system_prompt

    def get_prompt(self, seed, examples):
        system_prompt = "You are a domain expert who is able to generate many different examples given a description."

        prompt = ""

        # Randomly shuffle the examples
        random.seed(seed)
        random.shuffle(examples)

        # Include examples if they are available
        if len(examples) > 0:
            selected_example_count = min(self.max_history, len(examples))

            prompt += "Consider the following examples:\n"

            for i in range(selected_example_count):
                prompt += "----------------------------------------\n"
                prompt += f"{examples[i]}"
                prompt += "\n----------------------------------------\n"

        prompt += "Read the following description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += self.prompt
        prompt += "\n----------------------------------------\n"

        prompt += f"Generate {self.example_count} different example summaries following this description. Each example summary should be as specific as possible using at most 10 words.\n"

        return prompt, system_prompt

    def parse_result(self, results):
        all_examples = []
        for result in results:
            all_examples += [
                result.example_1,
                result.example_2,
                result.example_3,
                result.example_4,
                result.example_5,
            ]

        return all_examples


class DefaultExampleModifier:
    def __init__(
        self, config=None, model_name="meta-llama/Llama-2-7b-chat-hf", batch_size=10
    ):
        self.config = config
        self.model_name = model_name
        self.required_examples = 5
        self.batch_size = batch_size

    def modify_examples(self, examples):
        prompts, system_prompt = self.get_prompt_batch(examples)

        class FiveOutputs(Type):
            example_1: str = Context("")
            example_2: str = Context("")
            example_3: str = Context("")
            example_4: str = Context("")
            example_5: str = Context("")

        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        results = runner(
            inputs=prompts, system_prompt=system_prompt, output_type=FiveOutputs
        )

        logger.debug("+++++++ Default Example Modifier Result ++++++++")
        logger.debug(results)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(results)

        for example in examples:
            yield example

    def get_prompt_batch(self, examples):
        existing_examples = []

        for example in examples:
            existing_examples.append(example)
            if len(existing_examples) >= self.required_examples:
                break

        prompts = []

        for i in range(self.batch_size):
            prompt, system_prompt = self.get_prompt(
                seed=i, existing_examples=existing_examples
            )

            prompts.append(prompt)

        return prompts, system_prompt

    def get_prompt(self, seed, existing_examples):
        system_prompt = "You are a domain expert who is able to clearly understand these descriptions and modify them to be more diverse."

        examples = existing_examples.copy()

        # Randomly shuffle the examples
        random.seed(seed)
        random.shuffle(examples)

        example_count = min(5, len(examples))

        prompt = "Read the following descriptions carefully:\n"
        prompt += "----------------------------------------\n"
        for index, example in enumerate(examples[:example_count]):
            prompt += f"{index + 1}. {example}\n"
        prompt += "\n----------------------------------------\n"

        prompt += "Generate 5 more examples that are similar, but substantially different from those above. Each example should be as specific as possible using at most 10 words.\n"

        logger.debug("+++++++ Default Example Modifier Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Modifier System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return prompt, system_prompt

    def parse_result(self, results):
        all_examples = []
        for result in results:
            all_examples += [
                result.example_1,
                result.example_2,
                result.example_3,
                result.example_4,
                result.example_5,
            ]

        return all_examples


class DefaultExampleExpander:
    def __init__(self, prompt, config=None, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.prompt = prompt
        self.config = config
        self.model_name = model_name

    def expand_example(self, example_batch):
        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        prompts, system_prompt = self.get_prompt_batch(example_batch)

        results = runner(inputs=prompts, system_prompt=system_prompt)

        for result in results:
            logger.debug("+++++++ Default Example Expander Result ++++++++")
            logger.debug(result)
            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
            yield result["output"]

    def get_prompt_batch(self, example_batch):
        prompts = []

        for example in example_batch:
            prompt, system_prompt = self.get_prompt(example)

            prompts.append(prompt)

        return prompts, system_prompt

    def get_prompt(self, example):
        system_prompt = "You are a domain expert who is able to clearly understand this description and expand to a complete example from a short summary."

        prompt = "Read the following description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += self.prompt
        prompt += "\n----------------------------------------\n"
        prompt += "Now read the following summary of an example matching this description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += example
        prompt += "\n----------------------------------------\n"

        prompt += "Expand the summary to a complete example.  Be consistent with both the summary and the description.  Get straight to the point.\n"

        logger.debug("+++++++ Default Example Expander Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Expander System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return prompt, system_prompt
