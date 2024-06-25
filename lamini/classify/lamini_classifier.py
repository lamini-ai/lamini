import logging
import os
import pickle
import random
from itertools import chain
from typing import List

import jsonlines
from lamini import Lamini
from lamini.api.embedding import Embedding
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

class LaminiClassifier:
    """A zero shot classifier that uses the Lamini LlamaV2Runner to generate
    examples from prompts and then trains a final logistic regression on top
    of an LLM to classify the examples.
    """

    def __init__(
        self,
        config: dict = {},
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
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

        for class_name, prompt in prompts.items():
            try:
                logger.info(
                    f"Generating examples for class '{class_name}' from prompt {prompt}"
                )
                self.add_class(class_name)

                result = self.generate_examples_from_prompt(
                    class_name,
                    prompt,
                    self.examples.get(class_name, []))

                self.examples[class_name] = result

                # Save partial progress
                self.save_examples()

            except Exception as e:
                logger.error(f"Failed to generate examples for class {class_name}")
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

    # Add alias for tune
    tune = train

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

        embed = Embedding(self.config)
        embeddings = embed.generate(examples)
        return [embedding[0] for embedding in embeddings]

    def predict_proba(self, text):
        return self.logistic_regression.predict_proba(self.get_embeddings(text))

    def predict_proba_from_embedding(self, embeddings):
        return self.logistic_regression.predict_proba(embeddings)

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

    def classify_from_embedding(
        self, embedding, top_n=None, threshold=None, metadata=False
    ):
        is_singleton = True if len(embedding) == 1 else False

        batch_probs = self.predict_proba_from_embedding(embedding)

        return self._classify_impl(
            batch_probs, is_singleton, top_n, threshold, metadata
        )

    def classify(self, text, top_n=None, threshold=None, metadata=False):
        is_singleton = True if isinstance(text, str) else False

        batch_probs = self.predict_proba(text)

        return self._classify_impl(
            batch_probs, is_singleton, top_n, threshold, metadata
        )

    def _classify_impl(
        self, batch_probs, is_singleton, top_n=None, threshold=None, metadata=False
    ):
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
        obj = SavedLaminiClassifier(
            self.logistic_regression,
            self.class_names_to_ids,
            self.class_ids_to_metadata,
        )
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def save_local(self, filename):
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


class SavedLaminiClassifier:
    def __init__(
        self,
        logistic_regression: LogisticRegression,
        class_names_to_ids: dict,
        class_ids_to_metadata: dict,
    ):
        self.logistic_regression = logistic_regression
        self.class_names_to_ids = class_names_to_ids
        self.class_ids_to_metadata = class_ids_to_metadata

    @staticmethod
    def loads(data):
        return pickle.loads(data)


class DefaultExampleGenerator:
    def __init__(
        self,
        prompt,
        config=None,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        batch_size=10,
    ):
        self.prompt = prompt
        self.config = config
        self.example_count = 5
        self.model_name = model_name
        self.batch_size = batch_size

        self.max_history = 2

    def generate_examples(self, seed, examples):
        prompt_batch = self.get_prompts(
            seed=seed, examples=examples
        )

        runner = Lamini(config=self.config, model_name=self.model_name)

        results = runner.generate(
            prompt=prompt_batch,
            output_type={
                "example_1": "str",
                "example_2": "str",
                "example_3": "str",
            },
        )

        logger.debug("+++++++ Default Example Generator Result ++++++++")
        logger.debug(results)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(results)

        return examples

    def get_prompts(self, seed, examples):
        prompt = "You are a domain expert who is able to generate many different examples given a description."

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

        return prompt

    def parse_result(self, result):
        return [
                result["example_1"],
                result["example_2"],
                result["example_3"],
            ]

class DefaultExampleModifier:
    def __init__(
        self,
        config=None,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        batch_size=10,
    ):
        self.config = config
        self.model_name = model_name
        self.required_examples = 5
        self.batch_size = batch_size

    def modify_examples(self, examples):
        prompts = self.get_prompts(examples)

        runner = Lamini(config=self.config, model_name=self.model_name)

        results = runner.generate(
            prompt=prompts,
            output_type={
                "example_1": "str",
                "example_2": "str",
                "example_3": "str",
            },
        )

        logger.debug("+++++++ Default Example Modifier Result ++++++++")
        logger.debug(results)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(results)

        return examples

    def get_prompts(self, existing_examples):

        examples = existing_examples.copy()
        random.seed(42)

        prompts = []

        for batch_example in range(self.batch_size):
            # Randomly shuffle the examples
            random.shuffle(examples)

            example_count = min(5, len(examples))

            prompt = "You are a domain expert who is able to clearly understand these descriptions and modify them to be more diverse."
            prompt += "Read the following descriptions carefully:\n"
            prompt += "----------------------------------------\n"
            for index, example in enumerate(examples[:example_count]):
                prompt += f"{index + 1}. {example}\n"
            prompt += "\n----------------------------------------\n"

            prompt += "Generate 3 more examples that are similar, but substantially different from those above. Each example should be as specific as possible using at most 10 words.\n"

            logger.debug("+++++++ Default Example Modifier Prompt ++++++++")
            logger.debug(prompt)
            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

            prompts.append(prompt)

        return prompts

    def parse_result(self, results):
        all_examples = []
        for result in results:
            all_examples += [
                result["example_1"],
                result["example_2"],
                result["example_3"],
            ]

        return all_examples


class DefaultExampleExpander:
    def __init__(
        self, prompt, config=None, model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        self.prompt = prompt
        self.config = config
        self.model_name = model_name

    def expand_example(self, example_batch):
        runner = Lamini(config=self.config, model_name=self.model_name)

        prompts = self.get_prompts(example_batch)

        results = runner.generate(prompt=prompts)

        for result in results:
            logger.debug("+++++++ Default Example Expander Result ++++++++")
            logger.debug(result)
            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return results

    def get_prompts(self, example_batch):

        prompts = []

        for example in example_batch:

            prompt = "You are a domain expert who is able to clearly understand this description and expand to a complete example from a short summary."

            prompt += "Read the following description carefully:\n"
            prompt += "----------------------------------------\n"
            prompt += self.prompt
            prompt += "\n----------------------------------------\n"
            prompt += "Now read the following summary of an example matching this description carefully:\n"
            prompt += "----------------------------------------\n"
            prompt += example
            prompt += "\n----------------------------------------\n"

            prompt += "Expand the summary to a complete example with about 3 sentences.  Be consistent with both the summary and the description.  Get straight to the point.\n"

            logger.debug("+++++++ Default Example Expander Prompt ++++++++")
            logger.debug(prompt)
            logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

            prompts.append(prompt)

        return prompts