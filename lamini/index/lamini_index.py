import json
import logging
import os
from typing import Iterable, List, Union

import faiss
import numpy as np
from lamini.api.embedding import Embedding
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LaminiIndex:
    def __init__(self, config={}):
        self.config = config
        self.embedding_api = Embedding(self.config)
        self.index = None
        self.splits = []

    @staticmethod
    def load_index(path):
        lamini_index = LaminiIndex()
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")

        # Load the index from a file
        lamini_index.index = faiss.read_index(faiss_path)

        # Load the splits from a file
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)

        return lamini_index

    def init_index(self):
        self.splits = []
        self.index = None

    def add_stream(self, stream: Iterable[str]):
        for item in stream:
            embeddings = self.get_embeddings(item)
            assert len(embeddings.shape) == 2, "stream must be a iterable of prompts"
            self.add_embeddings(embeddings, item)
            self.splits.append(item)

    def add_embeddings(self, embedding: np.ndarray, prompt: str):
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding[0]))
        assert len(embedding.shape) == 2, "stream must be a iterable of prompts"
        self.index.add(embedding)
        self.splits.append(prompt)

    def add_batch(self, batch: List[str]):
        try:
            embeddings = self.get_embeddings(batch)
            assert len(embeddings.shape) == 3, "batch must be a list of prompts"
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0][0]))
            for emb in embeddings:
                self.index.add(emb)

            # save the splits
            self.splits.extend(batch)
        except:
            print("Error in adding embeddings to index.")

    def query_with_embedding(self, embedding: np.ndarray, k=5):
        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        _, indices = self.index.search(embedding_array, k)

        return [self.splits[i] for i in indices[0]]

    @staticmethod
    def build_index(loader: List[List[str]]) -> "LaminiIndex":
        lamini_index = LaminiIndex()
        lamini_index.init_index()
        total_batches = len(loader)
        print(f"Building index with {total_batches} batches")

        # load a batch of splits from a generator
        for split_batch in tqdm(loader):
            lamini_index.add_batch(split_batch)
        return lamini_index

    def get_embeddings(self, examples: Union[str, List[str]]):
        embeddings = self.embedding_api.generate(examples)
        return np.array(embeddings)

    def save_index(self, path: str):
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")

        logger.debug("Saving index to %s", faiss_path)
        logger.debug("Saving splits to %s", splits_path)

        logger.debug("Index size: %d", self.index.ntotal)

        # Save the index to a file
        faiss.write_index(self.index, faiss_path)

        # Save the splits to a file
        with open(splits_path, "w") as f:
            json.dump(self.splits, f)
