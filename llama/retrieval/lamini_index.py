from llama.program.util.run_ai import query_run_embedding

from tqdm import tqdm

import faiss
import json
import os
import numpy as np

import logging

logger = logging.getLogger(__name__)


class LaminiIndex:
    def __init__(self, loader=None, config={}):
        self.loader = loader
        self.config = config
        if loader is not None:
            self.build_index()

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

    def build_index(self):
        self.splits = []

        self.index = None

        # load a batch of splits from a generator
        for split_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(split_batch)

            if self.index is None:
                # initialize the index
                logger.info(f"Creating index with dimension {len(embeddings[0])}")
                self.index = faiss.IndexFlatL2(len(embeddings[0]))

            # add the embeddings to the index
            self.index.add(embeddings)

            # save the splits
            self.splits.extend(split_batch)

    def get_embeddings(self, examples):
        embeddings = query_run_embedding(examples, self.config)
        embedding_list = [embedding[0] for embedding in embeddings]

        return np.array(embedding_list)

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        return [self.splits[i] for i in indices[0]]

    def save_index(self, path):
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
