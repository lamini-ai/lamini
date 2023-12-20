
import os

import logging

logger = logging.getLogger(__name__)

class DefaultChunker:
    def __init__(self, chunk_size=512, step_size=128):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        # return a list of strings, each a substring of the text with length self.chunk_size
        # the last element of the list may be shorter than self.chunk_size
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=DefaultChunker()):
        self.directory = directory
        self.chunker = chunker
        self.batch_size = batch_size

    def load(self):
        # load all of the files in the directory recursively as text into a list of strings
        # return the list of strings
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                with open(os.path.join(root, file), 'r') as f:
                    logger.debug("Loading file: %s", os.path.join(root, file))
                    yield f.read()

    def get_chunks(self):
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self):
        # A generator that yields batches of chunks
        # Each batch is a list of strings, each a substring of the text with length self.chunk_size
        # the last element of the list may be shorter than self.chunk_size
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def __iter__(self):
        return self.get_chunk_batches()


