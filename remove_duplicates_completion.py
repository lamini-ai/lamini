
import jsonlines
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from nltk.tokenize.punkt import PunktSentenceTokenizer

def remove_duplicates(sentences):
    filtered_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        if sentence['sentence'] in seen_sentences:
            continue
        filtered_sentences.append(sentence['span'])
        seen_sentences.add(sentence['sentence'])
    return filtered_sentences

def fuzzy_remove_duplicates(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings = []

    for sentence in sentences:
        new_sentence = dict(sentence)
        new_sentence["embedding"] = model.encode(sentence["sentence"])
        embeddings.append(new_sentence)

    all_embeddings = np.concatenate(
        [np.expand_dims(sentence["embedding"], axis=0)
        for sentence in embeddings])

    all_similarities = cosine_similarity(all_embeddings)
    all_similarities -= np.identity(all_similarities.shape[0])

    to_remove = all_similarities > 0.9

    locations = np.where(to_remove)
    to_delete = set()

    for x, y in zip(locations[0], locations[1]):
        if x in to_delete:
            continue
        to_delete.add(y)

    filtered_sentences = [sentence for index, sentence in
        enumerate(embeddings) if index not in to_delete]

    return [filtered_sentence['span'] for filtered_sentence in filtered_sentences]

def main():
    start = datetime.now()
    print(f"Start: {start}")
    dataset = list(load_dataset("data/filtered_lamini_dataset.jsonl"))

    filtered_dataset = []

    remove_fuzzy = False
    num_removed = 0

    pt = PunktSentenceTokenizer()

    for i, example in enumerate(dataset):
        if not example["response"]:
            continue
        sentences = [{'sentence': sentence, 'span': span} for sentence, span in zip(pt.tokenize(example["response"]), pt.span_tokenize(example["response"]))]
        if remove_fuzzy:
            filtered_sentences = fuzzy_remove_duplicates(sentences)
        else:
            filtered_sentences = remove_duplicates(sentences)
        example["response"] = ''.join([example["response"][span[0]:span[1]] for span in filtered_sentences])
        filtered_dataset.append(example)

        if len(sentences) - len(filtered_sentences):
            print(f"Removed {len(sentences) - len(filtered_sentences)} duplicate sentences from example {i}")
            num_removed += 1

    save_dataset("data/filtered_lamini_dataset_dedup_completion.jsonl", filtered_dataset)
    end = datetime.now()
    print(f"Num Removed: {num_removed}")
    print(f"End: {end}")
    print(f"Total Time: {end - start}")

def load_dataset(path):
    with open(path) as dataset_file:
        reader = jsonlines.Reader(dataset_file)

        for example in reader:
            yield example

def save_dataset(path, dataset):
    with open(path, "w") as dataset_file:
        writer = jsonlines.Writer(dataset_file)

        for example in dataset:
            writer.write(example)


main()
