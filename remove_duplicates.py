
import jsonlines

def main():
    dataset = list(load_dataset("data/lamini_dataset.jsonl"))

    questions = set()

    filtered_dataset = []

    for example in dataset:
        if example["question"] in questions:
            continue

        filtered_dataset.append(example)
        questions.add(example["question"])

    print(f"Removed {len(dataset) - len(filtered_dataset)} duplicate questions")

    save_dataset("data/filtered_lamini_dataset.jsonl", filtered_dataset)

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
