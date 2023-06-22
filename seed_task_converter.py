import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="Converter", description="Converts seed task dataset to question/answer format."
    )

    parser.add_argument(
        "-i", "--in_path", default='old_seed_tasks.jsonl', help="Old seed dataset file; default is 'old_seed_tasks.jsonl'."
    )
    parser.add_argument(
        "-o", "--out_path", default='seed_tasks.jsonl', help="New seed dataset file; default is 'seed_tasks.jsonl'."
    )

    arguments = vars(parser.parse_args())
    in_path = arguments["in_path"]
    out_path = arguments["out_path"]
    convert_seed_tasks(in_path, out_path)

def convert_seed_tasks(in_path, out_path):
    with open(in_path) as in_file:
        with open(out_path, 'w') as out_file:
            for _, line in enumerate(in_file):
                datum = json.loads(line)
                for instance in datum["instances"]:
                    json.dump({'question': datum["instruction"], 'answer': instance["output"]}, out_file)
                out_file.write('\n')

if __name__ == "__main__":
    main()