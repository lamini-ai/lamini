import os
import jsonlines
from tqdm import tqdm
from lamini.api.lamini import Lamini
from datetime import datetime


def load_model(model):
    return Lamini(model)


def format_results(model_name, ecommerce_scores, earnings_scores, icd_scores) -> dict:
    formatted_results = {
        "config": {
            "model_name": model_name,  # Name of the model
        },
        "results": {},
    }
    formatted_results["results"]["product_response_subjective_score"] = (
        ecommerce_scores.get("product_response_subjective_score", -1)
    )
    formatted_results["results"]["product_id_precision_score"] = ecommerce_scores.get(
        "product_id_precision_score", -1
    )
    formatted_results["results"]["earnings_response_subjective_score"] = (
        earnings_scores.get("earnings_response_subjective_score", -1)
    )
    formatted_results["results"]["earnings_precision_score"] = earnings_scores.get(
        "earnings_precision_score", -1
    )
    formatted_results["results"]["icd11_response_subjective_score"] = icd_scores.get(
        "icd11_response_subjective_score", -1
    )
    formatted_results["results"]["icd11_precision_score"] = icd_scores.get(
        "icd11_precision_score", -1
    )

    return formatted_results


async def save_results(answers, model_name, task_name):
    directory = f"tmp/results/{model_name}"
    os.makedirs(directory, exist_ok=True)
    filename = f"custom-{task_name}-{datetime.now()}.jsonl"
    path = os.path.join(directory, filename)
    print(f"Writing benchmark results to {path} inside the current directory.")
    short_answers = []

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            try:
                answer = {
                    "prompt": answer.prompt,
                    "question": answer.data["question"],
                    "answer": answer.response,
                    "is_exact_match": answer.data["is_exact_match"],
                }
            except Exception as e:
                answer = {
                    "prompt": answer["prompt"],
                    "answer": answer["response"],
                    "is_exact_match": answer["is_exact_match"],
                }
            print(f"\n\n=======\n{answer}\n\n")
            short_answers.append(answer)
            writer.write(answer)
            pbar.update()
    return short_answers
