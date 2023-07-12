from llama import QuestionAnswerModel

def print_training_results(results):
    print("-"*100)
    print("Training Results")
    print(results)
    print("-"*100)

def print_inference(question, finetune_answer, base_answer):
    print('Running Inference for: '+ question)
    print("-"*100)
    print("Finetune model answer: ", finetune_answer)
    print("-"*100)
    print("Base model answer: ", base_answer)
    print("-"*100)
    
def main():
    question="How can I add data to Lamini?"
    model = QuestionAnswerModel()
    base_answer = model.get_answer(question)

    model.load_question_answer_from_jsonlines("data/seed.jsonl")
    model.train()
    results = model.get_eval_results()
    print_training_results(results)
    finetune_answer = model.get_answer(question)

    print_inference(question, finetune_answer, base_answer)
main()
