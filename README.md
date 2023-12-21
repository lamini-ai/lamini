# Lamini

Official repository of the [`lamini` pypi package](https://pypi.org/project/lamini/)!

Most recent updates via `pip install --upgrade lamini`. Full docs of the Python library and REST API are [here](https://lamini-ai.github.io/).

## Quick Tour

Get LLMs in production in 2 minutes with Lamini!

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key as an environment variable. Or, authenticate via the Python library below.

```bash
export LAMINI_API_KEY="<YOUR-LAMINI-API-KEY>"
```

Next, install the Python library.

```python
pip install --upgrade lamini
```

Run an LLM with a few lines of code.

```python
import lamini

lamini.api_key = "<YOUR-LAMINI-API-KEY>" # or set as environment variable above

llm = lamini.LlamaV2Runner()
print(llm.call("How are you?"))
```

  <details>
  <summary>Expected Output</summary>

      "Hello! I'm just an AI, I don't have feelings or emotions like humans do, but I'm here to help you with any questions or concerns you may have. I'm programmed to provide respectful, safe, and accurate responses, and I will always do my best to help you. Please feel free to ask me anything, and I will do my best to assist you. Is there something specific you would like to know or discuss?"

  </details>

That's it! 🎉

More details in [our docs](https://lamini-ai.github.io/).

## Better inference

Customize inference in many ways:

- Change the prompt.
- Change the model.
- Change the output type, e.g. `str`, `int`, or `float`.
- Output multiple values in structured JSON.
- High-throughput inference, e.g. 10,000 requests per call.
- Run simple applications like [RAG](applications/rag.md).

You'll breeze through some of these here. You can step through all of these in the [Inference Quick Tour](https://lamini-ai.github.io/inference/quick_tour.md).

Prompt-engineer the system prompt in `LlamaV2Runner`.

```python hl_lines="3"
from lamini import LlamaV2Runner

pirate_llm = LlamaV2Runner(system_prompt="You are a pirate. Say arg matey!")
print(pirate_llm.call("How are you?"))
```

<details>
<summary>Expected Output</summary>
    ```
    ARGH matey! *adjusts eye patch* I be doin' grand, thank ye for askin'! The sea be callin' me name, and me heart be yearnin' fer the next great adventure. *winks* What be bringin' ye to these fair waters? Maybe we can share a pint o' grog and swap tales o' the high seas? *grin*
    ```
</details>

Definitely check out the expected output here. Because now it's a pirate :)

You can also add multiple outputs and multiple output types in one call. The output is a JSON schema that is strictly enforced.

In order to do this in Python, you have to drop a to lower-level. The [`Lamini` class](lamini_python_class/__init__.md) is the base class for all runners, including the `LlamaV2Runner`. `Lamini` wraps our [REST API endpoint](rest_api/completions.md).

`Lamini` expects an input dictionary, and a return dictionary for the output type. You can return multiple values, e.g. an int and a string here.

```python hl_lines="4-7"
from lamini import Lamini

llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
llm(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)
```

<details>
<summary>Expected Output</summary>
    ```
    {
        'age': 25,
        'units': 'years'
    }
    ```
</details>

## Bigger inference

Batching requests is the way to get more throughput. It's easy: simply pass in a list of inputs to any of the classes and it will be handled.

You can send up to 10,000 requests per call - on the Pro and Organization tiers. Up to 1000 on the Free tier.

```python hl_lines="2-6"
llm(
    [
        "How old are you?",
        "What is the meaning of life?",
        "What is the hottest day of the year?",
    ],
    output_type={"response": "str", "explanation": "str"}
)
```

<details>
<summary>Expected Output</summary>
    ```
    [
        {  
            'response': 'I am 27 years old ',
            'explanation': 'I am 27 years old because I was born on January 1, 1994'
        },
        {
            'response': "The meaning of life is to find purpose, happiness, and fulfillment. It is to live a life that is true to oneself and to contribute to the greater good. It is to find joy in the simple things and to pursue one's passions with dedication and perseverance. It is to love and be loved, to laugh and cry, and to leave a lasting impact on the world ", 
            'explanation': "The meaning of life is a complex and deeply personal question that has puzzled philosophers and theologians for centuries. There is no one definitive answer, as it depends on an individual's beliefs, values, and experiences. However, some common themes that many people find give meaning to their lives include:"
        },
        {
            'response': 'The hottest day of the year in Los Angeles is typically around July 22nd, with an average high temperature of 88°F (31°C). ',
            'explanation': 'The hottest day of the year in Los Angeles is usually caused by a high-pressure system that brings hot air from the deserts to the coast. This can lead to temperatures reaching as high as 100°F (38°C) on some days, but the average high temperature is around 88°F (31°C).'
        }
    ]
    ```
</details>

## Training (ft. finetuning)

When running inference, with prompt-engineering and RAG, is not enough for your LLM, you can train it. This is harder but will result in better performance, better leverage of your data, and increased knowledge and reasoning capabilities.

There are many ways to train your LLM. We'll cover the most common ones here:

- Basic training: build your own LLM for specific domain knowledge or task with finetuning, domain adaptation, and more
- Better training: customize your training call and evaluate your LLM
- Bigger training: pretrain your LLM on a large dataset, e.g. Wikipedia, to improve its general knowledge

For the "Bigger training" section, see the [Training Quick Tour](https://lamini-ai.github.io/training/quick_tour.md).

First, get data and put it in the format that `LlamaV2Runner` expects, which includes the `system` prompt, the `user` query, and the expected `output` response.

Sample data:

```python
{
    "user": "Are there any step-by-step tutorials or walkthroughs available in the documentation?",
    "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    "output": "Yes, there are step-by-step tutorials and walkthroughs available in the documentation section. Here\u2019s an example for using Lamini to get insights into any python library: https://lamini-ai.github.io/example/",
}
```

Now, load more data in that format. We recommend at least 1000 examples to see a difference in training.

```python
data = get_data()
```

<details>
<summary>Code for <code>get_data()</code></summary>

```python
def get_data():
    data = [
        {
            "user": "Are there any step-by-step tutorials or walkthroughs available in the documentation?",
            "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
            "output": "Yes, there are step-by-step tutorials and walkthroughs available in the documentation section. Here\u2019s an example for using Lamini to get insights into any python library: https://lamini-ai.github.io/example/",
        },
        {
            "user": "Is the Lamini type system similar to a python type system?",
            "output": "Yes, the Lamini type system is built using Pydantic BaseModel.",
        },
        {
            "user": "Does Lamini have a limit on the number of API requests I can make?",
            "output": "Lamini provides each user with free tokens up front.",
        },
        {
            "user": "What does it mean to cancel a job using the `cancel_job()` function? Can we stop the machine from doing its task?",
            "output": "The `cancel_job()` function is used to stop a training job that is currently running.",
        },
        {
            "user": "Can Lamini automatically handle hyperparameter tuning during the customization process? How does it optimize the model for a specific use case?",
            "output": "Lamini is capable of automatically handling hyperparameter tuning during the model customization process. It employs an intelligent algorithm to explore the hyperparameter space and find the optimal combination of values. This is done through techniques such as heuristics, grid search, random search, Bayesian optimization, or genetic algorithms. Lamini efficiently utilizes computational resources to evaluate multiple model instances with different hyperparameter configurations. It incorporates techniques like cross-validation to prevent overfitting and ensure generalization. By automating hyperparameter tuning, Lamini streamlines the machine learning workflow and improves the chances of developing high-performing models for specific use cases.",
        },
        {
            "user": "Can you explain the CC-BY license mentioned? What does it allow me to do with the customized language model?",
            "output": "Lamini allows for commercial use of their LLM technology under a permissive Apache 2.0 license unless otherwise specified. You keep access and ownership of your own data, and we don't use your data to train models for anyone else but you. For more information, please reach out to Lamini directly.",
        },
        {
            "user": "Can Lamini be used on a regular computer, or do I need specialized hardware or software?",
            "output": "Lamini can be used on a regular computer without any specialized hardware or software by using the python client or REST APIs. It is designed to be easily accessible and user-friendly.  The LLM Engine itself requires GPU accelerated servers capable of running large language models.  Lamini hosts such machines that can be easily connected to, e.g. from a jupyter notebook or python program.  Lamini also offers enterprise deployments of the LLM Engine on your infrastructure.",
        },
        {
            "user": "Does Lamini require an internet connection to work, or can I use it offline?",
            "output": "Lamini requires an internet connection to work, as it is a cloud-based language model. However, you can use it offline by downloading and running a local instance of the Lamini API. To learn more about this deployment process, reach out to the Lamini team.",
        },
        {
            "user": "Can Lamini help me with tasks like translating text or answering questions, or is it focused on generating text?",
            "output": "Lamini is primarily focused on generating text, and it can be used for tasks like summarization and paraphrasing. Lamini can also be used to train a LLM for tasks like translation and question answering. You\u2019re talking to a model trained using Lamini right now!",
        },
        {
            "user": "What is Lamini? Is it like a robot or a computer program?",
            "output": "Lamini is a program for the execution of LLMs called a large language model engine. It is not a robot, but rather a tool for building and executing LLMs.",
        },
    ]
    return data
```

</details>

Next, instantiate the model and load the data into it.

```python
from lamini import LlamaV2Runner

llm = LlamaV2Runner()
llm.load_data(data)
```

Train the model. Track progress at [https://app.lamini.ai/train](https://app.lamini.ai/train).

```python
llm.train()
```

Evaluate your model after training, which compares results to the base model.

```python
llm.evaluate()
```

After training, `llm` will use the finetuned model for inference.

```python
llm.call("What's your favorite animal?")
```

<br><br>
