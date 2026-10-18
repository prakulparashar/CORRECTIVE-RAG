import json
from pathlib import Path
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from dotenv import load_dotenv
from deepeval.models import OpenAIModel
import os
from deepeval.models import OpenAIModel
from langchain_ollama import ChatOllama
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)

load_dotenv()

# Paths
EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_PATH = EVAL_DIR / "results" / "generation_results.jsonl"

# LLM for generating our responses, so that we can build our test cases, this is the same LLM used in our main CRAG pipeline
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)



# LLM for evaluation/ Judge LLM
judge_model = OpenAIModel(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)



def generate_answer(question, retrieved_context):
    prompt = f"""
    Answer the question using only the provided context.

    If the context does not contain enough information to answer,
    explicitly state that the information is insufficient.

    Context:
    {retrieved_context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content


def load_dataset():                          
    dataset = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

    return dataset


def main():
    dataset = load_dataset()

    # Keep only questions that have a reference context, in order to ignore greeting and all
    answerable_questions = [
        row for row in dataset
        if row.get("reference_context")
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for row in answerable_questions:
            question = row["question"]
            context = row["reference_context"]

            generated_answer = generate_answer(
                question,
                context
            )

            result = {
                "id": row.get("id"),
                "question": question,
                "reference_context": context,
                "ground_truth": row["ground_truth"],
                "generated_answer": generated_answer
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"Generated answer for: {question}")

    print(f"\nProcessed {len(answerable_questions)} questions.")
    print(f"Results saved to: {RESULTS_PATH}")



def load_retrieval_results():
    results = []

    with open (RESULTS_PATH, 'r', encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    return results




def build_test_cases(results):
    test_cases = []

    for row in results:  #no need to skip those without reference context here, cause in our dataset there are responses of only those questions whose reference_context is not there, i.e already skipped, check the main function 


        test_case = LLMTestCase(          #defining an object with predefined metrics for evaluation
            input=row["question"],
            actual_output=row["generated_answer"],
            expected_output=row["ground_truth"],
            retrieval_context=[row["reference_context"]],
        ) 

        test_cases.append(test_case)

    return test_cases

def get_metrics():
    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=False,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=False,
    )

    correctness = GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual output correctly answers the input "
            "and matches the expected output. Accept semantically equivalent "
            "answers even when their wording differs."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=judge_model,
        async_mode=False,
    )

    return [faithfulness, answer_relevancy, correctness]




def evaluate_generation():
    results = load_retrieval_results()
    test_cases = build_test_cases(results)

    test_cases = test_cases[:1]

    print(f"Loaded {len(test_cases)} generation test cases.")

    metrics = get_metrics()

    evaluate(
        test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
            throttle_value=3,
        ),
    )






if __name__ == "__main__":
    #main() #for generating and saving answers in generation_results.jsonl
    evaluate_generation() #for evaluation