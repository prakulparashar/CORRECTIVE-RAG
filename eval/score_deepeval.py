import json
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import AsyncConfig
from dotenv import load_dotenv
from deepeval.models import OpenAIModel
import os

load_dotenv()

# ground_truth = golden answer
# reference_context = gold/source passage supporting the answer
# retrieved_context = what FAISS actually retrieved

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "results" / "retrieval_results.jsonl"


judge_model = OpenAIModel(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)

def load_retrieval_results():
    results = []

    with open (RESULTS_PATH, 'r', encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    return results


def build_test_cases(results):
    test_cases = []

    for row in results:
        if not row["reference_context"]:   #we are skipping greetings for now, will handle them in routing eval. (logic is that rows with greetings do not have a reference_context, so we skip those rows)
            continue 


        test_case = LLMTestCase(                  #creating a structured object for every row  
            input=row["question"],
            actual_output="",
            expected_output=row["ground_truth"],
            retrieval_context=row["retrieved_context"],
            context=[row["reference_context"]],
        )

        test_cases.append(test_case)

    return test_cases

def main():
    results = load_retrieval_results()
    test_cases = build_test_cases(results)

    test_cases = test_cases[:2]

    print(f"Loaded {len(results)} retrieval results.")
    print(f"Scoring {len(test_cases)} answerable questions.")

    contextual_precision = ContextualPrecisionMetric(    #again defining an object with predefined metrics for evaluation
        threshold=0.5,
        include_reason=False,
        model=judge_model,
    )

    contextual_recall = ContextualRecallMetric(          #again defining an object with predefined metrics for evaluation
        threshold=0.5,
        include_reason=False,
        model=judge_model,
    )

    async_config = AsyncConfig(
        run_async=True,
        max_concurrent=1,   # Runs 1 test case at a time to prevent API spikes
        throttle_value=3    # Waits 3 seconds between test cases (keeps requests under 5/min)
    )


    evaluate(                                      #from deepeval import evaluate i.e evaluate() is a pre defined method provided by deepeval, with test_cases and metrics and arguemnts
        test_cases,
        metrics=[
            contextual_precision,
            contextual_recall,
        ],
        async_config=async_config,
    )


if __name__ == "__main__":
    main()


