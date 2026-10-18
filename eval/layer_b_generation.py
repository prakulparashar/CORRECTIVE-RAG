import json
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import AsyncConfig
from dotenv import load_dotenv
from deepeval.models import OpenAIModel
import os
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import OpenAIModel

from langchain_ollama import ChatOllama




# Paths
EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_PATH = EVAL_DIR / "results" / "generation_results.jsonl"

# Application LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0
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








if __name__ == "__main__":
    main()