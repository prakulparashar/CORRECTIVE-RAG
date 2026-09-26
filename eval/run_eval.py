import json
from pathlib import Path

from backend import ingest_pdf, _THREAD_RETRIEVERS


# Fixed thread ID for the entire evaluation PDF
EVAL_THREAD_ID = "eval_attention"

# Evaluation files
EVAL_DIR = Path(__file__).parent
PDF_PATH = EVAL_DIR / "attention.pdf"
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_PATH = EVAL_DIR / "results" / "retrieval_results.jsonl"


def setup_evaluation_document():
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    summary = ingest_pdf(
        pdf_bytes,
        thread_id=EVAL_THREAD_ID,
        filename=PDF_PATH.name,
    )

    print("Evaluation PDF indexed successfully.")


def run_retriever_evaluation():
    # Get the retriever created by ingest_pdf()
    retriever = _THREAD_RETRIEVERS[EVAL_THREAD_ID]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as dataset_file, \
         open(RESULTS_PATH, "w", encoding="utf-8") as results_file:

        for line in dataset_file:
            row = json.loads(line)

            question_id = row["id"]
            question = row["question"]


            # the isolated retriever is being evaluated here
            docs = retriever.invoke(question)

            result = {
                "id": question_id,
                "question": question,
                "expected_route": row["expected_route"],
                "ground_truth": row["ground_truth"],
                "reference_context": row.get("reference_context", ""),
                "retrieved_context": [
                    doc.page_content for doc in docs
                ]
            }

            results_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print(f"Results saved to: {RESULTS_PATH}")





if __name__ == "__main__":
    setup_evaluation_document()
    run_retriever_evaluation()