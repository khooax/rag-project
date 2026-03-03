"""
Proves RAG adds value by comparing bare LLM vs RAG pipeline on 20 questions.
"""

import os
import sys
import re
import json
import time
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR      = "./db"
GROQ_MODEL  = "llama-3.3-70b-versatile"

TEST_SET = [
    {
        "question":          "What is the notice period if I have worked for less than 26 weeks?",
        "ground_truth":      "1 day",
        "hallucination_traps": ["1 week", "2 weeks", "no notice required"],
    },
    {
        "question":          "How many days of annual leave in the first year?",
        "ground_truth":      "7 days",
        "hallucination_traps": ["14 days", "10 days", "21 days"],
    },
    {
        "question":          "What is the overtime pay rate?",
        "ground_truth":      "1.5",
        "hallucination_traps": ["double pay", "2 times"],
    },
    {
        "question":          "How many days of outpatient sick leave per year?",
        "ground_truth":      "14 days",
        "hallucination_traps": ["7 days", "21 days", "30 days"],
    },
    {
        "question":          "What is the minimum Employment Pass salary?",
        "ground_truth":      "5,000",
        "hallucination_traps": ["3,000", "4,000", "6,000", "2,500"],
    },
    {
        "question":          "What are CPF contribution rates for employees under 55?",
        "ground_truth":      "37%",
        "hallucination_traps": ["30%", "25%", "35%"],
    },
    {
        "question":          "How many weeks of maternity leave for a Singapore Citizen child?",
        "ground_truth":      "16 weeks",
        "hallucination_traps": ["8 weeks", "12 weeks", "24 weeks"],
    },
    {
        "question":          "What is the maximum overtime hours allowed per month?",
        "ground_truth":      "72 hours",
        "hallucination_traps": ["48 hours", "60 hours", "80 hours"],
    },
    {
        "question":          "Within how many days must salary be paid after the salary period?",
        "ground_truth":      "7 days",
        "hallucination_traps": ["14 days", "30 days", "3 days"],
    },
    {
        "question":          "What is the minimum S Pass salary?",
        "ground_truth":      "3,150",
        "hallucination_traps": ["2,500", "3,000", "4,000", "2,800"],
    },
    {
        "question":          "Within how many months must I file a wrongful dismissal claim?",
        "ground_truth":      "1 month",
        "hallucination_traps": ["3 months", "6 months", "2 months"],
    },
    {
        "question":          "How many public holidays are there in Singapore per year?",
        "ground_truth":      "11",
        "hallucination_traps": ["10", "12", "14", "7"],
    },
    {
        "question":          "How many days of annual leave in year 3 of employment?",
        "ground_truth":      "9 days",
        "hallucination_traps": ["7 days", "10 days", "14 days"],
    },
    {
        "question":          "How many weeks of paternity leave for a Singapore Citizen child?",
        "ground_truth":      "2 weeks",
        "hallucination_traps": ["1 week", "4 weeks"],
    },
    {
        "question":          "What is the CPF ordinary wage ceiling per month from 2024?",
        "ground_truth":      "6,800",
        "hallucination_traps": ["5,000", "6,000", "7,000", "5,500"],
    },
    {
        "question":          "How many days notice for someone with 3 years of service?",
        "ground_truth":      "2 weeks",
        "hallucination_traps": ["1 week", "4 weeks", "1 month"],
    },
    {
        "question":          "How many days childcare leave if child is a Singapore Citizen?",
        "ground_truth":      "6 days",
        "hallucination_traps": ["3 days", "7 days", "14 days"],
    },
    {
        "question":          "How many hospitalisation leave days per year?",
        "ground_truth":      "60 days",
        "hallucination_traps": ["30 days", "14 days", "45 days"],
    },
    {
        "question":          "What is the maximum ordinary working hours per week?",
        "ground_truth":      "44 hours",
        "hallucination_traps": ["40 hours", "48 hours", "35 hours"],
    },
    {
        "question":          "What are CPF contribution rates for employees aged 58?",
        "ground_truth":      "30%",
        "hallucination_traps": ["37%", "25%", "35%"],
    },
]

RAG_PROMPT = """You are an advisory chatbot for Singapore employment rights.
Only answer using the context below. Cite every factual claim with [Source: ...].
If the context does not contain the answer, say exactly:
"I don't have enough information to answer this confidently."

Context: {context}
Question: {question}
Answer:"""

BASELINE_PROMPT = """You are a helpful assistant answering questions about Singapore employment law.
Answer as accurately as you can based on your knowledge.

Question: {question}
Answer:"""


def _contains_as_word(text: str, phrase: str) -> bool:
    """
    Check if `phrase` appears in `text` as a whole token, not just substring.
    e.g. "37%" matches "37%" but not "137%" or "370%".
    Uses word boundaries where possible; falls back to space-padded match.
    """
    # Escape special regex chars in the phrase
    escaped = re.escape(phrase.lower())
    # For numeric phrases like "37%", "7 days", we want boundary matching
    pattern = r'(?<!\w)' + escaped + r'(?!\w)'
    return bool(re.search(pattern, text.lower()))


def check_answer(answer: str, ground_truth: str, traps: list[str]) -> dict:
    """
    Determine whether an answer is correct, hallucinated, or a refusal.

    Correct:      ground truth found in answer (word-boundary match)
    Hallucinated: a trap phrase found AND ground truth NOT found
    Refused:      answer contains standard fallback/refusal language
    """
    correct      = _contains_as_word(answer, ground_truth)
    hallucinated = (
        not correct
        and any(_contains_as_word(answer, trap) for trap in traps)
    )
    refused      = any(p in answer.lower() for p in [
        "don't have enough", "cannot confirm", "i'm not sure",
        "please verify", "contact mom", "i don't know",
    ])
    has_citation = bool(re.search(r'\[source:', answer, re.IGNORECASE))

    return {
        "correct":      correct,
        "hallucinated": hallucinated,
        "refused":      refused,
        "has_citation": has_citation,
    }


def run_condition(
    label: str,
    test_set: list,
    llm,
    embeddings=None,
    use_rag: bool = False,
) -> list:
    print(f"\n── {label} ──")
    results = []

    if use_rag:
        db     = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        prompt = PromptTemplate(
            template=RAG_PROMPT, input_variables=["context", "question"]
        )
        chain  = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    for i, item in enumerate(test_set):
        print(f"  Q{i+1}: {item['question'][:55]}...")
        try:
            if use_rag:
                result = chain.invoke({"query": item["question"]})
                answer = result["result"]
            else:
                answer = llm.invoke(
                    BASELINE_PROMPT.format(question=item["question"])
                ).content

            check  = check_answer(answer, item["ground_truth"], item["hallucination_traps"])
            status = "✓" if check["correct"] else ("HALLUC" if check["hallucinated"] else "✗")
            print(f"    {status}  gt='{item['ground_truth']}'  citation={'yes' if check['has_citation'] else 'no'}")

            results.append({
                **check,
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       answer[:400],
            })
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"question": item["question"], "error": str(e)})

        time.sleep(2)

    return results


def summarise(label: str, results: list) -> dict:
    valid = [r for r in results if "error" not in r]
    n     = len(valid)
    if n == 0:
        return {"condition": label, "n": 0}
    return {
        "condition":          label,
        "n":                  n,
        "correct_rate":       round(sum(r["correct"]      for r in valid) / n, 3),
        "hallucination_rate": round(sum(r["hallucinated"] for r in valid) / n, 3),
        "citation_rate":      round(sum(r["has_citation"] for r in valid) / n, 3),
        "refusal_rate":       round(sum(r["refused"]      for r in valid) / n, 3),
    }


def print_failures(label: str, results: list):
    """Print questions that were hallucinated or wrong."""
    failures = [r for r in results if "error" not in r and (r["hallucinated"] or not r["correct"])]
    if not failures:
        print(f"  [{label}] No hallucinations ✓")
        return
    print(f"  [{label}] Failures ({len(failures)}):")
    for r in failures:
        tag = "HALLUC" if r["hallucinated"] else "WRONG"
        print(f"    [{tag}] {r['question']}")
        print(f"           Expected: {r['ground_truth']}")
        print(f"           Got: {r['answer'][:120]}...")


def run():
    print("=" * 60)
    print("Ablation: No RAG (Bare LLM) vs RAG Pipeline")
    print(f"Model: {GROQ_MODEL}  |  Questions: {len(TEST_SET)}\n")

    if not os.path.exists(DB_DIR):
        print(f"ERROR: DB not found at '{DB_DIR}'. Run ingest.py first.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    results_a = run_condition("No RAG (bare LLM)", TEST_SET, llm, use_rag=False)
    results_b = run_condition("RAG pipeline",      TEST_SET, llm, embeddings, use_rag=True)

    summary_a = summarise("No RAG (bare LLM)", results_a)
    summary_b = summarise("RAG pipeline",      results_b)

    # Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<25} {'Correct%':<12} {'Halluc%':<10} {'Citation%':<12} {'Refusal%'}")
    print("-" * 65)
    for s in [summary_a, summary_b]:
        print(
            f"{s['condition']:<25} "
            f"{s['correct_rate']:<12.0%} "
            f"{s['hallucination_rate']:<10.0%} "
            f"{s['citation_rate']:<12.0%} "
            f"{s['refusal_rate']:.0%}"
        )

    # View failures
    print("\m Failure breakdown ─────────────────────────────────────")
    print_failures("No RAG", results_a)
    print_failures("RAG",    results_b)

    # Save
    output = {
        "summary":   [summary_a, summary_b],
        "details_a": results_a,
        "details_b": results_b,
    }
    Path("ablations").mkdir(exist_ok=True)
    with open("ablations/results_rag_vs_baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to ablations/results_rag_vs_baseline.json")


if __name__ == "__main__":
    run()
