
import os
import sys
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
GROQ_MODEL  = "llama-3.3-70b-versatile"   # must match production
K_VALUES    = [1, 3, 5, 8]

# Each question has a ground_truth token that must appear in a correct answer.
TEST_QUESTIONS = [
    {
        "question":     "What is the minimum notice period if I have worked for less than 26 weeks?",
        "ground_truth": "1 day",
        "traps":        ["1 week", "2 weeks", "no notice"],
    },
    {
        "question":     "How many days of annual leave in the first year?",
        "ground_truth": "7 days",
        "traps":        ["14 days", "10 days"],
    },
    {
        "question":     "What is the overtime pay rate in Singapore?",
        "ground_truth": "1.5",
        "traps":        ["double pay", "2 times"],
    },
    {
        "question":     "Can my employer deduct money from my salary for poor performance?",
        "ground_truth": "unauthorised",
        "traps":        ["yes", "allowed", "permitted"],
    },
    {
        "question":     "How many weeks of maternity leave for a Singapore Citizen child?",
        "ground_truth": "16 weeks",
        "traps":        ["8 weeks", "12 weeks"],
    },
    {
        "question":     "What should I do if my employer does not pay my salary on time?",
        "ground_truth": "mom",   # answer should reference MOM
        "traps":        ["nothing", "no recourse"],
    },
    {
        "question":     "What is the minimum salary for an Employment Pass?",
        "ground_truth": "5,000",
        "traps":        ["3,000", "4,000", "2,500"],
    },
    {
        "question":     "How many days of outpatient sick leave per year?",
        "ground_truth": "14 days",
        "traps":        ["7 days", "21 days"],
    },
    {
        "question":     "What are the CPF contribution rates for an employee aged 35?",
        "ground_truth": "37",   # 20% employee + 17% employer = 37% total
        "traps":        ["30%", "25%", "20% total"],
    },
    {
        "question":     "What is the maximum overtime hours per month?",
        "ground_truth": "72",
        "traps":        ["48 hours", "60 hours", "80 hours"],
    },
]

RAG_PROMPT = """You are an advisory chatbot for Singapore employment rights.
Only answer using the context below. Cite every factual claim with [Source: ...].
If the context does not contain the answer, say exactly:
"I don't have enough information to answer this confidently."

Context: {context}
Question: {question}
Answer:"""


def check_answer(answer: str, ground_truth: str, traps: list[str]) -> dict:
    a = answer.lower()
    correct      = ground_truth.lower() in a
    hallucinated = any(t.lower() in a for t in traps) and not correct
    is_fallback  = "don't have enough information" in a
    has_citation = bool("[source:" in a)
    return {
        "correct":      correct,
        "hallucinated": hallucinated,
        "is_fallback":  is_fallback,
        "has_citation": has_citation,
    }


def build_chain(k: int, embeddings, llm) -> RetrievalQA:
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    prompt = PromptTemplate(
        template=RAG_PROMPT, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def run():
    print("=" * 60)
    print("Ablation: Top-K Retrieved Chunks vs Answer Quality")

    if not os.path.exists(DB_DIR):
        print(f"ERROR: DB not found at '{DB_DIR}'. Run ingest.py first.")
        sys.exit(1)

    print(f"Model: {GROQ_MODEL}")
    print(f"k values: {K_VALUES}")
    print(f"Questions: {len(TEST_QUESTIONS)}\n")

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

    results = {}

    for k in K_VALUES:
        print(f"\n── k={k} ──")
        chain = build_chain(k, embeddings, llm)

        correct_count  = 0
        citation_count = 0
        fallback_count = 0
        halluc_count   = 0
        total_words    = 0
        per_q          = []

        for i, item in enumerate(TEST_QUESTIONS):
            print(f"  Q{i+1}: {item['question'][:55]}...")
            try:
                result = chain.invoke({"query": item["question"]})
                answer = result["result"]
                check  = check_answer(answer, item["ground_truth"], item["traps"])

                if check["correct"]:      correct_count  += 1
                if check["has_citation"]: citation_count += 1
                if check["is_fallback"]:  fallback_count += 1
                if check["hallucinated"]: halluc_count   += 1
                total_words += len(answer.split())

                per_q.append({
                    "question":    item["question"],
                    "ground_truth": item["ground_truth"],
                    "correct":     check["correct"],
                    "hallucinated": check["hallucinated"],
                    "is_fallback": check["is_fallback"],
                    "has_citation": check["has_citation"],
                    "answer":      answer[:300],
                })
                status = "✓" if check["correct"] else ("HALLUC" if check["hallucinated"] else "✗")
                print(f"    {status}  citation={'yes' if check['has_citation'] else 'no'}")
            except Exception as e:
                print(f"    Error: {e}")
                per_q.append({"question": item["question"], "error": str(e)})

            time.sleep(2)  # Groq rate limit

        n = len(TEST_QUESTIONS)
        results[k] = {
            "k":               k,
            "correct_rate":    round(correct_count  / n, 3),
            "citation_rate":   round(citation_count / n, 3),
            "fallback_rate":   round(fallback_count / n, 3),
            "halluc_rate":     round(halluc_count   / n, 3),
            "avg_words":       round(total_words    / n, 1),
            "per_question":    per_q,
        }

        print(f"\n  Correct   : {correct_count}/{n} = {correct_count/n:.0%}")
        print(f"  Citation  : {citation_count}/{n} = {citation_count/n:.0%}")
        print(f"  Fallback  : {fallback_count}/{n} = {fallback_count/n:.0%}")
        print(f"  Halluc    : {halluc_count}/{n}   = {halluc_count/n:.0%}")
        print(f"  Avg words : {total_words/n:.0f}")

    # ── Results table ──────────────────────────────────────────────────────────
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'k':<5} {'Correct%':<11} {'Halluc%':<10} {'Citation%':<12} {'Fallback%':<12} {'AvgWords'}")
    print("-" * 60)

    # Choose k by: highest correctness, then lowest hallucination as tiebreak
    best_k = max(K_VALUES, key=lambda k: (
        results[k]["correct_rate"],
        -results[k]["halluc_rate"],
        -results[k]["fallback_rate"],
    ))

    for k in K_VALUES:
        r      = results[k]
        marker = " ← chosen" if k == best_k else ""
        print(
            f"{k:<5} {r['correct_rate']:<11.0%} {r['halluc_rate']:<10.0%} "
            f"{r['citation_rate']:<12.0%} {r['fallback_rate']:<12.0%} "
            f"{r['avg_words']:<8.0f}{marker}"
        )


    # Save
    Path("ablations").mkdir(exist_ok=True)
    with open("ablations/results_top_k.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to ablations/results_top_k.json")


if __name__ == "__main__":
    run()
