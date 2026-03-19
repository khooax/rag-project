"""
Metrics measured:
  1.  Semantic Similarity        — cosine similarity between ans and ground truth
  2.  LLM-as-Judge Faithfulness  — does ans contradict retrieved context
  3.  LLM-as-Judge Correctness   — does ans match ground truth?
  4.  Hallucination Rate         — fraction of ans with claims not in context
  5.  Retrieval Hit Rate @K      — did any top-K chunk contain answer
  6.  Source Attribution         — which source documents were retrieved
  7.  Latency                    — end-to-end response time per query
  8.  Fallback Rate              — when did chatbot correctly say idk
  9.  Out-of-Scope Block Rate    — guardrail effectiveness
  10. Citation Rate              — does ans cite sources
"""

import os, sys, json, time, argparse, re
import numpy as np
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

#  Test set 
GOLDEN_TEST_SET = [
  {
    "question": "What is the minimum notice period if I have worked for less than 26 weeks and the contract is silent?",
    "ground_truth": "Under Section 10 of the Employment Act, the default notice period for service of less than 26 weeks is 1 day.",
    "answer_key": "1 day",
    "source_hint": "Employment Act Section 10"
  },
  {
    "question": "How many days of annual leave am I entitled to in my first year of service?",
    "ground_truth": "According to the Employment Act, employees who have served at least 3 months are entitled to 7 days of annual leave for the first year of service.",
    "answer_key": "7 days",
    "source_hint": "MOM Annual Leave"
  },
  {
    "question": "What is the overtime pay rate in Singapore for Part IV employees?",
    "ground_truth": "Overtime pay must be at least 1.5 times the hourly basic rate of pay for employees covered under Part IV of the Employment Act.",
    "answer_key": "1.5 times",
    "source_hint": "Employment Act Section 38"
  },
  {
    "question": "How many days of outpatient paid sick leave am I entitled to per year if I have served at least 6 months?",
    "ground_truth": "You are entitled to 14 days of outpatient sick leave per year, and up to 60 days of hospitalisation leave (which includes the 14 days).",
    "answer_key": "14 days",
    "source_hint": "MOM Sick Leave"
  },
  {
    "question": "When must my employer pay my salary?",
    "ground_truth": "Salary must be paid within 7 days after the end of the salary period. Overtime pay must be paid within 14 days.",
    "answer_key": "7 days",
    "source_hint": "Employment Act Section 21"
  },
  {
    "question": "How many weeks of maternity leave am I entitled to if my child is a Singapore Citizen?",
    "ground_truth": "Eligible employees are entitled to 16 weeks of government-paid maternity leave for a Singapore Citizen child.",
    "answer_key": "16 weeks",
    "source_hint": "MOM Maternity Leave"
  },
  {
    "question": "Can my employer deduct money from my salary for poor performance?",
    "ground_truth": "No. Employers can only make deductions for specific reasons under the Act (like absence or damage to goods). Poor performance is not an authorized deduction.",
    "answer_key": "No",
    "source_hint": "Employment Act Section 27"
  },
  {
    "question": "What is the maximum number of overtime hours allowed per month?",
    "ground_truth": "The maximum number of overtime hours an employee can work in a month is 72 hours.",
    "answer_key": "72 hours",
    "source_hint": "MOM Hours of Work"
  },
  {
    "question": "What is the minimum salary for a new Employment Pass (non-financial) in 2026?",
    "ground_truth": "The minimum fixed monthly salary for a new Employment Pass is $5,600, increasing with age.",
    "answer_key": "5,600",
    "source_hint": "MOM EP Eligibility"
  },
  {
    "question": "How many public holidays are there in Singapore per year?",
    "ground_truth": "There are 11 gazetted public holidays per year in Singapore.",
    "answer_key": "11",
    "source_hint": "MOM Public Holidays"
  },
  {
    "question": "How many weeks of paternity leave am I entitled to in 2026?",
    "ground_truth": "For Singapore Citizen children born on or after 1 April 2025, fathers are entitled to 4 weeks of government-paid paternity leave.",
    "answer_key": "4 weeks",
    "source_hint": "MOM Paternity Leave"
  },
  {
    "question": "What is the S Pass minimum salary for new applications in 2026?",
    "ground_truth": "The minimum fixed monthly salary for an S Pass is $3,300 (non-financial sector).",
    "answer_key": "3,300",
    "source_hint": "MOM S Pass"
  },
  {
    "question": "Is my employer required to give me a payslip?",
    "ground_truth": "Yes. Employers must issue itemised payslips within 3 working days of paying salary.",
    "answer_key": "3 working days",
    "source_hint": "MOM Payslips"
  },
  {
    "question": "What is the maximum number of working hours per week under Part IV of the Employment Act?",
    "ground_truth": "The limit is 44 ordinary hours per week for employees covered under Part IV.",
    "answer_key": "44",
    "source_hint": "MOM Hours of Work"
  },
  {
    "question": "How many days notice for someone with 3 years of service if the contract is silent?",
    "ground_truth": "For 2 to fewer than 5 years of service, the statutory minimum notice period is 2 weeks.",
    "answer_key": "2 weeks",
    "source_hint": "Employment Act Section 10"
  },
  {
    "question": "How do I file a wrongful dismissal claim?",
    "ground_truth": "You must file a claim at TADM within 1 month from your last day of employment.",
    "answer_key": "1 month",
    "source_hint": "MOM Wrongful Dismissal"
  },
  {
    "question": "How many days of annual leave in year 8 of employment?",
    "ground_truth": "From the 8th year of service onwards, the statutory minimum is 14 days of annual leave.",
    "answer_key": "14 days",
    "source_hint": "MOM Annual Leave"
  },
  {
    "question": "What is the 2026 statutory retirement age in Singapore?",
    "ground_truth": "Effective 1 July 2026, the statutory retirement age is 64 years.",
    "answer_key": "64",
    "source_hint": "MOM Retirement"
  },
  {
    "question": "What is the 2026 statutory re-employment age in Singapore?",
    "ground_truth": "Effective 1 July 2026, the statutory re-employment age is 69 years.",
    "answer_key": "69",
    "source_hint": "MOM Re-employment"
  },
  {
    "question": "Does Part IV of the Employment Act cover a manager earning $3,000?",
    "ground_truth": "No. Managers and executives are not covered by Part IV of the Employment Act, regardless of salary.",
    "answer_key": "No",
    "source_hint": "MOM Employment Act Coverage"
  },
  {
    "question": "How many days of sick leave can I take if I have worked for exactly 4 months?",
    "ground_truth": "After 4 months of service, you are entitled to 8 days of outpatient sick leave and 30 days of hospitalisation leave.",
    "answer_key": "8 days",
    "source_hint": "MOM Sick Leave"
  },
  {
    "question": "What is the Employment Assistance Payment (EAP) for a retrenched older worker in 2026?",
    "ground_truth": "The EAP is 3.5 months' salary, with a minimum of $6,250 and a maximum of $14,750.",
    "answer_key": "3.5 months",
    "source_hint": "MOM Re-employment"
  },
  {
    "question": "How long must a Manager or Executive serve to file for wrongful dismissal with notice?",
    "ground_truth": "Managers and executives must have served at least 6 months to be eligible to file a claim for wrongful dismissal with notice.",
    "answer_key": "6 months",
    "source_hint": "MOM Wrongful Dismissal"
  },
  {
    "question": "Can a job advertisement specify 'Singaporeans only' under Tripartite Guidelines?",
    "ground_truth": "No. Under the Fair Consideration Framework and Tripartite Guidelines, advertisements should avoid nationality bias and be based on merit.",
    "answer_key": "No",
    "source_hint": "Tripartite Guidelines"
  },
  {
    "question": "How soon must a workplace fatality be reported to MOM?",
    "ground_truth": "Employers must notify MOM immediately (within 24 hours) of a workplace fatality.",
    "answer_key": "immediately",
    "source_hint": "MOM WSH Reporting"
  },
  {
    "question": "What is the rest day entitlement for a workman covered under Part IV?",
    "ground_truth": "An employee covered under Part IV is entitled to one rest day per week, which is one whole day (24 hours) without pay.",
    "answer_key": "1 rest day",
    "source_hint": "Employment Act Section 36"
  },
  {
    "question": "Must an employer provide Key Employment Terms (KETs) in writing?",
    "ground_truth": "Yes. Employers must provide written KETs to employees covered by the Employment Act within 14 days of the start of employment.",
    "answer_key": "14 days",
    "source_hint": "MOM Key Employment Terms"
  },
  {
    "question": "What is the maximum time an employer has to respond to a formal Flexible Work Arrangement (FWA) request?",
    "ground_truth": "Under the Tripartite Guidelines on Flexible Work Arrangement Requests, employers must provide a written decision within 2 months of receiving a formal request.",
    "answer_key": "2 months",
    "source_hint": "MOM Flexible Work Arrangements"
  },
  {
    "question": "How many weeks of Shared Parental Leave (SPL) can parents share for a child born on or after 1 April 2026?",
    "ground_truth": "For children born on or after 1 April 2026, eligible parents are entitled to 10 weeks of government-paid Shared Parental Leave.",
    "answer_key": "10 weeks",
    "source_hint": "MOM Shared Parental Leave"
  },
  {
    "question": "What is the maximum medical expense claim limit for work injuries occurring from 1 November 2025 onwards?",
    "ground_truth": "For work injuries occurring from 1 November 2025, the maximum medical expense claim limit under WICA is $53,000.",
    "answer_key": "53,000",
    "source_hint": "MOM WICA Limits"
  }
]

OUT_OF_SCOPE_TESTS = [
    "What is the best recipe for chicken rice?",
    "What is the weather in Singapore today?",
    "Can you help me write a Python script?",
    "Who won the World Cup?",
    "What is the price of Bitcoin?",
]

#  LLM Judge prompts 

FAITHFULNESS_PROMPT = """You are evaluating a RAG system. Judge if the ANSWER is faithful to the CONTEXT.
Faithful = every claim in the answer is supported by the context.
CONTEXT: {context}
ANSWER: {answer}
Respond ONLY with JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are evaluating a QA system about Singapore employment law.
QUESTION: {question}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}
Score correctness 0.0-1.0. Respond ONLY with JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

HALLUCINATION_PROMPT = """You are a fact-checker for Singapore employment law.
Check if the ANSWER contains hallucinated (wrong/fabricated) facts vs GROUND TRUTH.
QUESTION: {question}
GROUND TRUTH: {ground_truth}
CONTEXT: {context}
ANSWER: {answer}
Respond ONLY with JSON: {{"hallucinated": <true/false>, "confidence": <0.0-1.0>, "example": "<specific wrong claim or null>"}}"""


#  Metric functions 

def measure_latency(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start

def compute_semantic_similarity(answer, ground_truth, embeddings):
    """
    Cosine similarity between answer and ground truth embeddings.
    Range 0-1. Captures paraphrases unlike exact string matching.
    Uses the same embedding model as the retriever (MiniLM) — free, no API.
    """
    vecs = embeddings.embed_documents([answer, ground_truth])
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])


def check_retrieval_hit(retrieved_docs, answer_key, source_hint):
    """
    Retrieval Hit Rate: did the correct chunk get retrieved?
    Checks at k=1, 3, 5 so you can see how rank affects coverage.
    """
    hits = {}
    for k in [1, 3, 5]:
        top_k_text = " ".join(d.page_content.lower() for d in retrieved_docs[:k])
        hits[f"hit@{k}"] = answer_key.lower() in top_k_text

    combined = " ".join(d.page_content.lower() for d in retrieved_docs)
    combined_meta = " ".join(str(d.metadata.get("source","")).lower() for d in retrieved_docs)

    return {
        **hits,
        "key_hit":    answer_key.lower() in combined,
        "source_hit": source_hint.lower() in combined or source_hint.lower() in combined_meta,
        "sources_retrieved": [
            {"source": d.metadata.get("source","unknown"), "url": d.metadata.get("url","")}
            for d in retrieved_docs
        ]
    }


def call_judge(prompt, judge_llm, retries=2):
    """
    Parse LLM judge response robustly.
    Handles: markdown fences, extra text before/after JSON, retries on failure.
    """
    for attempt in range(retries):
        try:
            response = judge_llm.invoke(prompt).content.strip()
            # Strip markdown code fences if present
            response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
            # Extract first JSON object
            m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if m:
                return json.loads(m.group())
        except json.JSONDecodeError:
            time.sleep(1)
        except Exception:
            break
    return None


def llm_judge(question, answer, ground_truth, context_docs, judge_llm):
    """Run all three LLM judges in one function. Returns dict with all scores."""
    ctx = "\n\n".join(d.page_content for d in context_docs[:3])[:2000]

    faith_result = call_judge(
        FAITHFULNESS_PROMPT.format(context=ctx, answer=answer), judge_llm
    ) or {"score": 0.5, "reason": "judge failed"}

    time.sleep(2.0)  # increased: Groq free tier rate limit

    correct_result = call_judge(
        CORRECTNESS_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer), judge_llm
    ) or {"score": 0.5, "reason": "judge failed"}

    time.sleep(2.0)  # increased: Groq free tier rate limit

    halluc_result = call_judge(
        HALLUCINATION_PROMPT.format(question=question, ground_truth=ground_truth, context=ctx, answer=answer), judge_llm
    ) or {"hallucinated": False, "confidence": 0.0, "example": None}

    return {
        "faithfulness":  float(faith_result.get("score", 0.5)),
        "faith_reason":  faith_result.get("reason", ""),
        "correctness":   float(correct_result.get("score", 0.5)),
        "correct_reason": correct_result.get("reason", ""),
        "hallucinated":  str(halluc_result.get("hallucinated", False)).lower() == "true",
        "halluc_conf":   float(halluc_result.get("confidence", 0.0)),
        "halluc_example": halluc_result.get("example"),
    }


#  Main eval

def run_evaluation(quick=False, use_llm_judge=True):
    print("=" * 60)
    print("SG Employment Chatbot — Comprehensive Evaluation")
    print("=" * 60)

    from rag_pipeline import ask, get_llm

    test_set = GOLDEN_TEST_SET[:10] if quick else GOLDEN_TEST_SET
    print(f"Questions : {len(test_set)}")
    print(f"LLM judge : {'enabled (self-judge — same model, note bias)' if use_llm_judge else 'disabled'}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    judge_llm = get_llm() if use_llm_judge else None

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "llama-3.1-8b-instant via Groq",
        "quick_mode": quick, "llm_judge": use_llm_judge,
        "per_question": [], "out_of_scope": [], "summary": {}
    }

    acc = {k: [] for k in [
        "sem_sim", "faithfulness", "correctness", "hallucinated",
        "hit@1", "hit@3", "hit@5", "latency", "has_citation", "is_fallback"
    ]}

    print(f"  {'Q':<4} {'Sem.Sim':<9} {'Faith.':<9} {'Correct':<9} {'Halluc.':<9} {'Hit@5':<7} {'Lat(s)'}")
    print("  " + "-" * 60)

    for i, item in enumerate(test_set):
        q, gt, ak, sh = item["question"], item["ground_truth"], item["answer_key"], item["source_hint"]

        try:
            raw, latency = measure_latency(ask, q)
            if len(raw) == 4:
                answer, sources, blocked, trace = raw
            else:
                answer, sources, blocked = raw
                trace = {"original": q, "final": q, "was_rewritten": False}
        except Exception as e:
            print(f"  Q{i+1:02d} ERROR: {e}")
            results["per_question"].append({"question": q, "error": str(e)})
            time.sleep(2)
            continue

        if blocked:
            print(f"  Q{i+1:02d} BLOCKED by guardrail — check keywords")
            continue

        # core metrics
        sem_sim  = compute_semantic_similarity(answer, gt, embeddings)
        retrieval = check_retrieval_hit(sources, ak, sh)
        has_cit  = bool(re.search(r'\[Source:', answer, re.IGNORECASE))
        is_fall  = "don't have enough information" in answer.lower()

        # LLM judge or heuristic fallback
        if use_llm_judge and sources:
            jscores = llm_judge(q, answer, gt, sources, judge_llm)
        else:
            jscores = {
                "faithfulness": 1.0 if retrieval["key_hit"] else 0.5,
                "faith_reason": "heuristic",
                "correctness":  1.0 if ak.lower() in answer.lower() else 0.0,
                "correct_reason": "heuristic",
                "hallucinated": False, "halluc_conf": 0.0, "halluc_example": None,
            }

        # accumulate
        acc["sem_sim"].append(sem_sim)
        acc["faithfulness"].append(jscores["faithfulness"])
        acc["correctness"].append(jscores["correctness"])
        acc["hallucinated"].append(int(jscores["hallucinated"]))
        acc["hit@1"].append(int(retrieval["hit@1"]))
        acc["hit@3"].append(int(retrieval["hit@3"]))
        acc["hit@5"].append(int(retrieval["hit@5"]))
        acc["latency"].append(latency)
        acc["has_citation"].append(int(has_cit))
        acc["is_fallback"].append(int(is_fall))

        print(
            f"  Q{i+1:02d}  "
            f"{sem_sim:<9.3f}"
            f"{jscores['faithfulness']:<9.2f}"
            f"{jscores['correctness']:<9.2f}"
            f"{'YES' if jscores['hallucinated'] else 'no':<9}"
            f"{'✓' if retrieval['hit@5'] else '✗':<7}"
            f"{latency:.2f}s"
        )

        results["per_question"].append({
            "question": q, "ground_truth": gt, "answer": answer,
            "query_trace": trace,
            "metrics": {
                "semantic_similarity": round(sem_sim, 4),
                "faithfulness":        round(jscores["faithfulness"], 4),
                "correctness":         round(jscores["correctness"], 4),
                "hallucinated":        jscores["hallucinated"],
                "halluc_example":      jscores["halluc_example"],
                "has_citation":        has_cit,
                "is_fallback":         is_fall,
                "latency_seconds":     round(latency, 3),
            },
            "retrieval": retrieval,
            "llm_judge_reasons": {
                "faithfulness": jscores["faith_reason"],
                "correctness":  jscores["correct_reason"],
            }
        })

        time.sleep(5.0 if use_llm_judge else 1.5)  # 3 judge calls + generation = ~6 API calls per Q

    # Out-of-scope test
    print("\nOut-of-scope guardrail test:")
    blocked_count = 0
    for query in OUT_OF_SCOPE_TESTS:
        try:
            _, _, was_blocked, _ = ask(query)
            blocked_count += int(was_blocked)
            results["out_of_scope"].append({"query": query, "correctly_blocked": was_blocked})
            print(f"  {'✓' if was_blocked else '✗'} {'Blocked' if was_blocked else 'NOT blocked'}: {query}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1.5)

    # Summary
    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
    def pct(lst): return f"{avg(lst)*100:.1f}%"
    n = len(acc["latency"])

    summary = {
        "n_questions":                n,
        "llm_judge_enabled":          use_llm_judge,
        "avg_semantic_similarity":    avg(acc["sem_sim"]),
        "avg_faithfulness":           avg(acc["faithfulness"]),
        "avg_correctness":            avg(acc["correctness"]),
        "hallucination_rate":         pct(acc["hallucinated"]),
        "retrieval_hit_rate_at_1":    pct(acc["hit@1"]),
        "retrieval_hit_rate_at_3":    pct(acc["hit@3"]),
        "retrieval_hit_rate_at_5":    pct(acc["hit@5"]),
        "citation_rate":              pct(acc["has_citation"]),
        "fallback_rate":              pct(acc["is_fallback"]),
        "out_of_scope_block_rate":    f"{blocked_count}/{len(OUT_OF_SCOPE_TESTS)}",
        "avg_latency_seconds":        avg(acc["latency"]),
        "p50_latency_seconds":        round(float(np.percentile(acc["latency"], 50)), 3) if acc["latency"] else 0,
        "p95_latency_seconds":        round(float(np.percentile(acc["latency"], 95)), 3) if acc["latency"] else 0,
    }
    results["summary"] = summary

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n── Answer Quality {'─'*42}")
    print(f"  Semantic Similarity (cosine, answer vs ground truth): {summary['avg_semantic_similarity']:.4f}")
    print(f"  LLM Judge Faithfulness (grounded in context?)       : {summary['avg_faithfulness']:.4f}")
    print(f"  LLM Judge Correctness  (matches ground truth?)      : {summary['avg_correctness']:.4f}")
    print(f"  Hallucination Rate                                  : {summary['hallucination_rate']}")
    print(f"\n── Retrieval Quality {'─'*39}")
    print(f"  Hit Rate @ 1  : {summary['retrieval_hit_rate_at_1']}")
    print(f"  Hit Rate @ 3  : {summary['retrieval_hit_rate_at_3']}")
    print(f"  Hit Rate @ 5  : {summary['retrieval_hit_rate_at_5']}")
    print(f"\n── Surface Metrics {'─'*41}")
    print(f"  Citation Rate          : {summary['citation_rate']}")
    print(f"  Fallback Rate          : {summary['fallback_rate']}")
    print(f"  Out-of-Scope Block Rate: {summary['out_of_scope_block_rate']}")
    print(f"\n── Latency {'─'*49}")
    print(f"  Avg : {summary['avg_latency_seconds']:.2f}s  |  p50 : {summary['p50_latency_seconds']:.2f}s  |  p95 : {summary['p95_latency_seconds']:.2f}s")

    # Worst questions
    scored = [r for r in results["per_question"] if "metrics" in r]
    worst  = sorted(scored, key=lambda r: r["metrics"]["correctness"])[:3]
    if worst:
        print(f"\n── 3 Lowest-Scoring Questions {'─'*30}")
        for r in worst:
            m = r["metrics"]
            print(f"  [{m['correctness']:.2f}] {r['question'][:65]}")
            print(f"         sem={m['semantic_similarity']:.3f} faith={m['faithfulness']:.2f} hit@5={'✓' if r['retrieval'].get('hit@5') else '✗'}")

    if use_llm_judge:
        print(f"\n⚠  LLM judge note: Faithfulness + correctness scored by the same model that")
        print(f"   generated answers (self-judge bias — scores likely inflated).")
        print(f"   For rigorous eval, use a different model as judge.")

    # Save
    Path("eval").mkdir(exist_ok=True)
    with open("eval/eval_report.json",  "w") as f: json.dump(results,  f, indent=2)
    with open("eval/eval_metrics.json", "w") as f: json.dump(summary, f, indent=2)

    summary_txt = f"""SG Employment Chatbot — Eval 
Generated : {results['timestamp']}
Questions : {n} | LLM Judge: {use_llm_judge}
{"="*55}
Semantic Similarity  : {summary['avg_semantic_similarity']:.4f}
Faithfulness         : {summary['avg_faithfulness']:.4f}
Correctness          : {summary['avg_correctness']:.4f}
Hallucination Rate   : {summary['hallucination_rate']}
Hit Rate @1/@3/@5    : {summary['retrieval_hit_rate_at_1']} / {summary['retrieval_hit_rate_at_3']} / {summary['retrieval_hit_rate_at_5']}
Citation Rate        : {summary['citation_rate']}
Fallback Rate        : {summary['fallback_rate']}
Out-of-Scope Block   : {summary['out_of_scope_block_rate']}
Avg / p50 / p95 lat  : {summary['avg_latency_seconds']:.2f}s / {summary['p50_latency_seconds']:.2f}s / {summary['p95_latency_seconds']:.2f}s
"""
    with open("eval/eval_summary.txt", "w") as f: f.write(summary_txt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true", help="10 questions only")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM judge")
    args = parser.parse_args()
    run_evaluation(quick=args.quick, use_llm_judge=not args.no_llm)
