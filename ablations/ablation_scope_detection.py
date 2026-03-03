"""
Benchmarks 2 out-of-scope detection approaches on the same test set:
  A) Semantic only 
  B) Tiered semantic + LLM judge 
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"

_THIS_FILE   = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent if _THIS_FILE.parent.name != "ablations" else _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Test set
# None of these queries appear verbatim in _IN_SCOPE or _OUT_OF_SCOPE in rag_pipeline.py to prevent data leakage.
# label: True = should be blocked (out-of-scope), False = should pass through (in-scope)

TEST_CASES = [
    # In-scope 
    ("What is the notice period if I resign after 2 years?",                False),
    ("How many days of annual leave am I entitled to?",                     False),
    ("My employer has not paid my salary for 3 weeks. What can I do?",      False),
    ("What are the CPF contribution rates for someone aged 30?",            False),
    ("Am I entitled to maternity leave if I am on a work permit?",          False),
    ("What is the minimum salary for an Employment Pass?",                  False),
    ("Can my employer deduct money from my pay for being late?",            False),
    ("How do I file a wrongful dismissal claim?",                           False),
    ("What is the maximum overtime hours per month?",                       False),
    ("What happens to my CPF if I am retrenched?",                          False),
    # Singlish (in-scope)
    ("Boss never pay me salary lah what to do",                             False),
    ("OT pay how to calculate one",                                         False),
    ("MC leave how many days can take per year",                            False),
    ("Kena retrench suddenly what are my rights",                           False),
    # Ambiguous-but-in-scope
    ("Can my employer monitor my work laptop?",                             False),
    ("Can I work part-time while on maternity leave?",                      False),
    ("My employer says I am a contractor not an employee. What are my rights?", False),
    ("Can I be fired for taking medical leave?",                            False),

    #  Out-of-scope (SHOULD be blocked) 
    ("What is the best chicken rice recipe in Singapore?",                  True),
    ("Can you help me write a Python function?",                            True),
    ("Who won the World Cup last year?",                                    True),
    ("What is the weather like in Singapore tomorrow?",                     True),
    ("How do I invest in Bitcoin?",                                         True),
    ("What are the symptoms of diabetes?",                                  True),
    ("How do I apply for a student visa to Australia?",                     True),
    ("Write me a poem about Singapore",                                     True),
    # Subtly out-of-scope 
    ("How do I register a company in Singapore?",                           True),
    ("What is the corporate income tax rate?",                              True),
    ("How do I apply for a HDB flat?",                                      True),
    ("What grants are available for startups?",                             True),
    ("How do I sue someone for defamation?",                                True),
    ("What is the GST rate in Singapore?",                                  True),
    ("How do I open a bank account in Singapore?",                          True),
    # Tricky — looks employment-related but isn't
    ("How much does a lawyer charge per hour in Singapore?",                True),
    ("What is the minimum salary to qualify for a bank loan?",              True),
    ("How do I set up payroll software for my 50-person company?",          True),
]

def evaluate(name: str, classify_fn, test_cases: list) -> dict:
    tp = fp = tn = fn = 0
    latencies = []
    misses    = []

    for query, should_block in test_cases:
        t0 = time.perf_counter()
        try:
            blocked = classify_fn(query)
        except Exception as e:
            blocked = False
            print(f"  ERROR classifying '{query}': {e}")
        latencies.append(time.perf_counter() - t0)

        if should_block     and     blocked:  tp += 1
        elif not should_block and not blocked: tn += 1
        elif not should_block and blocked:
            fp += 1
            misses.append(("FALSE POSITIVE", query))
        else:
            fn += 1
            misses.append(("FALSE NEGATIVE", query))

    n_pos = sum(1 for _, s in test_cases if s)
    n_neg = sum(1 for _, s in test_cases if not s)

    return {
        "name":           name,
        "n":              len(test_cases),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr":            round(tp / n_pos, 3) if n_pos else 0,
        "tnr":            round(tn / n_neg, 3) if n_neg else 0,
        "fpr":            round(fp / n_neg, 3) if n_neg else 0,
        "fnr":            round(fn / n_pos, 3) if n_pos else 0,
        "accuracy":       round((tp + tn) / len(test_cases), 3),
        "f1":             round(2*tp / (2*tp + fp + fn), 3) if (2*tp + fp + fn) > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)] * 1000, 2),
        "misses":         misses,
    }


def run():
    print("=" * 70)
    print("Ablation: Out-of-Scope Detection — 3 Approaches")
    print("=" * 70)
    n_oos = sum(1 for _, s in TEST_CASES if s)
    n_ins = sum(1 for _, s in TEST_CASES if not s)
    print(f"Test set: {len(TEST_CASES)} queries ({n_oos} out-of-scope, {n_ins} in-scope)\n")

    # Import production functions from rag_pipeline
    try:
        import rag_pipeline_new as rp
    except ImportError:
        print("ERROR: Cannot import rag_pipeline. Run from project root.")
        sys.exit(1)

    # Approach 1: semantic NN
    # Patch out _scope_judge_llm to always return False (pass-through)
    original_tier3 = rp._scope_judge_llm
    rp._scope_judge_llm = lambda q: False   # disable LLM tier

    result_a = evaluate("Semantic (no LLM)", rp.is_out_of_scope, TEST_CASES)
    rp._scope_judge_llm = original_tier3    # restore
    print(f"  Accuracy={result_a['accuracy']:.0%}  FPR={result_a['fpr']:.0%}  FNR={result_a['fnr']:.0%}")

    # ── Approach: Full tiered (with LLM judge) ─────────────────────────
    llm_calls = [0]
    original_tier3 = rp._scope_judge_llm
    def counting_tier3(query: str) -> bool:
        llm_calls[0] += 1
        return original_tier3(query)
    rp._scope_judge_llm = counting_tier3

    result_b = evaluate("C: Full tiered", rp.is_out_of_scope, TEST_CASES)
    rp._scope_judge_llm = original_tier3    # restore
    result_b["llm_calls"]     = llm_calls[0]
    result_b["llm_call_rate"] = f"{llm_calls[0]}/{len(TEST_CASES)}"
    print(f"  Accuracy={result_b['accuracy']:.0%}  FPR={result_b['fpr']:.0%}  FNR={result_b['fnr']:.0%}  LLM calls={llm_calls[0]}/{len(TEST_CASES)}")

    #  Results 
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Approach':<25} {'Acc':>5} {'TPR':>6} {'TNR':>6} {'FPR':>6} {'FNR':>6} {'F1':>6}  Latency")
    print("-" * 70)

    for r in [result_a, result_b]:
        llm_note = f"  [{r.get('llm_call_rate','')} LLM calls]" if "llm_calls" in r else ""
        print(
            f"{r['name']:<25}"
            f"{r['accuracy']:>5.0%}"
            f"{r['tpr']:>6.0%}"
            f"{r['tnr']:>6.0%}"
            f"{r['fpr']:>6.0%}"
            f"{r['fnr']:>6.0%}"
            f"{r['f1']:>6.3f}"
            f"  {r['avg_latency_ms']:.1f}ms{llm_note}"
        )

    # Pring misclassified samples
    print("\n Misclassifications ────────────────────────────────────────────")
    for r in [result_a, result_b]:
        if r["misses"]:
            print(f"\n  [{r['name']}] ({len(r['misses'])} misses):")
            for label, q in r["misses"]:
                print(f"    [{label}] {q}")
        else:
            print(f"\n  [{r['name']}] No misclassifications ✓")

    # Save
    Path("ablations").mkdir(exist_ok=True)
    all_results = {"a": result_a, "b": result_b}
    with open("ablations/results_scope_detection.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to ablations/results_scope_detection.json")


if __name__ == "__main__":
    run()
