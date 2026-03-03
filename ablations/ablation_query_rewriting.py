"""
ablation_query_rewriting.py

Tests whether query rewriting improves retrieval quality.

Fix vs v1:
  - v1 called a raw LLM directly for rewriting, not the actual
    preprocess_query_with_trace() pipeline. This meant the ablation was
    testing a different rewriter than what runs in production — the results
    were meaningless as a comparison.
  - v2 calls preprocess_query_with_trace() from query_understanding.py,
    so the ablation reflects exactly what your pipeline does.
  - Added a "no_change" tracking column so you can see cases where the
    pipeline correctly decided NOT to rewrite (and whether that helped).
  - Result table now shows delta clearly (+ or -) so regressions are obvious.

Run: python ablation_query_rewriting.py
Output: ablations/results_query_rewriting.json
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
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR      = "./db"
TOP_K       = 5

MESSY_QUERIES = [
    # Typos
    {
        "raw":          "anual leav entitlement singapore",
        "expected_kw":  ["annual leave", "7 days", "year"],
        "category":     "typo",
    },
    {
        "raw":          "pregant woman matarnity leave",
        "expected_kw":  ["maternity", "weeks", "citizen"],
        "category":     "typo",
    },
    {
        "raw":          "wrongful dimissal how to cliam",
        "expected_kw":  ["wrongful", "dismissal", "claim"],
        "category":     "typo",
    },
    # Abbreviations (rule-based expansion, should NOT need LLM)
    {
        "raw":          "OT pay rate",
        "expected_kw":  ["overtime", "1.5", "rate"],
        "category":     "abbreviation",
    },
    {
        "raw":          "MC leave how many days",
        "expected_kw":  ["sick leave", "14 days", "medical"],
        "category":     "abbreviation",
    },
    # Singlish / informal
    {
        "raw":          "my boss never pay me salary lah what can i do",
        "expected_kw":  ["salary", "7 days", "mom"],
        "category":     "singlish/informal",
    },
    {
        "raw":          "kena retrench what happen",
        "expected_kw":  ["retrench", "benefit", "notice"],
        "category":     "singlish/informal",
    },
    {
        "raw":          "boss anyhow deduct my pay can meh",
        "expected_kw":  ["deduct", "salary", "unauthorised"],
        "category":     "singlish/informal",
    },
    # Code-switching
    {
        "raw":          "CPF berapa percent employer kena pay",
        "expected_kw":  ["cpf", "employer", "%"],
        "category":     "code-switching",
    },
    # Vague
    {
        "raw":          "what are my rights",
        "expected_kw":  ["employment", "rights", "act"],
        "category":     "vague",
    },
    # Typo + abbreviation
    {
        "raw":          "EP minium sallary",
        "expected_kw":  ["employment pass", "5,000", "salary"],
        "category":     "typo + abbreviation",
    },
    # Clean query — should NOT be rewritten, should perform the same or better
    {
        "raw":          "notice period resign how long",
        "expected_kw":  ["notice", "weeks", "resign"],
        "category":     "informal",
    },
]


def precision_at_k(retrieved: list, expected_kw: list[str], k: int = 5) -> float:
    chunks   = retrieved[:k]
    relevant = sum(
        1 for c in chunks
        if any(kw.lower() in c.page_content.lower() for kw in expected_kw)
    )
    return relevant / len(chunks) if chunks else 0.0


def run():
    print("=" * 65)
    print("Ablation: Query Rewriting (Production Pipeline) vs Raw Query")
    print("=" * 65)
    print("Using: query_understanding_new.preprocess_query_with_trace()\n")

    if not os.path.exists(DB_DIR):
        print(f"ERROR: DB not found at '{DB_DIR}'. Run ingest.py first.")
        sys.exit(1)

    # Import production pipeline — this is the key fix vs v1
    try:
        from query_understanding_new import preprocess_query_with_trace
    except ImportError:
        print("ERROR: Cannot import query_understanding_new. Run from the project root.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db        = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    results = []

    for item in MESSY_QUERIES:
        raw = item["raw"]
        print(f"  [{item['category']}] \"{raw}\"")

        # Condition A: raw query, no rewriting
        retrieved_raw = retriever.invoke(raw)
        p_raw         = precision_at_k(retrieved_raw, item["expected_kw"])

        # Condition B: production pipeline rewrite
        trace     = preprocess_query_with_trace(raw)
        rewritten = trace["final"]
        was_rewritten = trace["was_rewritten"]
        used_llm  = trace["used_llm"]

        retrieved_rw = retriever.invoke(rewritten)
        p_rw         = precision_at_k(retrieved_rw, item["expected_kw"])

        delta    = p_rw - p_raw
        outcome  = "↑ improved" if delta > 0 else ("→ same" if delta == 0 else "↓ worse")
        rewrite_tag = "(LLM)" if used_llm else "(rule)" if was_rewritten else "(no rewrite)"

        print(f"    Raw        : P@5 = {p_raw:.2f}")
        print(f"    Rewritten  : \"{rewritten}\" {rewrite_tag}")
        print(f"    After rw   : P@5 = {p_rw:.2f}  {outcome}  (Δ={delta:+.2f})")

        results.append({
            "raw_query":       raw,
            "rewritten_query": rewritten,
            "was_rewritten":   was_rewritten,
            "used_llm":        used_llm,
            "category":        item["category"],
            "precision_raw":   round(p_raw, 3),
            "precision_rw":    round(p_rw,  3),
            "delta":           round(delta, 3),
            "improved":        delta > 0,
            "unchanged":       delta == 0,
            "worse":           delta < 0,
        })

        time.sleep(0.5)  # small pause between LLM rewrite calls

    # ── Summary by category ────────────────────────────────────────────────────
    categories = sorted(set(r["category"] for r in results))
    print("\n" + "=" * 65)
    print("RESULTS BY CATEGORY")
    print("=" * 65)
    print(f"{'Category':<22} {'Raw P@5':<10} {'RW P@5':<10} {'Delta':<8} {'Improved?'}")
    print("-" * 65)

    for cat in categories:
        cat_r    = [r for r in results if r["category"] == cat]
        avg_raw  = sum(r["precision_raw"] for r in cat_r) / len(cat_r)
        avg_rw   = sum(r["precision_rw"]  for r in cat_r) / len(cat_r)
        delta    = avg_rw - avg_raw
        improved = sum(r["improved"] for r in cat_r)
        worse    = sum(r["worse"]    for r in cat_r)
        print(f"{cat:<22} {avg_raw:<10.3f} {avg_rw:<10.3f} {delta:<+8.3f} {improved}/{len(cat_r)} improved, {worse}/{len(cat_r)} worse")

    overall_raw = sum(r["precision_raw"] for r in results) / len(results)
    overall_rw  = sum(r["precision_rw"]  for r in results) / len(results)
    total_imp   = sum(r["improved"]  for r in results)
    total_worse = sum(r["worse"]     for r in results)
    total_same  = sum(r["unchanged"] for r in results)
    overall_delta = overall_rw - overall_raw

    print(f"\n{'OVERALL':<22} {overall_raw:<10.3f} {overall_rw:<10.3f} {overall_delta:<+8.3f} "
          f"{total_imp}/{len(results)} improved")

    print(f"\n── Breakdown ──────────────────────────────────────────────")
    print(f"  Improved : {total_imp}")
    print(f"  Same     : {total_same}")
    print(f"  Worse    : {total_worse}")
    print(f"  Net precision gain: {overall_delta:+.3f}")

    # Flag regressions explicitly
    regressions = [r for r in results if r["worse"]]
    if regressions:
        print(f"\n⚠️  Regressions ({len(regressions)}) — rewriting hurt these queries:")
        for r in regressions:
            print(f"    [{r['category']}] \"{r['raw_query']}\"")
            print(f"      Rewritten to: \"{r['rewritten_query']}\"")
            print(f"      P@5: {r['precision_raw']:.2f} → {r['precision_rw']:.2f}")

    # Save
    Path("ablations").mkdir(exist_ok=True)
    with open("ablations/results_query_rewriting.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to ablations/results_query_rewriting.json")


if __name__ == "__main__":
    run()
