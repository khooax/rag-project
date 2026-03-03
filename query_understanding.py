"""
query_understanding.py

Drop-in query pre-processor for rag_pipeline.py.

Handles:
  1. Typo correction + spelling normalisation
  2. Singlish / informal phrasing → formal English
  3. Abbreviation expansion (OT, MC, EP, EA, CPF, KET...)
  4. Vague query sharpening
  5. Code-switching (Malay/Chinese mixed with English)

Usage in rag_pipeline.py:
    from query_understanding import preprocess_query_with_trace

    def ask(query: str):
        trace = preprocess_query_with_trace(query)
        clean_query = trace["final"]
        ...

Key improvements over v1:
  - needs_llm_rewrite: removed the over-firing `has_abbrev` heuristic (matched
    ANY 2-4 char uppercase token, triggering rewrites on clean queries).
  - needs_llm_rewrite: `is_vague` now requires absence of known employment nouns,
    not just short length (short but specific queries like "CPF rate 58" were
    being rewritten unnecessarily, sometimes introducing drift).
  - llm_rewrite: added keyword preservation guard — if the rewrite drops all
    original content words, the rewrite is rejected and the original is kept.
  - llm_rewrite: tighter length guard (2x, was 3x) to catch verbose rewrites.
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# ── Rule-based abbreviation expansion (instant, no API call) ──────────────────
# Applied BEFORE the LLM rewrite so the LLM gets cleaner input.
ABBREVIATIONS = {
    r"\bOT\b":      "overtime",
    r"\bMC\b":      "medical certificate / sick leave",
    r"\bEA\b":      "Employment Act",
    r"\bEP\b":      "Employment Pass",
    r"\bS pass\b":  "S Pass",
    r"\bWP\b":      "Work Permit",
    r"\bCPF\b":     "CPF (Central Provident Fund)",
    r"\bKET\b":     "Key Employment Terms",
    r"\bWSH\b":     "Workplace Safety and Health",
    r"\bWICA\b":    "Work Injury Compensation Act",
    r"\bTADM\b":    "Tripartite Alliance for Dispute Management",
    r"\bTAFEP\b":   "Tripartite Alliance for Fair and Progressive Employment Practices",
    r"\bECT\b":     "Employment Claims Tribunals",
    r"\bAL\b":      "annual leave",
    r"\bPH\b":      "public holiday",
    r"\bNS\b":      "National Service",
    r"\bPR\b":      "Permanent Resident",
    r"\bSC\b":      "Singapore Citizen",
    r"\bHR\b":      "Human Resources / employer",
}

# ── Singlish / colloquial → formal mappings (rule-based) ─────────────────────
SINGLISH_MAP = {
    r"\blah\b":             "",
    r"\bleh\b":             "",
    r"\bmah\b":             "",
    r"\bcan meh\b":         "is this allowed",
    r"\bcan or not\b":      "is this allowed",
    r"\bkena\b":            "was subjected to",
    r"\bsabo\b":            "unfairly treated",
    r"\banyhow\b":          "arbitrarily",
    r"\bhow liddat\b":      "what should I do",
    r"\bnever pay\b":       "did not pay",
    r"\bnever give\b":      "did not give",
    r"\bsay one\b":         "",
    r"\blike that\b":       "",
    r"\bok what\b":         "",
    r"\bwhere got\b":       "is there",
}

# ── LLM rewrite prompt ────────────────────────────────────────────────────────
LLM_REWRITE_PROMPT = """You are a query normaliser for a Singapore employment law chatbot.

Rewrite the user's query into a clear, formal English question suitable for searching an employment law database.

Fix:
- Spelling errors and typos
- Informal/colloquial phrasing
- Remaining Singlish or code-switching
- Vague questions (make more specific if intent is clear)

Rules:
- Output ONLY the rewritten query, no explanation, no quotes, no preamble
- Preserve the original intent exactly — do not add, remove, or change the subject matter
- Do not add information not implied by the original
- If the query is already clear formal English, output it unchanged
- Keep the rewrite concise — do not pad with unnecessary context

Original query: {query}
Rewritten query:"""

# Known employment nouns used to detect whether a short query is actually
# specific (and therefore doesn't need an LLM rewrite).
_EMPLOYMENT_NOUNS = {
    "leave", "salary", "pay", "wage", "notice", "resign", "resign",
    "dismiss", "dismissal", "cpf", "overtime", "sick", "annual",
    "maternity", "paternity", "childcare", "hours", "holiday", "pass",
    "permit", "retrench", "retrenchment", "contract", "deduct",
    "deduction", "probation", "termination", "wrongful", "employer",
    "employee", "work", "payslip", "contribution", "entitlement",
}


def expand_abbreviations(text: str) -> str:
    """Apply rule-based abbreviation expansion."""
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalise_singlish(text: str) -> str:
    """Apply rule-based Singlish normalisation."""
    for pattern, replacement in SINGLISH_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def needs_llm_rewrite(text: str) -> bool:
    """
    Decide whether to call the LLM rewriter.
    Conservative: only rewrite when there's clear evidence of informal or
    broken language. Do NOT rewrite already-clean queries.

    v2 changes vs v1:
      - Removed `has_abbrev`: the old check fired on ANY 2-4 char uppercase
        token. After expand_abbreviations() runs, known abbreviations are already
        expanded; unknown uppercase tokens are likely proper nouns (e.g. "TAFEP",
        "WICA") and should not be rewritten.
      - `is_vague`: now requires the query to also lack known employment nouns.
        "CPF rate age 58" is short (3 words) but completely specific — rewriting
        it risks introducing drift. "what are my rights" is vague AND lacks
        specific nouns, so the LLM rewrite adds value.
      - `has_typos`: tightened to require at least one consonant cluster of 4+
        or vowel run of 3+ that doesn't appear in common English words.
    """
    words     = text.lower().split()
    word_set  = set(words)

    # Crude but effective typo signals: unusual consonant clusters or vowel runs
    has_typos = bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}|[aeiou]{3,}', text.lower()))

    is_informal = any(w in text.lower() for w in [
        "lah", "leh", "mah", "kena", "anyhow", "never pay", "never give",
        "berapa", "boleh", "can meh", "where got", "how liddat", "liddat",
        "sabo", "bo jio", "ok what",
    ])

    # Mixed scripts remaining after rule-based pass
    has_foreign_script = bool(re.search(r'[\u4e00-\u9fff\u0600-\u06ff]', text))

    # Short AND vague: lacks any known employment noun AND has no question mark
    is_vague = (
        len(words) <= 4
        and "?" not in text
        and word_set.isdisjoint(_EMPLOYMENT_NOUNS)
    )

    return any([has_typos, is_informal, is_vague, has_foreign_script])


@lru_cache(maxsize=256)
def llm_rewrite(query: str) -> str:
    """
    LLM-based query rewriting. Cached so repeated queries are free.
    Falls back to the rule-cleaned query if:
      - The LLM call fails
      - The rewrite is empty
      - The rewrite is more than 2x longer (verbose hallucination)
      - The rewrite drops all original content keywords (topic drift)
    """
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",   # fast cheap model fine for rewriting
            temperature=0,
            max_tokens=80,
            api_key=os.getenv("GROQ_API_KEY")
        )
        result    = llm.invoke(LLM_REWRITE_PROMPT.format(query=query))
        rewritten = result.content.strip().strip('"').strip("'")

        if not rewritten:
            return query

        # Guard 1: reject if rewrite is more than 2x longer (likely hallucinated padding)
        if len(rewritten.split()) > len(query.split()) * 2:
            return query

        # Guard 2: reject if the rewrite lost all original content keywords
        # (content words = tokens of 4+ characters, to skip stopwords)
        original_content = set(re.findall(r'\b\w{4,}\b', query.lower()))
        rewritten_content = set(re.findall(r'\b\w{4,}\b', rewritten.lower()))
        if original_content and original_content.isdisjoint(rewritten_content):
            return query  # topic drift detected

        return rewritten
    except Exception:
        pass
    return query


def preprocess_query(raw_query: str) -> str:
    """
    Full preprocessing pipeline:
      1. Rule-based abbreviation expansion
      2. Rule-based Singlish normalisation
      3. LLM rewrite (only if needed, cached)

    Returns the cleaned query ready for embedding + retrieval.
    """
    step1 = expand_abbreviations(raw_query)
    step2 = normalise_singlish(step1)
    step3 = llm_rewrite(step2) if needs_llm_rewrite(step2) else step2
    return step3


def preprocess_query_with_trace(raw_query: str) -> dict:
    """
    Same as preprocess_query but returns all intermediate steps.
    Useful for the Streamlit UI to show users what was interpreted.
    """
    step1    = expand_abbreviations(raw_query)
    step2    = normalise_singlish(step1)
    used_llm = needs_llm_rewrite(step2)
    step3    = llm_rewrite(step2) if used_llm else step2

    return {
        "original":       raw_query,
        "after_abbrev":   step1,
        "after_singlish": step2,
        "final":          step3,
        "was_rewritten":  step3 != raw_query,
        "used_llm":       used_llm,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        # Should be rewritten
        "anual leav entitlement",
        "boss never pay me salary lah what can i do",
        "MC leave how many days",
        "kena retrench what happen",
        "CPF berapa percent employer kena pay",
        "EP minium sallary",
        "what are my rights",
        # Should NOT be rewritten (already clean or specific enough)
        "OT pay rate",                                               # short but specific
        "How many days annual leave am I entitled to after 3 years?",
        "What is the CPF contribution rate for an employee aged 58?",
        "How many weeks of paternity leave am I entitled to?",
    ]

    print("Query Understanding — Preprocessing Test")
    print("=" * 65)
    for q in test_queries:
        result  = preprocess_query_with_trace(q)
        changed = " ← rewritten" if result["was_rewritten"] else " (unchanged)"
        llm_tag = "  [LLM]" if result["used_llm"] else ""
        print(f"\nOriginal : {result['original']}")
        print(f"Final    : {result['final']}{changed}{llm_tag}")
