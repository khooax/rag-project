"""
ingest.py — Scrape, chunk, embed and store official SG employment docs.

Data sources:
  1. PDFs in data/ folder (e.g. Employment Act PDF)
  2. MOM website — recursive BFS crawler that follows links 
     until it reaches actual article pages with prose content.
     No manual URL listing needed.

Run once: python ingest.py
"""

import os
import re
import time
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ── Config ──────────────────────────────────────────────────────────────────────
DB_DIR        = "./db"
DATA_DIR      = "./data"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100  # increased from 50 to reduce boundary splits
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
REQUEST_DELAY = 1.0   # seconds between requests

# ── Crawler seeds ───────────────────────────────────────────────────────────────
# BFS starts here and recursively follows all in-scope links.
CRAWL_SEEDS = [
    "https://www.mom.gov.sg/employment-practices",
    "https://www.mom.gov.sg/workplace-safety-and-health",
    "https://www.mom.gov.sg/passes-and-permits",
]

# Only follow URLs that start with one of these prefixes.
# This prevents the crawler from leaving the relevant sections.
ALLOWED_PREFIXES = [
    "https://www.mom.gov.sg/employment-practices",
    "https://www.mom.gov.sg/workplace-safety-and-health",
    "https://www.mom.gov.sg/passes-and-permits",
]

MAX_PAGES_TOTAL = 200   # safety cap across all seeds
MIN_TEXT_LENGTH = 300   # minimum chars to count as real content (not just a nav grid)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ── MOM page type detection ─────────────────────────────────────────────────────

def is_landing_page(soup: BeautifulSoup) -> bool:
    """
    MOM landing/section pages have data_template="section-landing-template"
    in their meta tags. Article pages have a different template or none.
    We use this to decide whether to follow links vs extract content.
    """
    meta = soup.find("meta", {"name": "data_template"})
    if meta and "section-landing" in meta.get("content", ""):
        return True
    # Also check: if the main content is ONLY a media-grid with no <p> tags outside it
    main = soup.find(id="MainContent")
    if main:
        # Count paragraphs outside the media-grid
        grid = main.find(class_="media-grid")
        if grid:
            grid.decompose()
        remaining_text = main.get_text(strip=True)
        if len(remaining_text) < MIN_TEXT_LENGTH:
            return True
    return False


# ── Link extraction ─────────────────────────────────────────────────────────────

def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Extract all in-scope links from a page.
    Works for both landing pages (media-grid links) and article pages
    (inline links within content).
    """
    base = "https://www.mom.gov.sg"
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip anchors, javascript, external, mailto
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        # Build absolute URL
        if href.startswith("/"):
            full_url = base + href
        elif href.startswith("http"):
            full_url = href
        else:
            full_url = urljoin(base_url, href)
        # Strip query strings and fragments
        full_url = full_url.split("?")[0].split("#")[0]
        # Only keep in-scope URLs
        if any(full_url.startswith(p) for p in ALLOWED_PREFIXES):
            links.add(full_url)
    return list(links)


# ── Content extraction ──────────────────────────────────────────────────────────

def extract_text(soup: BeautifulSoup) -> str:
    """
    Extract clean article text from a MOM page.
    Targets #MainContent and strips all nav/footer/share boilerplate.
    """
    # Remove boilerplate elements
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    for selector in [
        ".menu-holder", ".topbar", "header", "nav", "footer",
        ".share-widget", ".ui-breadcrumbs", ".breadcrumbs-mobile",
        ".mom-last-updated", ".dxd-mom-footer", ".module-footer",
        ".module-share", "#navsearch_0_DivCode", ".skip-navigation",
        ".aspNetHidden", "#dialog", ".menu-wrapper",
    ]:
        for el in soup.select(selector):
            el.decompose()

    # Get the main content area
    main = soup.find(id="MainContent")
    if not main:
        return ""

    # Remove the media-grid (those are just nav links, not content)
    for grid in main.find_all(class_="media-grid"):
        grid.decompose()

    text = main.get_text(separator="\n", strip=True)
    # Clean up blank lines
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 2]
    return "\n".join(lines)


# ── Fetcher ─────────────────────────────────────────────────────────────────────

def fetch(url: str) -> BeautifulSoup | None:
    """Fetch URL and return BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", ""):
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return None


# ── Recursive BFS crawler ───────────────────────────────────────────────────────

def crawl_all(seeds: list[str], max_pages: int = MAX_PAGES_TOTAL) -> list[Document]:
    """
    Recursive BFS starting from seed URLs.

    For each page visited:
    - If it's a LANDING page (section-landing-template or media-grid only):
        → extract all links and add them to the queue (don't save content)
    - If it's an ARTICLE page (has real prose content):
        → extract and save the content as a Document

    This correctly handles MOM's 3-level hierarchy:
        /employment-practices                        (landing)
          /employment-practices/employment-act       (landing)
            /employment-practices/employment-act/who-is-covered  (article ✓)
    """
    visited  = set()
    queue    = deque(seeds)
    docs     = []
    landing_count  = 0
    article_count  = 0

    print(f"\n  Starting BFS from {len(seeds)} seed(s), max {max_pages} pages")
    print(f"  (Automatically handles 3-level MOM site hierarchy)")

    pbar = tqdm(total=max_pages, unit="page", desc="  Crawling")

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        pbar.update(1)
        pbar.set_postfix({"articles": article_count, "queued": len(queue)})

        soup = fetch(url)
        if not soup:
            time.sleep(REQUEST_DELAY)
            continue

        # Classify the page
        if is_landing_page(soup):
            # Landing page: follow its links, don't save content
            landing_count += 1
            new_links = extract_links(soup, url)
            for link in new_links:
                if link not in visited:
                    queue.append(link)
        else:
            # Article page: extract content
            text = extract_text(soup)
            if len(text) >= MIN_TEXT_LENGTH:
                title_tag = soup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else url
                docs.append(Document(
                    page_content=text,
                    metadata={"source": title, "url": url}
                ))
                article_count += 1
                # Also follow any in-content links (catches sub-articles)
                new_links = extract_links(soup, url)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)

        time.sleep(REQUEST_DELAY)

    pbar.close()
    print(f"\n  BFS complete:")
    print(f"    Pages visited   : {len(visited)}")
    print(f"    Landing pages   : {landing_count} (followed links, skipped content)")
    print(f"    Article pages   : {article_count} (content saved)")
    print(f"    Remaining queue : {len(queue)} (hit page cap)")

    return docs


# ── PDF loader ──────────────────────────────────────────────────────────────────

def load_pdfs(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Load all PDFs from data/.
    Drop any PDF (Employment Act, Tripartite Guidelines, etc.)
    into the data/ folder — all are auto-detected.
    """
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"  No PDFs found in {data_dir}/")
        print(f"  Tip: copy your Employment Act PDF into data/ and re-run.")
        return []

    docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name} ...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_path.stem.replace("_", " ").replace("-", " ")
                page.metadata["file"]   = pdf_path.name
            docs.extend(pages)
            print(f"    -> {len(pages)} pages loaded")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    return docs


# ── Fallback curated content ────────────────────────────────────────────────────

def load_fallback_docs() -> list[Document]:
    """
    Always-included hardcoded key provisions.
    Reliable baseline even when web scraping partially fails.
    """
    text = """
SOURCE: Singapore Employment Act — Core Provisions (Curated Fallback)

NOTICE PERIOD (Section 10)
- Employed < 26 weeks: 1 day notice
- Employed 26 weeks to < 2 years: 1 week notice
- Employed 2 to < 5 years: 2 weeks notice
- Employed >= 5 years: 4 weeks notice
- Notice can be waived by paying salary in lieu of notice.
- Both employer and employee must give the same notice period.

SALARY PAYMENT (Section 21)
- Must be paid within 7 days after end of salary period.
- Overtime pay: within 14 days after end of salary period.
- Unauthorised salary deductions are illegal.
- Allowable deductions: CPF contributions, absence from work, damage/loss
  caused by employee, accommodation provided, amenities provided.

ANNUAL LEAVE (Section 88A)
- Requires at least 3 months service to qualify.
- Year 1: 7 days; Year 2: 8 days; increases by 1 day/year up to 14 days from Year 8.
- Unused leave may be encashed or carried forward per contract terms.

SICK LEAVE (Section 89)
- Outpatient sick leave: 14 days/year.
- Hospitalisation leave: 60 days/year (includes the 14 outpatient days).
- Requires a valid medical certificate. Must inform employer within 48 hours of absence.

OVERTIME (Part IV — employees earning up to $2,600/month basic salary)
- Rate: 1.5x the hourly basic rate of pay.
- Maximum ordinary hours: 44/week.
- Maximum overtime: 72 hours/month.

REST DAYS (Section 36)
- 1 rest day per week (continuous 24 hours) for Part IV employees.
- Working on rest day at employer's request: 2x basic rate.

PUBLIC HOLIDAYS (Section 88)
- 11 gazetted public holidays per year in Singapore.
- If required to work on a public holiday: entitled to extra day's pay or off-in-lieu.
- If public holiday falls on rest day: next working day is a paid holiday.

MATERNITY LEAVE (Child Development Co-Savings Act)
- 16 weeks government-paid for Singapore Citizen child.
- 8 weeks for non-Citizen child.
- Employee must have worked for employer at least 3 months before delivery.

PATERNITY LEAVE
- 2 weeks government-paid for Singapore Citizen child.
- Employee must have worked at least 3 months.

CHILDCARE LEAVE
- 6 days/year if child is a Singapore Citizen (until child turns 7).
- 2 days/year if child is not a Singapore Citizen.
- Employee must have worked at least 3 months.

RETRENCHMENT
- No statutory retrenchment benefit unless specified in contract or collective agreement.
- Tripartite Advisory norm: 2 weeks pay per year of service for employees with >= 2 years service.
- Employers must notify MOM if retrenching >= 5 employees within any 6-month period.

WRONGFUL DISMISSAL
- Employee can claim if dismissed without just cause or excuse.
- Must file within 1 month from last day of employment.
- File at Employment Claims Tribunals (ECT) or via TADM.

PAYSLIP REQUIREMENT
- Itemised payslips must be issued to all employees covered by the Employment Act.
- Must be given within 3 working days of paying salary. Can be digital or paper.

KEY EMPLOYMENT TERMS (KET)
- Employer must provide written KET within 14 days of employment start.
- Must include: salary, working hours, leave entitlements, notice period, job scope.

CPF CONTRIBUTIONS
- Age <= 55:  Employee 20%, Employer 17% = 37% total
- Age 55-60:  Employee 15%, Employer 15% = 30% total
- Age 60-65:  Employee 10%, Employer 11.5% = 21.5% total
- Age > 65:   Employee 7.5%, Employer 9% = 16.5% total
- Applies to Singapore Citizens and Permanent Residents only.
- CPF Ordinary Wage ceiling: $6,800/month (from 2024).
- Foreign employees (EP/S Pass/Work Permit) do NOT contribute to CPF.

FAIR EMPLOYMENT (TAFEP Tripartite Guidelines)
- Must not discriminate based on: age, race, gender, religion,
  marital/family status, disability, or pregnancy.
- Recruitment must be merit-based.
- Job advertisements must not specify discriminatory criteria.
- Fair Consideration Framework (FCF): advertise on MyCareersFuture
  for at least 28 days before submitting EP application.

WORK PASSES
- Employment Pass (EP): min. $5,000/month fixed salary ($5,500 for financial services).
- S Pass: min. $3,150/month. Subject to sector quota (10-15% of workforce).
- Work Permit: semi-skilled workers in construction, marine, process,
  services, or manufacturing sectors.

WORK INJURY COMPENSATION (WICA)
- Covers all employees doing manual work regardless of salary.
- Also covers non-manual employees earning up to $2,600/month.
- Employer must report accidents causing > 3 days MC or hospitalisation to MOM.
- Compensation types: medical expenses, temporary/permanent incapacity, death.
- Employers must have WICA insurance for all covered employees.

CONTACTS
- MOM Helpline: 6438 5122 (Mon-Fri 8:30am-5:30pm)
- TADM (disputes): 1800 221 9088
- TAFEP (fair employment): 6838 0969
- Employment Claims Tribunals: www.employmentclaims.gov.sg
- WorkRight: www.workright.sg
- CPF Board: www.cpf.gov.sg
- MOM: www.mom.gov.sg
"""
    return [Document(
        page_content=text,
        metadata={
            "source": "Singapore Employment Act — Core Provisions (Curated)",
            "url": "https://sso.agc.gov.sg/Act/EA1968"
        }
    )]


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SG Employment Chatbot — Data Ingestion")
    print("=" * 60)

    Path(DATA_DIR).mkdir(exist_ok=True)
    all_docs = []

    # 1. Curated fallback (always included)
    print("\n[1/4] Loading curated fallback provisions...")
    all_docs.extend(load_fallback_docs())
    print(f"  Total so far: {len(all_docs)}")

    # 2. PDFs from data/
    print(f"\n[2/4] Loading PDFs from {DATA_DIR}/...")
    all_docs.extend(load_pdfs(DATA_DIR))
    print(f"  Total so far: {len(all_docs)}")

    # 3. Web crawl
    print(f"\n[3/4] Crawling MOM website (recursive BFS)...")
    web_docs = crawl_all(CRAWL_SEEDS)
    all_docs.extend(web_docs)

    print(f"\n  Total documents collected: {len(all_docs)}")

    # 4. Chunk + embed + store
    print(f"\n[4/4] Chunking, embedding, and storing...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  Created {len(chunks)} chunks")

    print(f"  Loading embedding model (first run ~90MB download)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"  Cleared existing DB")

    print(f"  Storing in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"\n{'=' * 60}")
    print(f"✅  Ingestion complete!")
    print(f"    Documents : {len(all_docs)}")
    print(f"    Chunks    : {len(chunks)}")
    print(f"    Stored at : {DB_DIR}/")
    print(f"{'=' * 60}")
    print(f"\nNext: streamlit run app.py\n")


if __name__ == "__main__":
    main()
