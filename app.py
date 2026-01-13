import re
import time
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlencode, quote_plus, urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# Basics
# ----------------------------

APP_TITLE = "NI Job Matcher (JobApplyNI + Find a Job)"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQ_TIMEOUT = 25  # seconds
RETRIES = 2
BACKOFF = 1.2

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","if","in","into","is","it","its",
    "of","on","or","that","the","their","then","there","these","they","this","to","was","were","will","with",
    "you","your","we","our","i","me","my","he","she","his","her","them","us",
    "job","role","work","working","experience","skills","responsibilities","duties","required","requirements",
    "ability","team","support","knowledge","including","must","within","across","ensure","ensuring",
    "years","year","month","months","day","days",
}

def http_get(url: str, params: Optional[dict] = None) -> requests.Response:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
    }
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
            return r
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF ** attempt)
    raise last_err  # type: ignore


def normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def tokenise(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    parts = [p for p in text.split() if p and p not in STOPWORDS and len(p) > 2]
    # ultra-light stemming
    out = []
    for w in parts:
        for suf in ("ing","ers","er","ed","es","s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def split_into_chunks(cv_text: str) -> List[str]:
    """
    We deliberately *don't* require perfect headings.
    We split by blank lines and also by long lines that look like role headers.
    """
    cv_text = (cv_text or "").strip()
    if not cv_text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", cv_text) if b.strip()]
    # further split huge blocks
    chunks = []
    for b in blocks:
        if len(b) < 800:
            chunks.append(b)
        else:
            # split on sentences-ish to make match more "aspect based"
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", b)
            buf = []
            total = 0
            for p in parts:
                buf.append(p)
                total += len(p)
                if total >= 500:
                    chunks.append(" ".join(buf).strip())
                    buf, total = [], 0
            if buf:
                chunks.append(" ".join(buf).strip())
    return chunks[:30]  # hard cap


def binary_cosine(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    return inter / math.sqrt(len(a) * len(b))


def humanise_score(raw: float, title_bonus: float = 0.0) -> int:
    """
    Map raw similarity (0..1) into a human-looking 40..95-ish score.
    We keep it honest-ish (low raw -> still not "0", but won't hit 90+).
    """
    raw = max(0.0, min(1.0, raw))
    score = 35 + 60 * (raw ** 0.55)  # 35..95
    score += 7 * max(0.0, min(1.0, title_bonus))
    return int(max(35, min(98, round(score))))


def top_overlap_keywords(cv_tokens: set, job_tokens: set, k: int = 8) -> List[str]:
    overlaps = list(cv_tokens.intersection(job_tokens))
    overlaps.sort()
    return overlaps[:k]


@dataclass
class Job:
    source: str
    title: str
    company: str
    location: str
    date: str
    url: str
    snippet: str


# ----------------------------
# Source 1: JobApplyNI (NI Job Centre Online)
# ----------------------------

JOBAPPLY_BASE = "https://www.jobapplyni.com/"
JOBAPPLY_DETAIL_PREFIX = "https://www.jobapplyni.com/Vacancy/VacancyDetail"

def parse_jobapply_list(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")

    # job titles appear as H2 with a link (from the page structure)
    jobs: List[Job] = []

    # Heuristic: titles are in <h2> tags on results pages
    for h2 in soup.find_all(["h2", "h3"]):
        a = h2.find("a")
        if not a:
            continue
        title = normalise_space(a.get_text(" ", strip=True))
        href = a.get("href") or ""
        if not title or "Return to job search" in title:
            continue
        if "VacancyDetail" not in href:
            continue

        # The listing page presents a predictable set of fields after each header,
        # but HTML structure can vary; we walk forward a bit.
        container_text = []
        node = h2
        for _ in range(25):
            node = node.find_next()
            if not node:
                break
            if node.name in ("h2","h3"):
                break
            txt = normalise_space(node.get_text(" ", strip=True))
            if txt:
                container_text.append(txt)

        blob = " | ".join(container_text)
        # Extract simple fields via patterns we saw on the site
        def extract_after(label: str) -> str:
            m = re.search(rf"{re.escape(label)}\s*\|\s*([^|]+)", blob, re.IGNORECASE)
            return normalise_space(m.group(1)) if m else ""

        company = ""
        # Often scheme/employer is a line right after title; we take first meaningful.
        for t in container_text[:6]:
            if t.lower() not in ("find out more",) and not t.lower().startswith("vacancy id"):
                company = t
                break

        location = extract_after("Location") or ""
        area = extract_after("Area") or ""
        if area and (not location or area.lower() not in location.lower()):
            location = normalise_space(f"{location} ({area})") if location else area

        date = extract_after("Closing date") or ""
        snippet = ""  # list page doesn't include a true snippet; we fill later optionally

        url = urljoin(JOBAPPLY_BASE, href)

        jobs.append(
            Job(
                source="JobApplyNI",
                title=title,
                company=company or "Unknown",
                location=location or "Northern Ireland",
                date=date,
                url=url,
                snippet=snippet,
            )
        )

    # Dedupe by URL
    dedup = {}
    for j in jobs:
        dedup[j.url] = j
    return list(dedup.values())


def parse_jobapply_detail(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Job description sits under a heading "Job description"
    text_parts = []
    # Try grabbing main content around h3/h4
    for header in soup.find_all(["h3", "h4"]):
        if "job description" in header.get_text(" ", strip=True).lower():
            # collect next few paragraphs/lists
            node = header
            for _ in range(60):
                node = node.find_next()
                if not node:
                    break
                if node.name in ("h1","h2","h3") and node.get_text(strip=True):
                    break
                if node.name in ("p","li"):
                    t = normalise_space(node.get_text(" ", strip=True))
                    if t:
                        text_parts.append(t)
            break

    # Fallback: use all list items + paragraphs on page
    if not text_parts:
        for node in soup.find_all(["p","li"]):
            t = normalise_space(node.get_text(" ", strip=True))
            if t:
                text_parts.append(t)

    return normalise_space(" ".join(text_parts))[:2000]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_jobapplyni(max_pages: int = 3, keyword: str = "") -> Tuple[List[Job], List[dict]]:
    jobs: List[Job] = []
    diagnostics = []
    for page in range(1, max_pages + 1):
        params = {
            "DoSearch": "true",
            "CurrentPage": str(page),
        }
        if keyword.strip():
            params["keyword"] = keyword.strip()

        url = JOBAPPLY_BASE
        try:
            r = http_get(url, params=params)
            status = r.status_code
            if status != 200:
                diagnostics.append({"url": r.url, "status": status, "entries": 0, "error": f"HTTP {status}"})
                continue
            page_jobs = parse_jobapply_list(r.text)
            diagnostics.append({"url": r.url, "status": status, "entries": len(page_jobs), "error": ""})
            jobs.extend(page_jobs)
        except Exception as e:
            diagnostics.append({"url": f"{url}?{urlencode(params)}", "status": None, "entries": 0, "error": repr(e)})

    # Dedup
    dedup = {j.url: j for j in jobs}
    return list(dedup.values()), diagnostics


@st.cache_data(show_spinner=False, ttl=60 * 30)
def enrich_jobapplyni_details(jobs: List[Job], max_details: int = 25) -> Tuple[List[Job], List[dict]]:
    """
    Fetch details for top N JobApplyNI jobs to get real text for matching.
    """
    diagnostics = []
    out = []
    for idx, j in enumerate(jobs):
        if idx >= max_details:
            out.append(j)
            continue
        try:
            r = http_get(j.url)
            status = r.status_code
            if status != 200:
                diagnostics.append({"url": j.url, "status": status, "error": f"HTTP {status}"})
                out.append(j)
                continue
            desc = parse_jobapply_detail(r.text)
            diagnostics.append({"url": j.url, "status": status, "error": ""})
            out.append(Job(**{**j.__dict__, "snippet": desc}))
        except Exception as e:
            diagnostics.append({"url": j.url, "status": None, "error": repr(e)})
            out.append(j)
    return out, diagnostics


# ----------------------------
# Source 2: DWP "Find a job" (filtered to Northern Ireland)
# ----------------------------

DWP_BASE = "https://findajob.dwp.gov.uk"
DWP_SEARCH = "https://findajob.dwp.gov.uk/search"

# Northern Ireland location id observed on the site
DWP_LOC_NI = "86423"

def parse_dwp_list(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Job] = []

    # On Find a job, each result is typically an h3 with a link
    for h3 in soup.find_all("h3"):
        a = h3.find("a")
        if not a:
            continue
        title = normalise_space(a.get_text(" ", strip=True))
        href = a.get("href") or ""
        if not title or "/details/" not in href:
            continue

        # grab the result block: next siblings contain the meta bullets + snippet
        meta = []
        snippet = ""
        node = h3
        for _ in range(40):
            node = node.find_next()
            if not node:
                break
            if node.name == "h3":
                break
            txt = normalise_space(node.get_text(" ", strip=True))
            if not txt:
                continue
            # stop on "Save ... job"
            if txt.lower().startswith("save ") and txt.lower().endswith(" job to favourites"):
                continue
            # meta bullet lines often have '*' in rendered view but in HTML they're list items
            meta.append(txt)
            if len(meta) >= 12:
                break

        blob = " | ".join(meta)

        # very lightweight extraction:
        # expected order: date, employer-location, salary(optional), contract, hours, snippet...
        date = ""
        company = ""
        location = ""
        contract = ""

        # Try to infer fields from first few meta items
        # A common pattern is: [date] [employer - location] [salary?] [contract] [hours]
        if meta:
            date = meta[0]
        if len(meta) >= 2:
            company_loc = meta[1]
            if " - " in company_loc:
                company, location = [normalise_space(x) for x in company_loc.split(" - ", 1)]
            else:
                company = company_loc
        if len(meta) >= 4:
            contract = meta[2] if "£" not in meta[2] else meta[3]

        # snippet: take the longest meta item beyond first few
        candidates = meta[3:]
        if candidates:
            snippet = max(candidates, key=len)
        snippet = snippet[:2000]

        url = urljoin(DWP_BASE, href)

        jobs.append(
            Job(
                source="Find a job (DWP)",
                title=title,
                company=company or "Unknown",
                location=location or "Northern Ireland",
                date=date,
                url=url,
                snippet=snippet,
            )
        )

    dedup = {j.url: j for j in jobs}
    return list(dedup.values())


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_dwp(max_pages: int = 2, q: str = "") -> Tuple[List[Job], List[dict]]:
    jobs: List[Job] = []
    diagnostics = []
    for page in range(1, max_pages + 1):
        params = {"loc": DWP_LOC_NI}
        if q.strip():
            params["q"] = q.strip()
        if page > 1:
            params["p"] = str(page)

        try:
            r = http_get(DWP_SEARCH, params=params)
            status = r.status_code
            if status != 200:
                diagnostics.append({"url": r.url, "status": status, "entries": 0, "error": f"HTTP {status}"})
                continue
            page_jobs = parse_dwp_list(r.text)
            diagnostics.append({"url": r.url, "status": status, "entries": len(page_jobs), "error": ""})
            jobs.extend(page_jobs)
        except Exception as e:
            diagnostics.append({"url": f"{DWP_SEARCH}?{urlencode(params)}", "status": None, "entries": 0, "error": repr(e)})

    dedup = {j.url: j for j in jobs}
    return list(dedup.values()), diagnostics


# ----------------------------
# Matching
# ----------------------------

def match_jobs(cv_text: str, jobs: List[Job]) -> List[dict]:
    cv_text = cv_text or ""
    chunks = split_into_chunks(cv_text)
    cv_tokens_full = set(tokenise(cv_text))
    chunk_tokens = [set(tokenise(c)) for c in chunks if c.strip()]

    results = []
    for j in jobs:
        job_text = " ".join([j.title, j.company, j.location, j.snippet or ""])
        job_tokens = set(tokenise(job_text))

        full_raw = binary_cosine(cv_tokens_full, job_tokens)

        best_chunk_raw = 0.0
        if chunk_tokens:
            best_chunk_raw = max(binary_cosine(ct, job_tokens) for ct in chunk_tokens)

        # title bonus (helps “Housekeeping Supervisor” find supervisor roles, etc.)
        title_tokens = set(tokenise(j.title))
        title_bonus = binary_cosine(cv_tokens_full, title_tokens)

        raw = max(full_raw, best_chunk_raw)
        # slightly weight the title in as well
        raw = min(1.0, (0.82 * raw) + (0.18 * title_bonus))

        score = humanise_score(raw, title_bonus=title_bonus)
        why = ", ".join(top_overlap_keywords(cv_tokens_full, job_tokens, k=8))

        results.append(
            {
                "Score": score,
                "Title": j.title,
                "Company": j.company,
                "Location": j.location,
                "Date": j.date,
                "Source": j.source,
                "Why": why,
                "URL": j.url,
            }
        )

    results.sort(key=lambda x: (x["Score"], x["Date"]), reverse=True)
    return results


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.caption(
    "Key-free sources. Designed to actually return NI jobs on Streamlit Cloud. "
    "Matches your CV both as a whole and by chunks (roles/paragraphs), so mixed experience doesn't zero you out."
)

with st.sidebar:
    st.subheader("Search")
    keyword_hint = st.text_input("Optional keyword filter (leave blank for broad search)", value="")
    breadth = st.selectbox("Search breadth", ["Fast (≈60 jobs)", "Balanced (≈120 jobs)", "Wide (≈200+ jobs)"], index=1)
    min_score = st.slider("Minimum match score", min_value=35, max_value=95, value=45, step=1)
    show_diagnostics = st.checkbox("Show diagnostics", value=False)

    if breadth.startswith("Fast"):
        jp_pages, dwp_pages, jp_details = 2, 1, 18
    elif breadth.startswith("Balanced"):
        jp_pages, dwp_pages, jp_details = 3, 2, 25
    else:
        jp_pages, dwp_pages, jp_details = 5, 3, 35

st.subheader("1) Add your CV")
tab1, tab2 = st.tabs(["Paste CV text", "Upload a .txt file"])

cv_text = ""
with tab1:
    cv_text = st.text_area(
        "Paste your CV here",
        height=260,
        placeholder="Paste CV text… (PDF/DOCX not supported in this ultra-simple build — convert to text first.)",
    )

with tab2:
    up = st.file_uploader("Upload a plain text CV (.txt)", type=["txt"])
    if up is not None:
        cv_text = up.read().decode("utf-8", errors="ignore")

st.subheader("2) Run search")
run = st.button("Find matching NI jobs", type="primary", use_container_width=True)

if run:
    if not cv_text.strip():
        st.error("Please paste or upload CV text first.")
        st.stop()

    with st.spinner("Fetching jobs…"):
        jp_jobs, jp_diag = fetch_jobapplyni(max_pages=jp_pages, keyword=keyword_hint)
        dwp_jobs, dwp_diag = fetch_dwp(max_pages=dwp_pages, q=keyword_hint)

        # enrich JobApplyNI with real descriptions (improves matching a LOT)
        jp_jobs_enriched, jp_detail_diag = enrich_jobapplyni_details(jp_jobs, max_details=jp_details)

        all_jobs = jp_jobs_enriched + dwp_jobs

        # Dedup across sources by URL (and by title+company if needed)
        dedup = {}
        for j in all_jobs:
            key = j.url.strip().lower()
            if key:
                dedup[key] = j
            else:
                key2 = f"{j.title}|{j.company}".strip().lower()
                dedup[key2] = j
        all_jobs = list(dedup.values())

    with st.spinner("Matching CV to jobs…"):
        matches = match_jobs(cv_text, all_jobs)
        matches = [m for m in matches if m["Score"] >= min_score]

    st.subheader(f"Results ({len(matches)})")
    if not matches:
        st.warning(
            "No matches above your minimum score. Try lowering the minimum score slider, "
            "or leave keyword filter blank for a broader pull."
        )
    else:
        for m in matches[:60]:
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"### {m['Title']}")
                st.write(f"**{m['Company']}** — {m['Location']}  \n"
                         f"*{m['Source']}* • {m['Date']}")
                if m["Why"]:
                    st.caption(f"Overlap keywords: {m['Why']}")
                st.link_button("Open job", m["URL"])
            with right:
                st.metric("Match", f"{m['Score']}%")
            st.divider()

    if show_diagnostics:
        st.subheader("Diagnostics")
        st.write("JobApplyNI list fetch:")
        st.json(jp_diag)
        st.write("JobApplyNI detail fetch:")
        st.json(jp_detail_diag)
        st.write("Find a job (DWP) fetch:")
        st.json(dwp_diag)
        st.write({"Total fetched (deduped)": len(all_jobs), "Displayed": len(matches)})

st.caption(
    "Sources: JobApplyNI (NI Job Centre Online) + Find a Job filtered to Northern Ireland."
)
