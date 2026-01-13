# app.py  (single-file Streamlit app)
# NI Job Matcher — TITLE-GATED tracks: Manager / Sales / IT
# - Jobs are INCLUDED or EXCLUDED based on JOB TITLE ONLY (your key requirement)
# - Description/snippet is used only for scoring AFTER title passes the gate
# - Sales can be manager-only (default) or include all sales roles (toggle)
#
# Paste this entire file into GitHub as app.py

import re
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlencode, urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# App config
# ----------------------------

APP_TITLE = "NI Job Matcher (Manager / Sales / IT)"
BUILD = "2026-01-13 v9 (TITLE-only gating + Sales toggle)"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQ_TIMEOUT = 25
RETRIES = 2
BACKOFF = 1.2

TRACKS = ["Manager", "Sales", "IT"]

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","if","in","into","is","it","its",
    "of","on","or","that","the","their","then","there","these","they","this","to","was","were","will","with",
    "you","your","we","our","i","me","my","he","she","his","her","them","us",
    "job","role","work","working","experience","skills","responsibilities","duties","required","requirements",
    "ability","team","support","knowledge","including","must","within","across","ensure","ensuring",
    "years","year","month","months","day","days",
}

# ---------- CV skill buckets (for scoring) ----------
# These are NOT used for inclusion/exclusion (title-only gating handles that).
OPS_KWS = {
    "supervisor","housekeeping","clean","cleaning","hygiene","hotel","hospitality","rota","schedule","scheduling",
    "shift","team","training","inventory","stock","order","ordering","audit","quality","compliance","facilities",
    "operations","operational","warehouse","logistics","stores","receiving","dispatch","customer","service","retail",
    "assistant","coordinator","administrator","admin","process","improvement","performance","record","records"
}

SALES_KWS = {
    "sales","sell","selling","business","development","bdm","account","accounts","client","clients","customer",
    "customers","crm","pipeline","prospecting","lead","leads","closing","negotiate","negotiation","marketing",
    "commercial","revenue","target","targets","quota","portfolio","relationship","relationships","broker","property"
}

TECH_KWS = {
    "python","javascript","typescript","html","css","react","node","api","sql","database","data","analytics",
    "developer","development","software","engineer","engineering","devops","cloud","aws","azure","git","github",
    "vscode","automation","testing","qa","technical","systems","network","security","cyber"
}


# ----------------------------
# TITLE-ONLY gating (your requirement)
# ----------------------------

# IMPORTANT:
# - Avoid overly-generic single words that create nonsense matches (e.g. "support").
# - Prefer phrases like "IT Support" or "Service Desk".
IT_TITLE_PHRASES = {
    "it support", "service desk", "helpdesk", "help desk", "desktop support",
    "technical support", "1st line", "first line", "2nd line", "second line",
    "software developer", "web developer", "front end", "frontend", "back end", "backend",
    "data analyst", "business analyst", "systems analyst",
    "network engineer", "cloud engineer", "devops", "cyber security", "cybersecurity",
    "qa tester", "test analyst", "automation tester"
}
IT_TITLE_WORDS = {
    "developer","programmer","engineer","technician","sysadmin","administrator",
    "analyst","devops","cloud","security","cyber","network","database"
}

SALES_TITLE_PHRASES = {
    "sales executive", "sales representative", "sales advisor", "sales consultant",
    "account manager", "account executive",
    "business development", "relationship manager", "client manager",
    "estate agent", "lettings", "property manager", "broker"
}
SALES_TITLE_WORDS = {"sales","commercial","bdm","crm"}

MANAGER_TITLE_PHRASES = {
    "team leader", "shift leader", "duty manager", "assistant manager", "deputy manager",
    "operations manager", "general manager", "store manager", "branch manager"
}
MANAGER_TITLE_WORDS = {"manager","supervisor","lead","leader","head","director"}


def _title_has_phrase_or_word(title: str, phrases: set, words: set) -> bool:
    t = (title or "").lower().strip()
    if not t:
        return False

    # phrase check (substring)
    for p in phrases:
        if p in t:
            return True

    # word-boundary check (avoids "accountant" matching "account")
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True

    return False


def classify_track_from_title(title: str) -> Optional[str]:
    """
    STRICT inclusion:
    - We only classify by JOB TITLE.
    - If title doesn't look like Manager/Sales/IT, return None (job is skipped).
    """
    # IT first (more distinctive)
    if _title_has_phrase_or_word(title, IT_TITLE_PHRASES, IT_TITLE_WORDS):
        return "IT"

    # Sales second
    if _title_has_phrase_or_word(title, SALES_TITLE_PHRASES, SALES_TITLE_WORDS):
        return "Sales"

    # Manager last
    if _title_has_phrase_or_word(title, MANAGER_TITLE_PHRASES, MANAGER_TITLE_WORDS):
        return "Manager"

    return None


def looks_sales_manager_title_only(title: str) -> bool:
    """
    Sales tab default is manager-only.
    This MUST be title-based (per your requirement).
    """
    t = (title or "").lower()
    # keep this intentionally tight
    mgr_signals = {
        "manager", "team leader", "lead", "head", "director", "supervisor",
        "account manager", "sales manager", "business development manager", "bdm",
        "store manager", "branch manager"
    }
    return any(s in t for s in mgr_signals)


# ----------------------------
# HTTP + text helpers
# ----------------------------

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
    out = []
    for w in parts:
        for suf in ("ing","ers","er","ed","es","s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def binary_cosine(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    return inter / math.sqrt(len(a) * len(b))


def top_overlap_keywords(cv_tokens: set, job_tokens: set, k: int = 10) -> List[str]:
    overlaps = list(cv_tokens.intersection(job_tokens))
    overlaps.sort()
    return overlaps[:k]


def build_cv_profiles(cv_text: str) -> Dict[str, str]:
    """
    Build three CV views for scoring.
    Note: inclusion is title-only; this is only for better scoring.
    """
    lines = [ln.strip() for ln in (cv_text or "").splitlines() if ln.strip()]
    manager_lines, sales_lines, it_lines = [], [], []

    for ln in lines:
        toks = set(tokenise(ln))
        if toks & OPS_KWS:
            manager_lines.append(ln)
        if toks & SALES_KWS:
            sales_lines.append(ln)
        if toks & TECH_KWS:
            it_lines.append(ln)

    full = (cv_text or "").strip()
    profiles = {
        "Manager": "\n".join(manager_lines).strip() or full,
        "Sales": "\n".join(sales_lines).strip() or full,
        "IT": "\n".join(it_lines).strip() or full,
    }
    return profiles


# ----------------------------
# Data model
# ----------------------------

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
# Source 1: JobApplyNI
# ----------------------------

JOBAPPLY_BASE = "https://www.jobapplyni.com/"

def parse_jobapply_list(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Job] = []

    for h2 in soup.find_all(["h2", "h3"]):
        a = h2.find("a")
        if not a:
            continue
        title = normalise_space(a.get_text(" ", strip=True))
        href = a.get("href") or ""
        if not title or "VacancyDetail" not in href:
            continue

        # gather nearby text for company/location/date
        container_text = []
        node = h2
        for _ in range(25):
            node = node.find_next()
            if not node:
                break
            if node.name in ("h2", "h3"):
                break
            txt = normalise_space(node.get_text(" ", strip=True))
            if txt:
                container_text.append(txt)

        blob = " | ".join(container_text)

        def extract_after(label: str) -> str:
            m = re.search(rf"{re.escape(label)}\s*\|\s*([^|]+)", blob, re.IGNORECASE)
            return normalise_space(m.group(1)) if m else ""

        company = ""
        for t in container_text[:6]:
            low = t.lower()
            if low and low not in ("find out more",) and not low.startswith("vacancy id"):
                company = t
                break

        location = extract_after("Location") or ""
        area = extract_after("Area") or ""
        if area and (not location or area.lower() not in location.lower()):
            location = normalise_space(f"{location} ({area})") if location else area

        date = extract_after("Closing date") or ""
        url = urljoin(JOBAPPLY_BASE, href)

        jobs.append(
            Job(
                source="JobApplyNI",
                title=title,
                company=company or "Unknown",
                location=location or "Northern Ireland",
                date=date,
                url=url,
                snippet="",
            )
        )

    return list({j.url: j for j in jobs}.values())


def parse_jobapply_detail(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text_parts = []

    for header in soup.find_all(["h3", "h4"]):
        if "job description" in header.get_text(" ", strip=True).lower():
            node = header
            for _ in range(60):
                node = node.find_next()
                if not node:
                    break
                if node.name in ("h1", "h2", "h3") and node.get_text(strip=True):
                    break
                if node.name in ("p", "li"):
                    t = normalise_space(node.get_text(" ", strip=True))
                    if t:
                        text_parts.append(t)
            break

    if not text_parts:
        for node in soup.find_all(["p", "li"]):
            t = normalise_space(node.get_text(" ", strip=True))
            if t:
                text_parts.append(t)

    return normalise_space(" ".join(text_parts))[:2000]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_jobapplyni(max_pages: int = 3, keyword: str = "") -> Tuple[List[Job], List[dict]]:
    jobs: List[Job] = []
    diagnostics = []

    for page in range(1, max_pages + 1):
        params = {"DoSearch": "true", "CurrentPage": str(page)}
        if keyword.strip():
            params["keyword"] = keyword.strip()

        try:
            r = http_get(JOBAPPLY_BASE, params=params)
            status = r.status_code
            if status != 200:
                diagnostics.append({"url": r.url, "status": status, "entries": 0, "error": f"HTTP {status}"})
                continue
            page_jobs = parse_jobapply_list(r.text)
            diagnostics.append({"url": r.url, "status": status, "entries": len(page_jobs), "error": ""})
            jobs.extend(page_jobs)
        except Exception as e:
            diagnostics.append({"url": f"{JOBAPPLY_BASE}?{urlencode(params)}", "status": None, "entries": 0, "error": repr(e)})

    return list({j.url: j for j in jobs}.values()), diagnostics


@st.cache_data(show_spinner=False, ttl=60 * 30)
def enrich_jobapplyni_details(jobs: List[Job], max_details: int = 25) -> Tuple[List[Job], List[dict]]:
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
# Source 2: DWP Find a Job (NI filter)
# ----------------------------

DWP_BASE = "https://findajob.dwp.gov.uk"
DWP_SEARCH = "https://findajob.dwp.gov.uk/search"
DWP_LOC_NI = "86423"

def parse_dwp_list(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Job] = []

    for h3 in soup.find_all("h3"):
        a = h3.find("a")
        if not a:
            continue
        title = normalise_space(a.get_text(" ", strip=True))
        href = a.get("href") or ""
        if not title or "/details/" not in href:
            continue

        meta = []
        node = h3
        for _ in range(40):
            node = node.find_next()
            if not node:
                break
            if node.name == "h3":
                break
            txt = normalise_space(node.get_text(" ", strip=True))
            if txt:
                meta.append(txt)
            if len(meta) >= 12:
                break

        date = meta[0] if meta else ""
        company = ""
        location = ""
        if len(meta) >= 2:
            company_loc = meta[1]
            if " - " in company_loc:
                company, location = [normalise_space(x) for x in company_loc.split(" - ", 1)]
            else:
                company = company_loc

        snippet = ""
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

    return list({j.url: j for j in jobs}.values())


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

    return list({j.url: j for j in jobs}.values()), diagnostics


# ----------------------------
# Matching (TITLE-gated tracks)
# ----------------------------

def match_jobs_by_track(
    cv_text: str,
    jobs: List[Job],
    include_non_manager_sales: bool
) -> Dict[str, List[dict]]:

    cv_text = (cv_text or "").strip()
    profiles = build_cv_profiles(cv_text)

    prof_tokens = {k: set(tokenise(v)) for k, v in profiles.items()}
    full_tokens = set(tokenise(cv_text))

    buckets: Dict[str, List[dict]] = {k: [] for k in TRACKS}

    for j in jobs:
        # TITLE-ONLY gate (the big change)
        track = classify_track_from_title(j.title)
        if track is None:
            continue

        # Sales: manager-only unless toggled
        if track == "Sales" and (not include_non_manager_sales) and (not looks_sales_manager_title_only(j.title)):
            continue

        # scoring text can include snippet now (since it passed the title gate)
        job_text = " ".join([j.title, j.company, j.location, j.snippet or ""])
        job_tokens = set(tokenise(job_text))

        base_raw = binary_cosine(prof_tokens[track], job_tokens)

        title_tokens = set(tokenise(j.title))
        title_bonus = binary_cosine(full_tokens, title_tokens)

        raw = min(1.0, 0.85 * base_raw + 0.15 * title_bonus)

        buckets[track].append({
            "_raw": raw,
            "_title_bonus": title_bonus,
            "Title": j.title,
            "Company": j.company,
            "Location": j.location,
            "Date": j.date,
            "Source": j.source,
            "URL": j.url,
            "Why": ", ".join(top_overlap_keywords(prof_tokens[track], job_tokens, k=10)),
            "Track": track,
        })

    # Percentile-normalise within each track (so top results look like top results)
    for track, rows in buckets.items():
        n = len(rows)
        if n == 0:
            continue

        idx_sorted = sorted(range(n), key=lambda i: rows[i]["_raw"])
        ranks = [0] * n
        for r, idx in enumerate(idx_sorted):
            ranks[idx] = r

        for i, row in enumerate(rows):
            pct = ranks[i] / (n - 1) if n > 1 else 0.5
            score = 55 + 40 * pct                           # 55..95
            score += min(3.0, 10.0 * row["_title_bonus"])    # up to +3

            overlap_count = 0 if not row["Why"] else len([x for x in row["Why"].split(",") if x.strip()])
            score += min(2.0, 0.25 * overlap_count)          # up to +2

            row["Score"] = int(max(35, min(98, round(score))))
            del row["_raw"]
            del row["_title_bonus"]

        rows.sort(key=lambda x: (x["Score"], x["Date"]), reverse=True)

    return buckets


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(f"BUILD: {BUILD}")

st.write(
    "This version is **strict**: it only includes jobs whose **titles** look like **Manager**, **Sales**, or **IT**. "
    "So things like 'Diabetes Specialist Dietitian' will not appear unless the title matches your tracks."
)

with st.sidebar:
    st.subheader("Search")
    keyword_hint = st.text_input("Optional keyword filter (blank = broad search)", value="")
    breadth = st.selectbox("Search breadth", ["Fast", "Balanced", "Wide"], index=1)
    min_score = st.slider("Minimum match score", min_value=35, max_value=95, value=50, step=1)

    st.divider()
    st.subheader("Sales options")
    include_non_manager_sales = st.checkbox("Sales: include non-manager roles", value=False)

    st.divider()
    show_diagnostics = st.checkbox("Show diagnostics", value=False)

    if breadth == "Fast":
        jp_pages, dwp_pages, jp_details = 2, 1, 18
    elif breadth == "Balanced":
        jp_pages, dwp_pages, jp_details = 3, 2, 25
    else:
        jp_pages, dwp_pages, jp_details = 5, 3, 35


st.subheader("1) Paste CV text")
cv_text = st.text_area(
    "Paste CV text here",
    height=280,
    placeholder="Paste your CV text…",
)

st.subheader("2) Run search")
run = st.button("Find matching NI jobs", type="primary", use_container_width=True)

if run:
    if not cv_text.strip():
        st.error("Please paste your CV text first.")
        st.stop()

    with st.spinner("Fetching jobs…"):
        jp_jobs, jp_diag = fetch_jobapplyni(max_pages=jp_pages, keyword=keyword_hint)
        dwp_jobs, dwp_diag = fetch_dwp(max_pages=dwp_pages, q=keyword_hint)

        # Enrich JobApplyNI detail pages (better scoring, still title-gated for inclusion)
        jp_jobs_enriched, jp_detail_diag = enrich_jobapplyni_details(jp_jobs, max_details=jp_details)

        all_jobs = jp_jobs_enriched + dwp_jobs

        # Dedup across sources by URL
        dedup = {}
        for j in all_jobs:
            key = (j.url or "").strip().lower()
            if key:
                dedup[key] = j
            else:
                key2 = f"{j.title}|{j.company}".strip().lower()
                dedup[key2] = j
        all_jobs = list(dedup.values())

    with st.spinner("Matching CV to jobs (TITLE-gated tracks)…"):
        buckets = match_jobs_by_track(cv_text, all_jobs, include_non_manager_sales=include_non_manager_sales)
        for t in buckets:
            buckets[t] = [m for m in buckets[t] if m["Score"] >= min_score]

    tabs = st.tabs(TRACKS)

    for tab, track in zip(tabs, TRACKS):
        with tab:
            rows = buckets.get(track, [])
            st.subheader(f"{track} ({len(rows)})")

            if not rows:
                if track == "Sales" and (not include_non_manager_sales):
                    st.info("No sales-manager roles above your minimum score. Turn on ‘Sales: include non-manager roles’ to widen.")
                else:
                    st.info("No results above your minimum score. Try lowering it or removing the keyword filter.")
                continue

            for m in rows[:60]:
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {m['Title']}")
                    st.write(
                        f"**{m['Company']}** — {m['Location']}  \n"
                        f"*{m['Source']}* • {m['Date']}"
                    )
                    if m.get("Why"):
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
        st.write({"Total fetched (deduped)": len(all_jobs)})

st.caption("Sources: JobApplyNI + DWP Find a Job (Northern Ireland filter).")
