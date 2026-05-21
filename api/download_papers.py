"""
Hydroponic Research Paper Downloader
-------------------------------------
Fixes applied vs previous version:
  - Exponential backoff on HTTP 429 (rate limit)
  - CORE.ac.uk as a fallback source
  - Retries for transient network errors
  - Better PDF validation (checks header + size)
  - Saves a metadata JSON log for every topic
  - Polite delays between every request
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import os
import time
import random

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR             = "docs/knowledge_base"
MAX_PAPERS_PER_QUERY = 3       # How many results to fetch per query
DELAY_MIN            = 3.0     # Minimum seconds between requests
DELAY_MAX            = 6.0     # Maximum seconds (random jitter)
MAX_RETRIES          = 4       # Retries on 429 / network error
BACKOFF_BASE         = 8       # Seconds for first backoff; doubles each retry

# ── Topics & queries ──────────────────────────────────────────────────────────
TOPICS = {
    "crop_growth": [
        "hydroponic lettuce growth optimization",
        "plant growth stages hydroponics",
        "crop yield hydroponic system",
        "vertical farming plant production",
        "hydroponic tomato growth stages",
        "nutrient film technique crop growth",
        "hydroponic spinach basil growth",
        "plant phenology indoor farming",
    ],
    "iot_sensors": [
        "IoT sensor hydroponic monitoring",
        "smart greenhouse automation sensors",
        "pH EC sensor agricultural system",
        "wireless sensor network hydroponics",
        "real time plant monitoring IoT",
        "ESP32 Arduino hydroponic control",
        "automated nutrient monitoring system",
    ],
    "nutrients": [
        "hydroponic nutrient solution management",
        "nitrogen phosphorus potassium hydroponics",
        "mineral nutrition plant hydroponics",
        "nutrient deficiency hydroponic plants",
        "fertilizer management hydroponic lettuce",
        "calcium magnesium micronutrient hydroponics",
        "nutrient uptake plant root zone",
    ],
    "ph_management": [
        "pH management hydroponic system",
        "electrical conductivity pH plant growth",
        "pH control automated hydroponics",
        "acid base nutrient solution pH",
        "pH monitoring water quality agriculture",
        "optimal pH range hydroponic crops",
    ],
    "plant_health": [
        "plant disease detection hydroponics",
        "plant stress monitoring agriculture",
        "leaf disease classification deep learning",
        "plant health monitoring computer vision",
        "hydroponic plant pathology management",
        "root disease hydroponic system",
        "crop health precision agriculture",
    ],
}

HEADERS = {
    "User-Agent": (
        "HydroponicResearchBot/1.0 "
        "(Academic paper collection; contact: researcher@example.com)"
    ),
    "Accept": "application/json",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def polite_sleep():
    """Random delay to be polite to servers."""
    t = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(t)


def safe_get(url, retries=MAX_RETRIES, accept_json=True):
    """GET with exponential backoff on 429."""
    headers = dict(HEADERS)
    if not accept_json:
        headers["Accept"] = "*/*"

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 3)
                print(f"    Rate limited (429). Waiting {wait:.0f}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            elif e.code in (403, 404):
                print(f"    HTTP {e.code} — skipping this URL.")
                return None
            else:
                print(f"    HTTP error {e.code}: {e.reason}")
                return None
        except Exception as e:
            if attempt < retries:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"    Network error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Failed after {retries} retries: {e}")
                return None
    return None


def safe_filename(title, year=""):
    """Create a safe, readable filename from a paper title."""
    clean = "".join(c if (c.isalnum() or c in " _-") else "" for c in title)
    clean = clean.strip().replace(" ", "_")[:70]
    return f"{clean}_{year}.pdf" if year else f"{clean}.pdf"


def is_valid_pdf(content):
    """Check if bytes look like a real PDF and are a reasonable size."""
    return (
        isinstance(content, bytes)
        and content[:4] == b"%PDF"
        and len(content) > 10_000   # at least ~10 KB
    )


# ── Source 1: Semantic Scholar ────────────────────────────────────────────────

def search_semantic_scholar(query, limit=3):
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded}&limit={limit}"
        f"&fields=title,year,openAccessPdf,authors,externalIds"
    )
    raw = safe_get(url)
    if raw is None:
        return []
    try:
        return json.loads(raw).get("data", [])
    except Exception:
        return []


# ── Source 2: CORE API (no key needed for basic search) ───────────────────────

def search_core(query, limit=3):
    """
    CORE (core.ac.uk) has a free tier that doesn't require an API key
    for basic full-text search.
    """
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.core.ac.uk/v3/search/works"
        f"?q={encoded}&limit={limit}&fulltext=true"
    )
    raw = safe_get(url)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
        results = []
        for item in data.get("results", []):
            pdf_url = item.get("downloadUrl") or item.get("sourceFulltextUrls", [None])[0]
            if pdf_url:
                results.append({
                    "title": item.get("title", "untitled"),
                    "year":  str(item.get("yearPublished") or ""),
                    "pdf_url": pdf_url,
                })
        return results
    except Exception:
        return []


# ── Source 3: Unpaywall (DOI-based, very reliable) ────────────────────────────

def get_unpaywall_pdf(doi):
    """Given a DOI, ask Unpaywall for an open-access PDF link."""
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email=researcher@example.com"
    raw = safe_get(url)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url")
    except Exception:
        return None


# ── Download one PDF ──────────────────────────────────────────────────────────

def download_pdf(pdf_url, save_path):
    content = safe_get(pdf_url, accept_json=False)
    if content is None:
        return False
    if not is_valid_pdf(content):
        print(f"    Not a valid PDF (magic bytes wrong or file too small)")
        return False
    with open(save_path, "wb") as f:
        f.write(content)
    return True


# ── Main downloader ───────────────────────────────────────────────────────────

def download_all():
    total_new      = 0
    total_skipped  = 0
    total_failed   = 0
    metadata_log   = {}  # topic -> list of paper metadata

    for topic, queries in TOPICS.items():
        folder = os.path.join(BASE_DIR, topic)
        os.makedirs(folder, exist_ok=True)
        existing = set(os.listdir(folder))

        metadata_log[topic] = []

        print(f"\n{'='*62}")
        print(f"  TOPIC: {topic.upper()}  (existing files: {len(existing)})")
        print(f"{'='*62}")

        for query in queries:
            print(f"\n  Query: \"{query}\"")

            # ── Try Semantic Scholar first ────────────────────────────────
            papers_ss = search_semantic_scholar(query, limit=MAX_PAPERS_PER_QUERY)
            polite_sleep()

            for paper in papers_ss:
                title    = (paper.get("title") or "untitled").strip()
                year     = str(paper.get("year") or "")
                pdf_info = paper.get("openAccessPdf") or {}
                pdf_url  = pdf_info.get("url", "")
                doi      = (paper.get("externalIds") or {}).get("DOI", "")

                # If no direct PDF, try Unpaywall with the DOI
                if not pdf_url and doi:
                    print(f"    Trying Unpaywall for DOI: {doi[:40]}")
                    pdf_url = get_unpaywall_pdf(doi) or ""
                    polite_sleep()

                filename = safe_filename(title, year)

                if filename in existing:
                    print(f"    Already have : {filename[:55]}")
                    total_skipped += 1
                    metadata_log[topic].append({"title": title, "year": year, "status": "skipped"})
                    continue

                if not pdf_url:
                    print(f"    No free PDF  : {title[:55]}")
                    total_failed += 1
                    metadata_log[topic].append({"title": title, "year": year, "status": "no_pdf"})
                    continue

                print(f"    Downloading  : {title[:55]}")
                ok = download_pdf(pdf_url, os.path.join(folder, filename))
                polite_sleep()

                if ok:
                    existing.add(filename)
                    total_new += 1
                    print(f"    ✓ Saved!")
                    metadata_log[topic].append({
                        "title": title, "year": year,
                        "filename": filename, "status": "downloaded",
                        "source": "semantic_scholar"
                    })
                else:
                    total_failed += 1
                    metadata_log[topic].append({"title": title, "year": year, "status": "download_failed"})

            # ── Fallback: CORE API ────────────────────────────────────────
            if len([p for p in metadata_log[topic] if p["status"] == "downloaded"]) == 0:
                print(f"    → Trying CORE fallback...")
                core_papers = search_core(query, limit=MAX_PAPERS_PER_QUERY)
                polite_sleep()

                for paper in core_papers:
                    title   = (paper.get("title") or "untitled").strip()
                    year    = paper.get("year", "")
                    pdf_url = paper.get("pdf_url", "")
                    filename = safe_filename(title, year)

                    if filename in existing:
                        print(f"    Already have : {filename[:55]}")
                        total_skipped += 1
                        continue

                    if not pdf_url:
                        continue

                    print(f"    [CORE] Downloading: {title[:50]}")
                    ok = download_pdf(pdf_url, os.path.join(folder, filename))
                    polite_sleep()

                    if ok:
                        existing.add(filename)
                        total_new += 1
                        print(f"    ✓ Saved!")
                        metadata_log[topic].append({
                            "title": title, "year": year,
                            "filename": filename, "status": "downloaded",
                            "source": "core"
                        })
                    else:
                        total_failed += 1

        # Save metadata log per topic
        log_path = os.path.join(folder, "_metadata.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(metadata_log[topic], f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  New papers downloaded : {total_new}")
    print(f"  Already existed       : {total_skipped}")
    print(f"  No free PDF / failed  : {total_failed}")
    print(f"{'='*62}")

    print("\nFinal PDF counts per folder:")
    for topic in TOPICS:
        folder = os.path.join(BASE_DIR, topic)
        count  = len([f for f in os.listdir(folder) if f.endswith(".pdf")])
        print(f"  {topic:<22}: {count} PDFs")


if __name__ == "__main__":
    download_all()