# src/vendor_scraper.py
"""
Vendor scraper utilities:
- scrape_indiamart(keyword): targeted scraping of IndiaMart directory search (best-effort)
- simple_search_scrape(keyword): DuckDuckGo HTML search fallback
- refresh_vendor_json(items, out_path): helper to write vendor_data/vendors.json
Notes: This is a lightweight, best-effort scraper for demo purposes only.
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import json
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CTAI-ProcurementBot/1.0; +mailto:your-email@example.com)"
}
DEFAULT_TIMEOUT = 10.0
RATE_SLEEP = 1.0  # seconds between requests (politeness)

def _safe_get(url, params=None, timeout=DEFAULT_TIMEOUT):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        time.sleep(RATE_SLEEP)
        return r.text
    except Exception:
        return None

def simple_search_scrape(keyword, max_results=8):
    """
    Use DuckDuckGo HTML results as a quick fallback.
    Returns list of vendor dicts: {company, url, services, location, contact}
    """
    q = f"{keyword} supplier India"
    url = "https://html.duckduckgo.com/html/"
    try:
        resp = _safe_get(url, params={"q": q})
        if not resp:
            return []
        soup = BeautifulSoup(resp, "html.parser")
        vendors = []
        # DuckDuckGo's html endpoint uses links with class 'result__a' or 'result-title'
        results = soup.select("a.result__a") or soup.select("a.result-title")
        for a in results[:max_results]:
            title = a.get_text(strip=True)
            href = a.get("href") or a.get("data-redirect")
            vendors.append({
                "company": title,
                "url": href,
                "services": keyword,
                "location": None,
                "contact": None,
                "source": "duckduckgo"
            })
        return vendors
    except Exception:
        return []

def scrape_indiamart(keyword, max_results=8):
    """
    Best-effort IndiaMart scraping.
    Returns list of vendor dicts. HTML structure may vary; this parser is tolerant.
    """
    base = "https://dir.indiamart.com/search.mp"
    params = {"ss": keyword}
    html = _safe_get(base, params=params)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    vendors = []

    # IndiaMart uses multiple result card layouts. We'll try a few selectors.
    # 1) cards with class 'prdctst' / 'comp-name' etc.
    # 2) fallback: any <a> with href containing 'detail' or company domain-like links.
    try:
        # Primary approach: look for listing blocks
        blocks = soup.select("div.srchHdng") or soup.select("div.listingBox") or soup.select("div.row")
        # But some pages put company names in h2 / h3
        if not blocks:
            blocks = soup.select("div.comp_name") or soup.select("div[data-comp]")
        seen = set()
        for b in blocks:
            # try to extract company name
            name = None
            link = None
            # company name candidates
            for sel in ["h2 a", "h3 a", "a.compName", "a", "div.company-name a"]:
                tag = b.select_one(sel)
                if tag and tag.get_text(strip=True):
                    name = tag.get_text(strip=True)
                    link = tag.get("href")
                    break
            # fallback: find any strong/bold text
            if not name:
                text = b.get_text(" ", strip=True)
                name = text.split(" - ")[0][:100] if text else None
            # location and contact heuristics
            loc = None
            contact = None
            # find text patterns inside block
            txt = b.get_text(" ", strip=True) if b else ""
            # extract phone/email if present
            phone_match = re.search(r'(\+?\d{2,4}[-\s]?)?(\d{6,12})', txt)
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', txt)
            if phone_match:
                contact = phone_match.group(0)
            if email_match:
                contact = email_match.group(0) if not contact else f"{contact}; {email_match.group(0)}"
            # dedupe
            key = (name or "") + (link or "")
            if not key or key in seen:
                continue
            seen.add(key)
            vendors.append({
                "company": name or "IndiaMart vendor",
                "url": link if link and link.startswith("http") else ("https://www.indiamart.com" + link if link else None),
                "services": keyword,
                "location": loc,
                "contact": contact,
                "source": "indiamart"
            })
            if len(vendors) >= max_results:
                break
    except Exception:
        pass

    # If primary blocks didn't produce results, try a looser parsing of result links
    if not vendors:
        try:
            anchors = soup.find_all("a", href=True)
            for a in anchors:
                href = a["href"]
                text = a.get_text(strip=True)
                if not text:
                    continue
                if "detail" in href or "company" in href or "supplier" in href:
                    vendors.append({
                        "company": text,
                        "url": href if href.startswith("http") else "https://www.indiamart.com" + href,
                        "services": keyword,
                        "location": None,
                        "contact": None,
                        "source": "indiamart"
                    })
                if len(vendors) >= max_results:
                    break
        except Exception:
            pass

    return vendors

def scrape_tradeindia(keyword, max_results=8):
    """
    Lightweight TradeIndia scraping attempt. Similar approach to IndiaMart; may be fragile.
    """
    base = "https://www.tradeindia.com/search.html"
    params = {"q": keyword}
    html = _safe_get(base, params=params)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    vendors = []
    try:
        items = soup.select("div.productBox") or soup.select("div.srclist")
        for it in items[:max_results]:
            name_tag = it.select_one("h2 a") or it.select_one("a")
            name = name_tag.get_text(strip=True) if name_tag else None
            href = name_tag.get("href") if name_tag else None
            vendors.append({
                "company": name or "TradeIndia vendor",
                "url": href if href and href.startswith("http") else ("https://www.tradeindia.com" + href if href else None),
                "services": keyword,
                "location": None,
                "contact": None,
                "source": "tradeindia"
            })
    except Exception:
        pass
    return vendors

def refresh_vendor_json(items, out_path="vendor_data/vendors.json"):
    """
    Write structured vendor_data JSON from a list of items.
    items: list of dicts or list of (MasterItemNo, vendors) pairs.
    Format:
    {"items": [{"MasterItemNo": 123, "vendors": [{company,url,services,location,contact,source}, ...]}, ...]}
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"items": []}
    # If items is a dict mapping MasterItemNo->vendors, adapt
    if isinstance(items, dict):
        for k,v in items.items():
            payload["items"].append({"MasterItemNo": int(k) if str(k).isdigit() else str(k), "vendors": v})
    else:
        # try list of vendor dicts: group under a generic entry
        payload["items"].append({"MasterItemNo": None, "vendors": items})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path
