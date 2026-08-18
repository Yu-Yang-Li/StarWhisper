"""ADS-first astronomy paper search. Standard library only.

Order: NASA ADS (if ADS_API_TOKEN / ADS_DEV_KEY is set), then arXiv astro-ph.
Dry-run prints the planned requests and never invents papers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ADS_SEARCH = "https://api.adsabs.harvard.edu/v1/search/query"
ARXIV_SEARCH = "https://export.arxiv.org/api/query"
USER_AGENT = "StarWhisper-skills/ads-first (https://github.com/Yu-Yang-Li/StarWhisper)"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def ads_token() -> str:
    return (
        os.environ.get("ADS_API_TOKEN")
        or os.environ.get("ADS_DEV_KEY")
        or os.environ.get("NASA_ADS_TOKEN")
        or ""
    ).strip()


def planned_requests(query: str, rows: int) -> dict:
    q = query.strip()
    return {
        "query": q,
        "ads": {
            "method": "GET",
            "url": ADS_SEARCH,
            "params": {
                "q": q,
                "fq": 'database:astronomy OR bibstem:"arXiv"',
                "fl": "bibcode,title,author,year,pub,doi,identifier,doctype,arxiv_class",
                "rows": rows,
                "sort": "score desc",
            },
            "auth": "Authorization: Bearer $ADS_API_TOKEN",
            "token_present": bool(ads_token()),
        },
        "arxiv": {
            "method": "GET",
            "url": ARXIV_SEARCH,
            "params": {
                "search_query": f"cat:astro-ph AND all:{q}",
                "start": 0,
                "max_results": rows,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        },
    }


def _get(url: str, headers: dict[str, str]) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""
        return exc.code, exc.headers.get("Content-Type", "") if exc.headers else "", body


def search_ads(query: str, rows: int) -> dict:
    token = ads_token()
    if not token:
        return {
            "source": "ads",
            "status": "接口受限",
            "reason": "missing ADS_API_TOKEN",
            "docs": [],
        }
    params = planned_requests(query, rows)["ads"]["params"]
    url = ADS_SEARCH + "?" + urllib.parse.urlencode(params)
    status, content_type, body = _get(
        url,
        {
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    if status != 200 or "json" not in content_type.lower():
        excerpt = body[:240].decode("utf-8", "replace")
        return {
            "source": "ads",
            "status": "接口受限",
            "http_status": status,
            "content_type": content_type,
            "raw_excerpt": excerpt,
            "docs": [],
        }
    payload = json.loads(body.decode("utf-8"))
    docs = []
    for item in payload.get("response", {}).get("docs", []):
        identifiers = item.get("identifier") or []
        arxiv_id = next((x.replace("arXiv:", "") for x in identifiers if "arXiv:" in x or x.count(".") == 1 and x[:2].isdigit()), None)
        docs.append(
            {
                "bibcode": item.get("bibcode"),
                "title": (item.get("title") or [""])[0],
                "authors": (item.get("author") or [])[:8],
                "year": item.get("year"),
                "venue": item.get("pub"),
                "doi": (item.get("doi") or [None])[0],
                "arxiv_id": arxiv_id,
                "url": f"https://ui.adsabs.harvard.edu/abs/{item.get('bibcode')}/abstract" if item.get("bibcode") else None,
                "verification_status": "ADS hit",
            }
        )
    return {"source": "ads", "status": "ok", "n_found": payload.get("response", {}).get("numFound"), "docs": docs}


def _text(node, path: str) -> str:
    found = node.find(path, ARXIV_NS)
    return (found.text or "").strip() if found is not None and found.text else ""


def search_arxiv(query: str, rows: int) -> dict:
    params = planned_requests(query, rows)["arxiv"]["params"]
    url = ARXIV_SEARCH + "?" + urllib.parse.urlencode(params)
    status, content_type, body = _get(url, {"User-Agent": USER_AGENT, "Accept": "application/atom+xml"})
    if status != 200:
        return {
            "source": "arxiv",
            "status": "接口受限",
            "http_status": status,
            "content_type": content_type,
            "raw_excerpt": body[:240].decode("utf-8", "replace"),
            "docs": [],
        }
    root = ET.fromstring(body)
    docs = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = _text(entry, "atom:id").rsplit("/abs/", 1)[-1]
        published = _text(entry, "atom:published")[:4]
        doi = _text(entry, "arxiv:doi") or None
        docs.append(
            {
                "bibcode": None,
                "title": " ".join(_text(entry, "atom:title").split()),
                "authors": [a.text for a in entry.findall("atom:author/atom:name", ARXIV_NS) if a.text][:8],
                "year": int(published) if published.isdigit() else None,
                "venue": "arXiv",
                "doi": doi,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                "verification_status": "preprint",
            }
        )
    return {"source": "arxiv", "status": "ok", "docs": docs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Search astronomy papers via NASA ADS, then arXiv astro-ph.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests; send nothing.")
    args = parser.parse_args()
    query = args.query.strip()
    if not query:
        raise SystemExit("query is empty")

    plan = planned_requests(query, args.rows)
    if args.dry_run:
        json.dump(
            {
                "query": query,
                "route": ["nasa-ads", "arxiv-astro-ph", "giiisp-supplement"],
                "planned_requests": plan,
                "docs": [],
                "safety": "dry-run only; no request was sent and no papers were invented",
            },
            fp=sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        print()
        return 0

    ads = search_ads(query, args.rows)
    arxiv = {"source": "arxiv", "status": "skipped", "docs": []}
    if not ads.get("docs"):
        arxiv = search_arxiv(query, args.rows)

    used = ads if ads.get("docs") else arxiv
    json.dump(
        {
            "query": query,
            "route": ["nasa-ads", "arxiv-astro-ph"],
            "used_source": used["source"],
            "ads_status": ads.get("status"),
            "arxiv_status": arxiv.get("status"),
            "docs": used.get("docs", []),
            "safety": "only records returned by ADS or arXiv; empty list means no verified hit",
        },
        fp=sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
