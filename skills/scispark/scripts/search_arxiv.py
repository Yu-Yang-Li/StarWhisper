#!/usr/bin/env python
"""Search arXiv and emit normalized Scispark literature records."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def text(entry: ET.Element, name: str) -> str:
    return " ".join(entry.findtext(f"atom:{name}", default="", namespaces=ATOM).split())


def search_arxiv(query: str, max_results: int) -> list[dict]:
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    if len(terms) <= 1:
        search_query = f'all:"{query}"'
    else:
        search_query = " AND ".join(f'all:"{term}"' for term in terms)
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = "https://export.arxiv.org/api/query?" + params
    with urllib.request.urlopen(url, timeout=30) as response:
        root = ET.fromstring(response.read())

    records = []
    for entry in root.findall("atom:entry", ATOM):
        entry_url = text(entry, "id")
        arxiv_id = entry_url.rsplit("/", 1)[-1]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM)
            for a in entry.findall("atom:author", ATOM)
        ]
        title = text(entry, "title")
        abstract = text(entry, "summary")
        year = text(entry, "published")[:4]
        categories = [c.attrib.get("term") for c in entry.findall("atom:category", ATOM)]
        records.append(
            {
                "title": title,
                "authors": [a for a in authors if a],
                "year": year,
                "venue": "arXiv",
                "doi": None,
                "arxiv_id": arxiv_id,
                "url": entry_url,
                "pdf_url": pdf_url,
                "abstract": abstract,
                "categories": categories,
                "source_api": "arXiv API",
                "query": query,
                "verification_status": "待核验",
                "match_reason": "Returned by arXiv relevance search for the Scispark query.",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args()

    records = search_arxiv(args.query, args.max_results)
    payload = {
        "query": args.query,
        "source_api": "arXiv API",
        "count": len(records),
        "normalized_results": records,
    }
    text_out = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text_out, encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
