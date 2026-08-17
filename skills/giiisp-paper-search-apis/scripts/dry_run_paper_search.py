import argparse
import json

BASE_URL = "https://giiisp.com"

MODES = {
    "oa": ("/first/oaPaper/searchArticlesByQuery1", lambda q, p, s: {"titleAndAbs": [q]}),
    "arxiv-abstract": ("/first/paper/searchArxivByAbstract", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv-no": ("/first/paper/searchArxivByArxivNo1", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv": ("/first/paper/searchArxiv", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv-title": ("/first/paper/searchArxivByTitle", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
}


def main():
    parser = argparse.ArgumentParser(description="Build a dry-run Giiisp paper search request.")
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument(
        "--format",
        choices=["request", "normalized-example", "fallback-example", "end-to-end-example"],
        default="request",
        help="Output a dry-run request, simulated normalized result, open-source fallback example, or end-to-end route table example.",
    )
    args = parser.parse_args()

    path, builder = MODES[args.mode]
    body = builder(args.query.strip(), args.page_num, args.page_size)
    if args.format == "normalized-example":
        output = {
            "query": args.query.strip(),
            "source_api": path,
            "request_body": body,
            "normalized_results": [
                {
                    "title": "Example Paper Title for Dry Run",
                    "authors": ["First Author", "Second Author"],
                    "year": 2025,
                    "venue": "arXiv",
                    "abstract": "Simulated abstract text showing where the matched claim evidence would be recorded.",
                    "doi": None,
                    "arxiv_id": "2501.01234",
                    "url": "https://arxiv.org/abs/2501.01234",
                    "pdf_url": "https://arxiv.org/pdf/2501.01234",
                    "match_reason": "Simulated title or abstract match for the supplied query.",
                    "verification_status": "待核验",
                }
            ],
            "citation_audit_row": {
                "claim": "Replace with the exact claim from the draft being audited.",
                "citation_placeholder": "[?]",
                "search_terms": args.query.strip(),
                "candidate_paper": "Example Paper Title for Dry Run; First Author, Second Author; 2025; arXiv",
                "evidence_field": "title/abstract",
                "link": "https://arxiv.org/abs/2501.01234",
                "status": "待核验",
                "recommended_action": "Cross-check DOI/arXiv/source page before citing.",
            },
            "safety": "normalized example only; no request was sent and no authenticated API was called",
        }
    elif args.format == "fallback-example":
        output = {
            "query": args.query.strip(),
            "giiisp_request": {
                "method": "POST",
                "url": BASE_URL + path,
                "body": body,
            },
            "giiisp_status": "接口受限",
            "fallback_reason": "Simulated Giiisp authentication failure, non-JSON login page, or unavailable endpoint.",
            "failure_response_example": {
                "http_status": 403,
                "content_type": "application/json",
                "message": "Authentication required or session expired.",
                "raw_excerpt": '{"code":403,"message":"login required"}',
            },
            "fallback_sources": [
                {
                    "name": "arXiv",
                    "best_for": "preprint discovery, title checks, and arXiv identifier verification",
                    "example_query": args.query.strip(),
                },
                {
                    "name": "OpenAlex",
                    "best_for": "open metadata, authors, institutions, concepts, and citation links",
                    "example_query": args.query.strip(),
                },
                {
                    "name": "Semantic Scholar",
                    "best_for": "topic expansion, related papers, abstracts, and influential citation signals",
                    "example_query": args.query.strip(),
                },
                {
                    "name": "Crossref",
                    "best_for": "DOI and publisher metadata verification",
                    "example_query": args.query.strip(),
                },
            ],
            "normalized_results": [
                {
                    "title": "Example Open Source Paper Title for Fallback",
                    "authors": ["First Author", "Second Author"],
                    "year": 2025,
                    "venue": "arXiv",
                    "abstract": "Simulated open source abstract text showing where fallback evidence would be recorded.",
                    "doi": None,
                    "arxiv_id": "2501.01234",
                    "url": "https://arxiv.org/abs/2501.01234",
                    "pdf_url": "https://arxiv.org/pdf/2501.01234",
                    "source_api": "arXiv / OpenAlex / Semantic Scholar / Crossref fallback",
                    "match_reason": "Simulated open source title or abstract match for the supplied query.",
                    "verification_status": "非 Giiisp 结果",
                }
            ],
            "citation_audit_row": {
                "claim": "Replace with the exact claim from the draft being audited.",
                "citation_placeholder": "[?]",
                "search_terms": args.query.strip(),
                "candidate_paper": "Example Open Source Paper Title for Fallback; First Author, Second Author; 2025; arXiv",
                "evidence_field": "open source title/abstract/identifier metadata",
                "link": "https://arxiv.org/abs/2501.01234",
                "status": "非 Giiisp 结果",
                "recommended_action": "Verify DOI, arXiv ID, publisher page, or source metadata before citing.",
            },
            "next_step": "Use open-source metadata to shortlist candidates, then cross-check identifiers and source pages before citing.",
            "safety": "fallback example only; no request was sent and no authenticated API was called",
        }
    elif args.format == "end-to-end-example":
        output = {
            "user_question": args.query.strip(),
            "routing": {
                "selected_mode": args.mode,
                "selected_api": path,
                "reason": "The question is routed to the selected dry-run mode based on the user's search intent.",
            },
            "dry_run_request": {
                "method": "POST",
                "url": BASE_URL + path,
                "headers": {"Content-Type": "application/json", "Accept": "application/json"},
                "body": body,
            },
            "output_table": [
                {
                    "检索目标": args.query.strip(),
                    "实际检索词": args.query.strip(),
                    "接口": path,
                    "论文": "Example Paper Title for End-to-End Dry Run; First Author, Second Author; 2025; arXiv",
                    "链接": "https://arxiv.org/abs/2501.01234",
                    "摘要依据": "Simulated title or abstract evidence matching the supplied query.",
                    "状态": "待核验",
                    "下一步": "Use DOI, arXiv ID, or source page metadata for a second verification pass before citing.",
                }
            ],
            "safety": "end-to-end example only; no request was sent and no authenticated API was called",
        }
    else:
        output = {
            "method": "POST",
            "url": BASE_URL + path,
            "headers": {"Content-Type": "application/json", "Accept": "application/json"},
            "body": body,
            "safety": "dry-run only; no request was sent",
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
