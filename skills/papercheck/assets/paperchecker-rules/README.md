# Local citation rules

This is the citation-checking subset of TaShan-PaperChecker distributed with
the [upstream research skills](https://github.com/TashanGKD/tashan-research-skills).
The original MIT [license](LICENSE) is retained.

It includes document extraction, numbered and author-year citation matching,
UCAS / GB/T 7714 checks, and the report contract used by PaperCheck. Web pages,
API hosting, accounts, SMS, remote semantic-review clients and deployment tools
are excluded. Semantic review is handled by the model running the skill.

From this directory, install DOCX dependencies with
`python -m pip install -r requirements.txt`. For local PDF parsing, use
`python -m pip install -r requirements-pdf.txt`.

Use the entry commands in [PaperCheck](../../SKILL.md) and its
[runtime notes](../../references/runtime.md). This subset does not run a server.
The complete earlier bundle remains available in Git history.
