# PT2 GitHub Issues Data

## Current Dataset
- **File:** `pt2_all_issues.json` — 2,970 issues from pytorch/pytorch with `oncall: pt2` label
- **Downloaded:** via GitHub API (see `fetch_issues.py`)
- **Coverage:** 1,220 open + 1,750 closed = 2,970 total
- **Known gap:** GitHub currently shows 2,076 open + 7,089 closed = 9,165 total. Our dataset is ~32% of the full set. Likely a pagination issue in the fetch script — needs re-fetching with proper pagination to get all ~9K issues.

## Files
- `pt2_all_issues.json` — raw issue data (number, title, state, created_at, updated_at, comments, labels, body, html_url, user)
- `pt2_analysis_data.json` — processed analysis data (journey classification, pain points)
- `pt2_analysis_summary.json` — summary statistics
- `pt2_analysis_output.txt` — human-readable analysis output
- `oncall-pt2-issue-analysis.md` — detailed analysis writeup
- `fetch_issues.py` — script used to download issues
- `analyze_issues.py` — script used to classify and analyze issues

## TODO
- [ ] Re-fetch with proper pagination to get all ~9,165 issues
- [ ] Re-run journey classification on full dataset
- [ ] Validate close rates against full dataset
- [ ] Add new journeys (compile-time, regression) to classification
