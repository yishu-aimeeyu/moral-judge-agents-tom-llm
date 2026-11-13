# Venue requirements (checked 9 Oct 2026)

`../main.pdf` is the integrated journal-length version: about 16 pages of main text, 3 pages of references, and a 12-page supplement (`\input{supplement}`). It is not a conference submission. The venue drafts in this folder are the submission versions.

| Venue | Draft | Length rule | Appendix | Archival / parallel submission | Template |
|---|---|---|---|---|---|
| ICLR 2027 workshop | `iclr2027_workshop/` | Set by each workshop; recent ICLR workshop short tracks were typically 4 pages excluding references | Usually unlimited after references; reviewers need not read it | Usually non-archival; confirm per workshop | ICLR style (2026 files until 2027 files are released) |
| CogSci 2027 | `cogsci2027/` | 6 pages including figures, tables, and appendices; references unlimited | Must fit inside the 6 pages (or link an anonymized OSF supplement) | Society states proceedings are not considered archival for later journal submission; an abstract-only publication option exists | `cogsci.sty` (2026 template), biblatex APA, double-blind |
| CCN 2027, Extended Abstract | `ccn2027_extended_abstract/` | 2 pages including title block and abstract; references excluded | Not permitted | No DOI; parallel submission allowed | `ccn.cls` (2026 template), biblatex APA, double-blind |
| CCN 2027, Proceedings (not drafted) | — | 8 pages excluding references | Allowed | DOI in CCN-P; parallel submission not allowed | `ccn.cls` LaTeX or Typst only |

## Dates
- ICLR 2027 workshops: accepted workshops announced 29 Nov 2026; ICLR's suggested contribution deadline is 1 Feb 2027; organizers must notify authors by 26 Feb 2027; workshops are 29-30 Apr 2027. Each workshop's exact format is posted in its own CFP.
- CogSci 2027: Bilbao, Spain, 28-31 Jul 2027 (the spreadsheet's San Francisco entry is out of date). The 2027 deadline was not posted; the 2026 cycle closed 2 Feb 2026.
- CCN 2027: Edinburgh, 26-29 Jul 2027. The 2027 call was not posted; the 2026 pattern was Proceedings abstracts 9 Feb, full papers 12 Feb, and Extended Abstracts 2 Apr.

## Corrections to `Paper_submission_timeline.xlsx`
- CogSci 2027 is in Bilbao, 28-31 Jul, not San Francisco, 30 Jul-2 Aug.
- CogSci appendices count toward the 6 pages.
- The CCN Proceedings track (8 pages, DOI) is archival-like and forbids parallel submission. Only the 2-page Extended Abstract is non-archival.
- The spreadsheet note calling LEACE erasure fabricated is out of date: LEACE outcome erasure is a real project result (`outputs/experiments/W8A_AMNESIC.md`, `code/experiments/63_w8a_amnesic_outcome.py`).

## Sources
- https://iclr.cc/Conferences/2027/CallForWorkshops
- Example ICLR 2026 workshop CFPs: https://trustworthy-ai-workshop.github.io/iclr2026/, https://hc-air.github.io/hcair26/, https://sites.google.com/view/iclr-2026-llmreasoning
- https://cognitivesciencesociety.org/cogsci-2027/ ; https://cognitivesciencesociety.org/submissions/ ; https://cognitivesciencesociety.org/submissions-faq/ ; https://github.com/neilbramley/cogscisoc
- https://2027.ccneuro.org/ ; https://2026.ccneuro.org/call-for-papers/ ; https://cogcompneuro.github.io/docs/2026/Extended%20Abstracts/Submission%20guidelines/ ; https://cogcompneuro.github.io/docs/2026/Proceedings/Submission%20guidelines/ ; https://github.com/cogcompneuro/ccn-template

## Before submitting
- Replace the 2026 template files with the 2027 releases and recheck page counts.
- For ICLR, follow the chosen workshop's CFP (page limit, anonymization, and whether references or statements count).
- Complete the AI-use statement in each draft; CogSci and CCN require it.
- Closed-API results remain provisional until archival provenance is completed.
- Compile CogSci and CCN with biber, which is the Overleaf default for biblatex.
