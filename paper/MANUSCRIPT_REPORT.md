# Brief 16 — Final body integration report

## Build

- Replaced Abstract through Broader Impact Statement with `paper_final_body.tex` **verbatim** (floats re-inserted; captions unchanged).
- Kept: preamble, `\input{numbers.tex}`, title, anonymous authors, bibliography, appendix A–I.
- Output: `paper/main.pdf`.

## Page counts

| Quantity | Pages |
|----------|------:|
| Main text through Broader Impact Statement | **11** (≤ 12) |
| Total PDF | **17** |

Broader Impact ends on page 11; References start on page 12; Appendix A on page 13.

## Checks

| Check | Result |
|-------|--------|
| Undefined references | **0** |
| Undefined citations | **0** |
| Overfull boxes > 10 pt | **0** |
| `pdffonts` | all embedded Type 1; **no Type 3** |
| ASCII hyphen negatives (`-0.`) in prose/tables | **0** (46 unicode −) |
| Anonymization grep | **clean** |
| PDF metadata | blank |
| Numbers audit | prose/macro/table OK; remaining PDF tokens = figure axis ticks |
| `[CHECK]` / `[PI]` | only `[PI: keep, edit or remove.]` in Appendix I |
