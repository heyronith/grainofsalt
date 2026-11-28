# Brief 15 / 15b — Manuscript report

## Build

- Body from `RESULTS/paper_full_text.tex` (Brief 15) + Brief 15b audit fixes.
- Title: *Kept but Unused? Fine-Tuned Language Models Encode a Cue-Contingent Marker of Debunked Claims*.
- Authors: Anonymous. Style: TMLR anonymous.
- Output: `paper/main.pdf`.

## Brief 15b edits

1. **Limitations “Fourth”:** replaced with control-slot wording. Checked `data/confirm/K1/{CORE,MIS}/fact_framings.csv`: Core discounted framings = 400 each; +MIS = 300 each → **a quarter fewer** is correct (left as written).
2. §4.4: added “more data and longer training”.
3. App.\ F Pilot/Step~2: “Asserted-vs-all-discounted M1”.
4. App.\ F R1: six-framing / control framings wording.
5. Fig.\ A1 (`fig:steer`) caption replaced (three directions; primary site noted).
6. Minus signs: negative macros in `numbers.tex` wrapped with `\ensuremath{...}`; Tables 4 / steering / instrument use math-mode negatives. PDF body: 30 unicode minus (−), 0 ASCII hyphen-negatives.
7. App.\ H: “in-distribution AUROC”.
8. `[PI: keep, edit or remove.]` left in App.\ I.

## Page counts

| Quantity | Pages |
|----------|------:|
| Main text through end of Conclusion | **9** (≤ 12) |
| Total PDF | **14** |

## LaTeX checks

| Check | Result |
|-------|--------|
| Undefined references | **0** |
| Undefined citations | **0** |
| Overfull boxes > 10 pt | **0** |

## Fonts (`pdffonts`)

All embedded Type 1 (Latin Modern); **no Type 3**.

## Anonymization

`.tex` / report greps and `pdftotext` / `strings` of `main.pdf`: **clean**.  
PDF metadata blanked (`Author`/`Title`/`Subject`/`Creator`/`Producer`/`Keywords` empty).

## Numbers audit

Prose/table/macro numbers accounted for. Remaining PDF-only tokens are figure axis ticks from embedded plots (`0.05`–`0.85`, `0.5`/`0.50`). Exceptions per Brief 15 honored.

## Remaining `[CHECK]` / `[PI]` markers

| Location | Marker |
|----------|--------|
| App.\ I (Use of AI tools) | `[PI: keep, edit or remove.]` |

No `[CHECK]` markers.
