# Paper: When Split Conformal Overcorrects

Workshop draft. Compile-once skeleton ready for submission to a TSF /
calibration workshop (ICLR/NeurIPS/ICML workshop tracks) or TMLR.

## Compile

```
cd paper/
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # one more pass for cross-refs
```

The `-interaction=nonstopmode` flag prevents latex from halting on missing-package
errors (it logs and continues, producing a PDF if possible). On a fresh Colab
runtime, install the LaTeX packages first:

```
!apt-get update -qq && apt-get install -y -qq texlive texlive-latex-extra texlive-fonts-recommended
```

The `\includegraphics` lines for figures are commented out by default
(so the paper compiles even before figures are generated). To enable
figures:

1. Run `python scripts/generate_figures.py` from the repo root. This
   creates 4 PDFs in `paper/figures/`.
2. Uncomment the `% \begin{figure} ... % \end{figure}` blocks in
   `main.tex` (search for `\includegraphics`).

## Status checklist

Sections complete:
- Abstract
- Introduction
- Related Work
- Method
- Experiments (Sections 4.1–4.3)
- Discussion
- Conclusion
- Appendices A (dataset-id ablation), B (hyperparameters), D
  (RL picks), E (shuffled-split conformal), F (Conv1D ablation)

Sections still TODO:
- Appendix C: sensitivity to test window count (256 / 512 / 1024).
  Currently a TODO marker; can be filled when those runs land.
- Figure inclusion: see "Compile" section above.

## Numbers used in tables/figures

All numbers in Tables 1 and 2 are pulled from these JSON files
(which `scripts/generate_figures.py` reads):

- `results/diag_oracle.json` — per-agent and oracle CRPS (Table 1).
- `results/diag_conformal.json` — headline comparison (Table 2).
- `results/pilot_gate2.json` — RL training history + per-dataset eval.
- `results/diag_within_dataset.json` — Path C ceiling diagnostic.

If you regenerate these (e.g., with different test window counts),
update Tables 1 and 2 inline in `main.tex` to match.

## Venue notes

- **TMLR** (rolling): 12-page limit (excluding refs/appendix), more
  rigorous than workshops. Current draft is in TMLR scope.
- **ICLR Workshop** (March deadline for May): typically 4-pp short or
  8-pp full. Compress experiments / drop appendices to fit.
- **NeurIPS Time Series Workshop** (Oct deadline for Dec): 4-pp short or
  8-pp full. Similar compression rules.
- **ICML Workshop**: April-May deadline; usually 4–8 pp.

For a first submission, I'd recommend TMLR: rolling deadline, accepts
this kind of solid-empirical-with-honest-limitations work.
