# Report Build Instructions

This report compiles from `docs/report/main.tex` into a single PDF output.

## Compile target

- Final PDF path: `docs/report/build/main.pdf`

## Build steps

From the repository root:

```bash
mkdir -p docs/report/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=docs/report/build docs/report/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=docs/report/build docs/report/main.tex
```

Running `pdflatex` twice ensures references and table of contents metadata are resolved.
