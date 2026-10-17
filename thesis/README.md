# Thesis (LaTeX)

Open `thesis/main.tex` in VS Code. LaTeX Workshop builds on save (pdflatex + biber via latexmk)
and puts everything in `thesis/build/` (gitignored). The PDF is `thesis/build/main.pdf`.
From a terminal: `cd thesis && latexmk main.tex`.

| file | content |
|---|---|
| `main.tex` | document order; switch sections on and off here |
| `preamble.tex` | packages and macros; `\drafttrue` / `\draftfalse` at the top |
| `frontmatter/` | title page, abstract, declaration |
| `sections/01..09` | introduction to conclusion |
| `appendix/A..D` | pilot studies, prompts, extra results, reproducibility |
| `references.bib` | bibliography; entries marked VERIFY need checking |
| `figures/` | new thesis figures (script figures in `D3code-main/scripts/figures/` also work) |

Writing aids (draft mode):
- `\fillin{...}`: red placeholder, stays visible even in final mode so nothing slips through.
- `\src{...}`: grey note naming the report or CSV a number came from; hidden in final mode.
- `\todo{...}`: margin note, collected in the "Open TODOs" list after the contents.
- Each section file starts with comments: what goes there, the source reports, the current numbers.

Citations: `\citet{key}` gives "Author (Year)", `\citep{key}` gives "(Author, Year)".
Cross references: `\cref{tab:steering}` gives "Table 11".
Macros: `\llama`, `\qwen`, `\dthree`, `\pyes`, `\variant{prefix}`, `\cond{a woman}`.
