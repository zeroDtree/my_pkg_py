---
description: "Use when: writing Typst with inline code (backticks) and inline math (dollar signs). Prevents nesting code inside math or math inside code."
applyTo: "**/*.typ"
---

# Typst Inline Code and Math

In Typst:

- Backticks `` `...` `` mark inline code (identifiers, field names, shapes).
- Dollar signs `$...$` mark inline math.

Do not nest one inside the other — Typst treats them as separate modes and nesting breaks rendering.

```typst
// ❌ BAD — math inside backticks
`token mask $M_t$`
`以干净支路 $bold(x)^"known"$ 承载条件`

// ✅ GOOD — alternate code and math as separate spans
`tok_mask`, token mask $M_t$
以干净支路 $bold(x)^"known"$（字段 `X`）承载条件

// ❌ BAD — code inside math
$`x_t` in RR^(L times 3)$

// ✅ GOOD
`x_t` $in RR^(L times 3)$

// ❌ BAD — long backtick block that also contains $...$
`P_tok_rep[k] = coords[j]；索引 $k$，下标 $j$`

// ✅ GOOD — keep math outside the code span
`P_tok_rep[k] = coords[j]`；索引 $k$，下标 $j$
```

When both appear in one bullet or sentence, write them side by side with plain text between — never `` `...$...$...` `` or `$...`...`...$`.
