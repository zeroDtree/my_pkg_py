---
description: "Use when: writing Typst math mode with subscript + function argument notation. Ensures proper spacing between subscript and argument list."
applyTo: "**/*.typ"
---

# Typst Math Spacing

In Typst math mode, a subscript immediately followed by `(...)` is parsed as one subscript expression. Insert a space between the subscript and the argument list.

## Subscript + function argument

```typst
// ❌ BAD — phi(bold(x)) becomes part of the subscript
$E_phi(bold(x)) in RR$
$p_phi(bold(x)) >= 0$
$tilde(q)_phi(bold(x))$

// ✅ GOOD — subscript is only phi; (bold(x)) is the argument
$E_phi (bold(x)) in RR$
$p_phi (bold(x)) >= 0$
$tilde(q)_phi (bold(x))$
```

Apply the same pattern everywhere: `E_phi (bold(x))`, `p_phi (bold(x)')`, `ln p_phi (x)`, etc.
