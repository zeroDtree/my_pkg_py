---
globs: **/*.typ,**/*.tex,**/*.md
---

# Notation Conventions

Follow these conventions when writing math notation in this project.

## 1. Math Symbols

- Vectors as bold lowercase: $\mathbf{a}$, matrices as bold uppercase: $\mathbf{A}$
- $a_i$ = $i$-th element of $\mathbf{a}$, $a_{ij}$ = $(i,j)$-th element of $\mathbf{A}$
- All vectors are column vectors
- $\mathbb{Z}$ integers, $\mathbb{Q}$ rationals, $\mathbb{R}$ reals, $\mathbb{C}$ complex numbers
- $\mathbb{F}$ filtration, $\mathbb{P}$ probability measure, $\mathbb{E}$ expectation operator
- $[n] := \{1,2,\dots,n\}$
- $\mathbb{Z}_n := \{0,1,\dots,n-1\}$
- $\mathbb{Z}_{\geq n} := \{i \mid i \in \mathbb{Z} \wedge i \geq n\}$
- $\overline{\mathbb{R}} := \mathbb{R} \cup \{-\infty,+\infty\}$
- $X^+ := \{x \in X \mid x > 0\}$ for $X \subseteq \overline{\mathbb{R}}$
- $X^* := \{x \in X \mid x \geq 0\}$ for $X \subseteq \overline{\mathbb{R}}$
- $\mathbf{a} \leq \mathbf{b} := \forall i,\ a_i \leq b_i$ for $\mathbf{a},\mathbf{b} \in \overline{\mathbb{R}}^n$
- Class of sets as calligraphic: $\mathcal{A}, \mathcal{B}, \mathcal{F}$
- $\mathcal{P}(X)$ = power set of $X$
- $\biguplus$ = union of disjoint sets
- $A^{\#}$ = adjoint matrix of $A$
- $(X, \tau)$ = topological space, briefly $X$
- $\mathcal{B}(X) := \sigma(\tau)$ where $(X,\tau)$ is a topological space
- $\wedge$ = **and** (logic) or **min** (function)
- $\vee$ = **or** (logic) or **max** (function)
- $A := B$ means $A$ is defined as $B$
- $A =: B$ means $B$ is defined as $A$

## 2. Logic Symbols

- Lowercase letters $p,q,r$ for propositional variables
- $\neg$ negation, $\wedge$ conjunction, $\vee$ disjunction
- $\to$ implication, $\iff$ equivalence
- $\mathscr{A} \implies \mathscr{B}$ means $(\mathscr{A} \to \mathscr{B})$ is True
- $\mathscr{A} \iff \mathscr{B}$ means $((\mathscr{A} \to \mathscr{B}) \wedge (\mathscr{B} \to \mathscr{A}))$ is True
- $\mathscr{L}_0$ = zero-order language, $\mathscr{L}$ = first-order language
- Script upper letters $\mathscr{A},\mathscr{B},\mathscr{C}$ for formulas (wfs) of $\mathscr{L}$ and $\mathscr{L}_0$

## 3. Computer Symbols

- Capital letters $A,B,C$ = local networks
- Lowercase letters $a,b,c$ = hosts in networks $A,B,C$
- Greek letters $\alpha,\beta,\gamma$ = public IPs
- Subscripts distinguish hosts in the same network: $a_1, a_2, a_5 \in A$
