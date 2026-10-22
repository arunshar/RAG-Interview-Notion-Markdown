# Math rendering and compatibility

This page is a smoke-test fixture for the book’s GitHub previews. The examples cover the same inline, table, and display syntax used throughout the collection.

## Writing equations

- Keep inline equations inside GitHub’s dollar-and-backtick delimiters.
- Put a blank line before and after each display equation’s opening and closing delimiter block.
- Use supported named-operator notation and retain the appropriate limit behavior.
- Prefer named vertical-bar delimiters inside tables so a bar cannot be mistaken for a Markdown column separator.

## Inline smoke tests

Retriever: $`\mathop{\mathrm{RET}}\nolimits(q,D)`$.
Generator: $`\mathop{\mathrm{GEN}}\nolimits_\theta(x,c)`$.
An optimization operator: $`\mathop{\mathrm{arg\,max}}\nolimits_{d\in D}s(q,d)`$.

## Table smoke tests

| Case | Rendered equation |
| --- | --- |
| Named operator | $`\mathop{\mathrm{Recall}}\nolimits@k`$ |
| Norm | $`\left\lVert x-y\right\rVert_2`$ |
| Absolute value | $`\left\lvert x\right\rvert`$ |
| Fractions and subscripts | $`\frac{N}{1+n_t}`$ |
| Aligned expressions | $`\begin{aligned}a_1&=b_2\\c_3&=d_4\end{aligned}`$ |

## Display smoke tests

A full expression with subscripts:

$$
(c_1,\ldots,c_k)=\mathop{\mathrm{RET}}\nolimits(q,D),\qquad y\sim p_\theta(\cdot\mid x,c_1,\ldots,c_k).
$$

A multiline aligned expression:

$$
\begin{aligned}
a_1&=b_2,\\
c_3&=d_4.
\end{aligned}
$$

A matrix:

$$
\begin{pmatrix}1&2\\3&4\end{pmatrix}
$$

An operator with display limits:

$$
\mathop{\mathrm{arg\,max}}\limits_{d\in D}s(q,d)
$$

## Validation scope

Repository checks validate math delimiters, display-block boundaries, braces, environments, and known incompatible commands. The publishing audit also checks the actual GitHub previews for rendering errors and raw LaTeX outside intentional code examples. GitHub’s renderer can change independently of this repository.

The Notion import retains blank-line-separated display equations rather than GitHub-only rendering assets. Verify native equation and diagram blocks after importing into Notion.
