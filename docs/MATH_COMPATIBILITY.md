# Math rendering and compatibility

This page is a smoke-test fixture for the book’s GitHub previews. The examples cover the same inline, table, and display syntax used throughout the collection.

## Writing equations

- Keep inline equations inside GitHub’s dollar-and-backtick delimiters.
- Use a fenced `math` block for display equations, separated from the surrounding prose with blank lines. Fences protect TeX from Markdown list, link, emphasis, and HTML parsing.
- Use supported named-operator notation and retain the appropriate limit behavior.
- Use `\lt` and `\gt` for comparison symbols so the renderer cannot mistake them for HTML.
- Prefer named vertical-bar delimiters inside tables so a bar cannot be mistaken for a Markdown column separator.

## Inline smoke tests

Retriever: $`\mathop{\mathrm{RET}}\nolimits(q,D)`$.
Generator: $`\mathop{\mathrm{GEN}}\nolimits_\theta(x,c)`$.
An optimization operator: $`\mathop{\mathrm{arg\,max}}\nolimits_{d\in D}s(q,d)`$.

A comparison in a history subscript: $`x_{\lt t},\quad z\gt 0`$.

## Table smoke tests

| Case | Rendered equation |
| --- | --- |
| Named operator | $`\mathop{\mathrm{Recall}}\nolimits@k`$ |
| Norm | $`\left\lVert x-y\right\rVert_2`$ |
| Absolute value | $`\left\lvert x\right\rvert`$ |
| Comparisons | $`0\lt p\lt 1,\quad x\gt 0`$ |
| Fractions and subscripts | $`\frac{N}{1+n_t}`$ |
| Aligned expressions | $`\begin{aligned}a_1&=b_2\\c_3&=d_4\end{aligned}`$ |

## Display smoke tests

A full expression with subscripts:

```math
(c_1,\ldots,c_k)=\mathop{\mathrm{RET}}\nolimits(q,D),\qquad y\sim p_\theta(\cdot\mid x,c_1,\ldots,c_k).
```

A multiline aligned expression:

```math
\begin{aligned}
a_1&=b_2,\\
c_3&=d_4.
\end{aligned}
```

A matrix:

```math
\begin{pmatrix}1&2\\3&4\end{pmatrix}
```

An operator with display limits:

```math
\mathop{\mathrm{arg\,max}}\limits_{d\in D}s(q,d)
```

A comparison inside a display subscript:

```math
\sum_t\log p_\theta(y_t\mid y_{\lt t},x),\qquad 0\lt p\lt 1.
```

## Compatibility changes

GitHub currently rejects `\operatorname`. The book uses `\mathop{\mathrm{…}}\nolimits` for the same upright operator labels and side-subscript behavior. Explicit limit placement is retained where requested. Raw angle comparison signs are normalized to the equivalent `\lt` and `\gt` commands. Protected math fences preserve stars, braces, multiline alignments, and bracket expressions. The equations are not replaced by images or external rendering services.

## Validation scope

Repository checks validate math delimiters, display-block boundaries, braces, environments, and known incompatible commands. The publishing audit also checks the actual GitHub previews for rendering errors and raw LaTeX outside intentional code examples. GitHub’s renderer can change independently of this repository.

The import builder converts the protected math fences to Notion-native display-equation delimiters with the TeX contents unchanged. Code and diagram fences are not converted. Verify native equation and diagram blocks after importing into Notion.
