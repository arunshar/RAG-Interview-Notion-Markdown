#!/usr/bin/env python3
"""Check portable GitHub math syntax without third-party dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INLINE = re.compile(r"\$`(?P<tex>.*?)`\$", re.S)
DISPLAY_BOUNDARY = re.compile(r"(?m)^[ \t]*\$\$[ \t]*$")
FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
ENVIRONMENT = re.compile(r"\\(begin|end)\{([^{}]+)\}")
BLOCKED = re.compile(r"\\operatorname\b")


@dataclass(frozen=True)
class MathSpan:
    start: int
    end: int
    tex_start: int
    tex_end: int
    kind: str
    syntax: str = "dollar"

    def tex(self, text: str) -> str:
        return text[self.tex_start:self.tex_end]


def mask_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in ranges:
        for i in range(start, end):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Return complete fenced literals, including an unclosed final fence."""
    result = []
    active = None
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if active:
            char, size, start = active
            if re.fullmatch(r" {0,3}" + re.escape(char) +
                            "{" + str(size) + r",}[ \t]*", body):
                result.append((start, offset + len(line)))
                active = None
        else:
            match = FENCE_START.match(body)
            if match:
                active = (match[1][0], len(match[1]), offset)
        offset += len(line)
    if active:
        result.append((active[2], len(text)))
    return result


def scan_math(text: str) -> tuple[list[MathSpan], list[tuple[int, str]]]:
    """Find inline math, protected GitHub math fences, and dollar display math."""
    fences = fenced_ranges(text)
    visible = mask_ranges(text, fences)
    boundaries = list(DISPLAY_BOUNDARY.finditer(visible))
    spans = []
    errors = []
    for start, end in fences:
        lines = text[start:end].splitlines(keepends=True)
        opening = FENCE_START.match(lines[0].rstrip("\r\n"))
        if opening[2].strip() != "math":
            continue
        marker = opening[1]
        if len(lines) < 2 or not re.fullmatch(
                r" {0,3}" + re.escape(marker[0]) +
                "{" + str(len(marker)) + r",}[ \t]*", lines[-1].rstrip("\r\n")):
            errors.append((start, "Unclosed math fence."))
            continue
        closing_start = end - len(lines[-1])
        closing_end = closing_start + len(lines[-1].rstrip("\r\n"))
        tex_start = start + len(lines[0])
        tex_end = closing_start
        if tex_end > tex_start and text[tex_end - 1] == "\n":
            tex_end -= 1
        if tex_end > tex_start and text[tex_end - 1] == "\r":
            tex_end -= 1
        spans.append(MathSpan(start, closing_end, tex_start, tex_end,
                              "display", "fence"))
    if len(boundaries) % 2:
        errors.append((boundaries[-1].start(), "Unpaired display-math delimiter."))
    for opening, closing in zip(boundaries[::2], boundaries[1::2]):
        tex_start = opening.end()
        if text[tex_start:tex_start + 1] == "\n":
            tex_start += 1
        tex_end = closing.start()
        if text[tex_end - 1:tex_end] == "\n":
            tex_end -= 1
        spans.append(MathSpan(opening.start(), closing.end(),
                              tex_start, tex_end, "display"))
    visible = mask_ranges(visible, [(s.start, s.end) for s in spans])
    for match in INLINE.finditer(visible):
        spans.append(MathSpan(match.start(), match.end(),
                              match.start("tex"), match.end("tex"), "inline"))
    visible = mask_ranges(visible, [(s.start, s.end) for s in spans])
    # Intentional inline code may discuss math delimiters and TeX commands.
    visible = re.sub(r"(`+).*?\1", lambda m: " " * len(m[0]), visible, flags=re.S)
    for match in re.finditer(r"\$`|`\$", visible):
        errors.append((match.start(), "Unpaired inline-math delimiter."))
    spans.sort(key=lambda span: span.start)
    return spans, errors


def tex_errors(tex: str) -> list[str]:
    errors = []
    if not tex.strip():
        errors.append("Empty math expression.")
    if BLOCKED.search(tex):
        errors.append(r"GitHub rejects \operatorname; use supported operator notation.")
    depth = 0
    i = 0
    while i < len(tex):
        char = tex[i]
        if char == "\\":
            i += 1
            if i < len(tex) and tex[i].isalpha():
                while i < len(tex) and tex[i].isalpha():
                    i += 1
            else:
                i += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                errors.append("Closing brace without an opening brace.")
                depth = 0
        i += 1
    if depth:
        errors.append("Unclosed math brace.")
    environments = []
    for match in ENVIRONMENT.finditer(tex):
        action, name = match.groups()
        if action == "begin":
            environments.append(name)
        elif not environments or environments.pop() != name:
            errors.append("Mismatched math environment: " + name)
    if environments:
        errors.append("Unclosed math environment: " + ", ".join(environments))
    return errors


def validate_text(text: str) -> tuple[list[str], dict[str, int]]:
    spans, problems = scan_math(text)
    stats = {"inline": 0, "display": 0}
    for span in spans:
        stats[span.kind] += 1
        tex = span.tex(text)
        problems.extend((span.start, message) for message in tex_errors(tex))
        if span.kind == "inline":
            if "\n" in tex:
                problems.append((span.start, "Inline math must stay on one source line."))
            line_start = text.rfind("\n", 0, span.start) + 1
            if text[line_start:span.start].lstrip().startswith("|") and "|" in tex:
                problems.append((span.start, "Use named bar delimiters inside table math."))
        else:
            before = text[:span.start].splitlines()
            after = text[span.end:].splitlines()
            if before and before[-1].strip():
                problems.append((span.start, "Display math needs a blank line before it."))
            # The first item is the remainder of the closing-delimiter line.
            if len(after) > 1 and after[1].strip():
                problems.append((span.end, "Display math needs a blank line after it."))
            if span.syntax == "dollar" and (
                    re.search(r"[<*]|\\[{}]|\]\s*\(", tex) or
                    re.search(r"(?m)^[ \t]*(?:[+=-](?:[ \t]|$)|#{1,6}[ \t])", tex)):
                problems.append((span.start,
                                 "Markdown-sensitive display math needs a protected math fence."))
    visible = mask_ranges(text, fenced_ranges(text) +
                          [(span.start, span.end) for span in spans])
    visible = re.sub(r"(`+).*?\1", lambda m: " " * len(m[0]), visible, flags=re.S)
    for match in re.finditer(r"\\[A-Za-z]{2,}|\$\$", visible):
        problems.append((match.start(), "Possible raw LaTeX outside math or intentional code."))
    errors = [
        f"line {text.count(chr(10), 0, position) + 1}: {message}"
        for position, message in problems
    ]
    return errors, stats


def validate_paths(paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    errors = []
    totals = {"files": len(paths), "inline": 0, "display": 0}
    for file in paths:
        problems, stats = validate_text(file.read_text(encoding="utf-8"))
        label = file.relative_to(ROOT) if file.is_relative_to(ROOT) else file
        errors.extend(f"{label}:{problem}" for problem in problems)
        for kind in ("inline", "display"):
            totals[kind] += stats[kind]
    return errors, totals


def main() -> int:
    paths = ([Path(p).resolve() for p in sys.argv[1:]] if len(sys.argv) > 1 else
             sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts))
    errors, totals = validate_paths(paths)
    if errors:
        print("\n".join("ERROR: " + error for error in errors), file=sys.stderr)
        return 1
    print(f"PASS: math syntax in {totals['files']} Markdown files; "
          f"{totals['inline']} inline and {totals['display']} display expressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())