#!/usr/bin/env python3
"""Check every relative link and heading anchor across the Markdown docs.

Documentation rot is silent: a link to a section that has been renamed or moved still looks
fine in a diff and still renders as a link. This project leans on its docs heavily and splits
them across six files, so a moved section is a real risk — splitting the README broke four
links the moment it happened, and this script is what found them.

Run with no arguments from the repository root. Exits non-zero on the first problem, so it
works as a CI gate.
"""
import pathlib, re, sys

root = pathlib.Path(".")
docs = sorted(list(root.glob("*.md")) + list(root.glob("docs/*.md")))

def slug(heading: str) -> str:
    # GitHub's rule: lowercase, drop anything that is not word/space/hyphen, then turn each
    # space into a hyphen. Runs of spaces become runs of hyphens, which is why a heading with
    # an em dash yields a double hyphen.
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")

def anchors(text):
    found, fenced = set(), False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            found.add(slug(m.group(1)))
    return found

amap = {d.as_posix(): anchors(d.read_text()) for d in docs}
broken = []
checked = 0

for d in docs:
    for link in re.findall(r"\]\(([^)\s]+)\)", d.read_text()):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        checked += 1
        target, _, frag = link.partition("#")
        if not target:
            if frag and frag not in amap[d.as_posix()]:
                broken.append((d.as_posix(), link, "no such anchor here"))
            continue
        resolved = (d.parent / target).resolve()
        if not resolved.exists():
            broken.append((d.as_posix(), link, "file not found"))
            continue
        rel = resolved.relative_to(root.resolve()).as_posix()
        if frag and rel.endswith(".md") and frag not in amap.get(rel, set()):
            broken.append((d.as_posix(), link, f"no such anchor in {rel}"))

if broken:
    print(f"{len(broken)} broken link(s) of {checked} checked:")
    for f, link, why in broken:
        print(f"  {f}: {link} -> {why}")
    sys.exit(1)
print(f"ok: {checked} relative links across {len(docs)} documents all resolve")
