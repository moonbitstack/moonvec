"""Fill the counts on the site's pages from the package interfaces.

A number that a reader sees on the homepage should come from the code, not from memory. Each count
sits between two markers in the page and is replaced in place; CI regenerates the pages and requires
an empty diff, so a stale number cannot reach the site.

Usage: python scripts/gen_stats.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "docs" / "index.html"]
MBTI = [
    ROOT / "src" / "pkg.generated.mbti",
    ROOT / "src" / "linalg" / "pkg.generated.mbti",
    ROOT / "src" / "cute" / "pkg.generated.mbti",
]


def count_items(path):
    text = path.read_text(encoding="utf-8")
    items = 0
    for line in text.splitlines():
        if re.match(r"pub(?:\(all\)|\(open\))? fn ", line):
            items += 1
        elif re.match(r"pub(?:\(all\)|\(open\))? (?:struct|enum|suberror|trait|typealias) ", line):
            items += 1
        elif re.match(r"pub let ", line):
            items += 1
    return items


def count_dtypes(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"pub\(all\) enum DType \{(.*?)\n\}", text, re.S)
    if not match:
        raise SystemExit("DType enum not found in the interface")
    return len([line for line in match.group(1).splitlines() if line.strip()])


def replace(page, name, value):
    pattern = re.compile(rf"(<!--gen:{name}-->).*?(<!--/gen-->)", re.S)
    if not pattern.search(page):
        raise SystemExit(f"marker {name} not found")
    return pattern.sub(rf"\g<1>{value}\g<2>", page)


def main():
    items = sum(count_items(path) for path in MBTI)
    dtypes = count_dtypes(MBTI[0])
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        text = replace(text, "dtypes", str(dtypes))
        text = replace(text, "public", str(items))
        page.write_text(text, encoding="utf-8", newline="\n")
    print(f"{dtypes} dtypes and {items} public items written into {len(PAGES)} page(s)")


if __name__ == "__main__":
    main()
