"""Generate docs/api.html from each package's pkg.generated.mbti and the `///` comments beside the
declarations.

The mbti file says what is public; the source says what it means. Every public item in the three
mbti files appears on the page: types with their constructors or fields, methods under the type they
belong to, operator implementations listed with that type, free functions, values, errors and traits.
The page is checked in, and CI regenerates it and requires an empty diff.

Usage: python scripts/gen_api.py [--out docs/api.html]
"""

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PACKAGES = [
    (
        "moonvec",
        "src",
        "The array type and everything that works element by element: dtypes, creation, indexing and "
        "views, broadcasting, universal functions, reductions, joining, sorting and matrix products.",
    ),
    (
        "moonvec/linalg",
        "src/linalg",
        "Dense linear algebra on the same array: solving, inversion, determinants, Cholesky, QR, SVD, "
        "symmetric eigenvalues, least squares, norms, matrix powers and Kronecker products.",
    ),
    (
        "moonvec/cute",
        "src/cute",
        "A layout is a length and a stride, or a tuple of layouts. It holds no data; it names an offset "
        "that an array view can be built on, which is what a tiled kernel needs.",
    ),
]

KINDS = {
    "enum": "Types",
    "struct": "Types",
    "suberror": "Errors",
    "trait": "Traits",
    "typealias": "Types",
}

DECL = re.compile(
    r"^(?:pub(?:\(all\)|\(open\))?\s+)?(fn|struct|enum|suberror|let|trait|typealias|impl)\b"
)


class Item:
    def __init__(self, kind, name, lines, section):
        self.kind = kind
        self.name = name
        self.lines = lines
        self.section = section
        self.members = []
        self.doc = ""
        self.methods = []
        self.impls = []

    @property
    def signature(self):
        return "\n".join(self.lines).strip()


def read_mbti(path):
    """Every declaration in an mbti file, in the order it was written."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    items = []
    section = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("// "):
            section = line[3:].strip()
            index += 1
            continue
        match = re.match(
            r"^pub(?:\(all\)|\(open\))?\s+(fn|struct|enum|suberror|let|trait|typealias)\b(.*)$",
            line,
        )
        if not match:
            if line.startswith("pub impl "):
                items.append(Item("impl", line[len("pub impl "):].strip(), [line], section))
            index += 1
            continue
        kind = match.group(1)
        rest = match.group(2).strip()
        name = declaration_name(kind, rest)
        block = [line]
        if kind in ("struct", "enum", "suberror", "trait") and line.rstrip().endswith("{"):
            index += 1
            while index < len(lines) and lines[index].rstrip() != "}":
                block.append(lines[index])
                index += 1
            if index < len(lines):
                block.append(lines[index])
        item = Item(kind, name, block, section)
        if kind in ("struct", "enum", "suberror", "trait"):
            item.members = member_lines(block)
        items.append(item)
        index += 1
    return items


def declaration_name(kind, rest):
    if kind == "fn":
        match = re.match(r"(?:\[[^\]]*\]\s*)?([A-Za-z_][\w]*(?:::[A-Za-z_][\w]*)?)", rest)
    else:
        match = re.match(r"([A-Za-z_][\w]*)", rest)
    return match.group(1) if match else rest


def member_lines(block):
    members = []
    for line in block[1:-1]:
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "#")):
            members.append(stripped)
    return members


def source_docs(package_dir):
    """Doc comment by declaration key, read from the package's own source.

    The doc block is the run of `///` lines above a declaration, allowing attribute lines such as
    `#derive(Debug)` in between. The first line, the `///|` separator, is not part of the text."""
    docs = {}
    for path in sorted(package_dir.glob("*.mbt")):
        if path.name.endswith(("_test.mbt", "_wbtest.mbt")):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not DECL.match(line):
                continue
            key = declaration_key(line)
            if not key:
                continue
            collected = []
            cursor = index - 1
            while cursor >= 0:
                text = lines[cursor].strip()
                if text.startswith("#"):
                    cursor -= 1
                    continue
                if text.startswith("///"):
                    collected.append(text)
                    cursor -= 1
                    continue
                break
            collected.reverse()
            paragraphs = []
            for entry in collected:
                body = re.sub(r"^///\|?", "", entry).strip()
                if body:
                    paragraphs.append(body)
            if paragraphs:
                docs.setdefault(key, "\n".join(paragraphs))
    return docs


def declaration_key(line):
    stripped = line.strip()
    impl = re.match(r"^(?:pub\s+)?impl\s+(.+?)\s+for\s+([\w@.]+)", stripped)
    if impl:
        method = re.match(r".*\bwith\s+(?:fn\s+)?([A-Za-z_]\w*)", stripped)
        if method:
            return f"{impl.group(2)}::{method.group(1)}"
        return f"impl {impl.group(1).strip()} for {impl.group(2)}"
    match = re.match(
        r"^(?:pub(?:\(all\)|\(open\))?\s+)?(fn|struct|enum|suberror|let|trait|typealias)\b(.*)$",
        stripped,
    )
    if not match:
        return None
    kind, rest = match.group(1), match.group(2).strip()
    if kind == "fn":
        named = re.match(r"(?:\[[^\]]*\]\s*)?([A-Za-z_][\w]*(?:::[A-Za-z_][\w]*)?)", rest)
        return named.group(1) if named else None
    named = re.match(r"([A-Za-z_][\w]*)", rest)
    return named.group(1) if named else None


def escape(text):
    return html.escape(text, quote=False)


def doc_html(text):
    if not text:
        return ""
    parts = []
    for paragraph in re.split(r"\n\s*\n", text):
        body = escape(paragraph).strip()
        body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
        body = body.replace("\n", " ")
        parts.append(f"<p>{body}</p>")
    return "".join(parts)


def signature_html(signature, name):
    shown = escape(signature)
    # The name is marked before any tag is written: `span` would otherwise match the `span` in a
    # tag that the keyword pass had already inserted.
    token = "\x00name\x00"
    if name:
        shown = re.sub(rf"\b{re.escape(escape(name))}\b", token, shown, count=1)
    # `pub(all)` ends in a bracket, where a trailing word boundary does not exist, so the pair is set
    # aside before the plain keywords are marked.
    shown = re.sub(r"\bpub\((all|open)\)", lambda m: f"\x01{m.group(1)}\x01", shown)
    shown = re.sub(
        r"\b(pub|fn|let|impl|trait|struct|enum|suberror|raise|typealias)\b",
        r'<span class="kw">\1</span>',
        shown,
    )
    shown = shown.replace("\x01all\x01", '<span class="kw">pub(all)</span>')
    shown = shown.replace("\x01open\x01", '<span class="kw">pub(open)</span>')
    shown = re.sub(r"\b([A-Z][A-Za-z0-9_]*)\b", r'<span class="ty">\1</span>', shown)
    shown = re.sub(r"(@[a-z_][\w]*\.)", r'<span class="ty">\1</span>', shown)
    if name:
        shown = shown.replace(token, f'<span class="nm">{escape(name)}</span>')
    return shown


def parameters(signature):
    """The parameters of a function signature, in order, split at top-level commas."""
    start = signature.find("(")
    if start < 0:
        return []
    depth = 0
    end = start
    for position in range(start, len(signature)):
        char = signature[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                end = position
                break
    inner = signature[start + 1 : end]
    parts = []
    depth = 0
    current = ""
    for char in inner:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    found = []
    for part in parts:
        if part == "Self":
            found.append(("self", "Self", "the receiver"))
            continue
        match = re.match(r"([A-Za-z_][\w]*)([?~]?)\s*:\s*(.+)", part, re.S)
        if not match:
            # The mbti file drops the labels of a function whose parameters are all positional, so
            # the parameter is known by its type alone.
            found.append(("—", part, ""))
            continue
        label, marker, rest = match.groups()
        default = ""
        if "=" in rest:
            rest, default = rest.split("=", 1)
            rest, default = rest.strip(), default.strip()
        if marker == "?":
            note = "optional"
        elif marker == "~":
            note = f"default <code>{escape(default)}</code>" if default else "labelled"
        else:
            note = "required"
        found.append((label, rest.strip(), note))
    return found


def signature_name(item):
    if item.kind == "fn":
        return item.name
    return item.name


def anchor(package, name):
    # Case is kept: `Cpu` the type and `cpu` the value would otherwise share an anchor.
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", f"{package}-{name}").strip("-")
    return f"api-{cleaned}"


def render_entry(item, package, docs, level=0):
    doc = docs.get(item.name if item.kind != "impl" else f"impl {item.name}", "")
    signature = item.signature
    params = parameters(signature) if item.kind == "fn" else []
    out = [f'<article class="entry" id="{anchor(package, item.name)}">']
    out.append(f'<div class="signature">{signature_html(signature, item.name if item.kind == "fn" else "")}</div>')
    if doc:
        out.append(f'<div class="doc">{doc_html(doc)}</div>')
    if params and len(params) > 0 and not (len(params) == 1 and params[0][0] == "self"):
        rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{escape(kind)}</td><td>{note}</td></tr>"
            for name, kind, note in params
        )
        out.append(
            '<table class="params"><thead><tr><th>name</th><th>type</th><th></th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    out.append("</article>")
    return "\n".join(out)


def build_package(package, directory, note):
    mbti = directory / "pkg.generated.mbti"
    items = read_mbti(mbti)
    docs = source_docs(directory)
    types = {item.name: item for item in items if item.kind in ("enum", "struct", "typealias")}
    errors = [item for item in items if item.kind == "suberror"]
    error_names = {item.name for item in errors}
    methods = {}
    impls = {}
    for item in items:
        if item.kind == "fn" and "::" in item.name:
            owner, method = item.name.split("::", 1)
            methods.setdefault(owner, []).append(item)
        elif item.kind == "impl":
            match = re.match(r"(.+?)\s+for\s+([\w@.]+)", item.name)
            if match:
                impls.setdefault(match.group(2), []).append(item)

    for owner, folds in methods.items():
        if owner in types:
            types[owner].methods.extend(folds)
        elif owner in error_names:
            continue
        else:
            types[owner] = Item("typealias", owner, [f"pub type {owner}"], "Types")
            types[owner].methods.extend(folds)
    for owner, applied in impls.items():
        if owner in types:
            types[owner].impls.extend(applied)
    for item in errors:
        item.impls.extend(impls.get(item.name, []))

    traits = [item for item in items if item.kind == "trait"]
    values = [item for item in items if item.kind == "let"]
    functions = [item for item in items if item.kind == "fn" and "::" not in item.name]

    def sort_key(entry):
        return entry.name.lower()

    body = [f'<section class="api-package" id="pkg-{package.replace("/", "-")}">']
    body.append('<header>')
    body.append(f"<h2>{escape(package)}</h2>")
    body.append(f'<p class="pkg-note">{escape(note)}</p>')
    body.append("</header>")

    if types:
        body.append('<div class="api-group"><h3>Types</h3>')
        for item in sorted(types.values(), key=sort_key):
            doc = docs.get(item.name, "")
            body.append('<article class="entry" id="' + anchor(package, item.name) + '">')
            body.append(f'<div class="signature">{signature_html(item.signature, item.name)}</div>')
            if doc:
                body.append(f'<div class="doc">{doc_html(doc)}</div>')
            if item.impls:
                names = "".join(
                    f"<li>{escape(re.sub(r'\\s+for\\s+.*$', '', impl.name))}</li>" for impl in item.impls
                )
                body.append(f'<div class="doc"><p>Operator forms:</p></div><ul class="members">{names}</ul>')
            body.append("</article>")
            for method in sorted(item.methods, key=sort_key):
                body.append(render_entry(method, package, docs))
        body.append("</div>")

    if functions:
        body.append('<div class="api-group"><h3>Functions</h3>')
        for item in sorted(functions, key=sort_key):
            body.append(render_entry(item, package, docs))
        body.append("</div>")

    if values:
        body.append('<div class="api-group"><h3>Values</h3>')
        for item in sorted(values, key=sort_key):
            body.append(render_entry(item, package, docs))
        body.append("</div>")

    if errors:
        body.append('<div class="api-group"><h3>Errors</h3>')
        for item in sorted(errors, key=sort_key):
            doc = docs.get(item.name, "")
            body.append('<article class="entry" id="' + anchor(package, item.name) + '">')
            body.append(f'<div class="signature">{signature_html(item.signature, item.name)}</div>')
            if doc:
                body.append(f'<div class="doc">{doc_html(doc)}</div>')
            if item.impls:
                names = "".join(
                    f"<li>{escape(impl.name.split(' for ')[0])}</li>" for impl in item.impls
                )
                body.append(f'<div class="doc"><p>Trait implementations:</p></div><ul class="members">{names}</ul>')
            body.append("</article>")
            for method in sorted(methods.get(item.name, []), key=sort_key):
                body.append(render_entry(method, package, docs))
        body.append("</div>")

    if traits:
        body.append('<div class="api-group"><h3>Traits</h3>')
        for item in sorted(traits, key=sort_key):
            body.append(render_entry(item, package, docs))
        body.append("</div>")

    body.append("</section>")
    return "\n".join(body), types, functions, values, errors, traits


def sidebar_package(package, types, functions, values, errors, traits):
    out = [f'<h2><a href="#pkg-{package.replace("/", "-")}" style="color:inherit;text-decoration:none">{escape(package)}</a></h2>', "<ul>"]
    for item in sorted(types.values(), key=lambda entry: entry.name.lower()):
        out.append(f'<li><a href="#{anchor(package, item.name)}">{escape(item.name)}</a></li>')
        for method in sorted(item.methods, key=lambda entry: entry.name.lower()):
            out.append(f'<li><a href="#{anchor(package, method.name)}">&nbsp;&nbsp;{escape(method.name)}</a></li>')
    for item in sorted(functions, key=lambda entry: entry.name.lower()):
        out.append(f'<li><a href="#{anchor(package, item.name)}">{escape(item.name)}</a></li>')
    for item in sorted(values + errors + traits, key=lambda entry: entry.name.lower()):
        out.append(f'<li><a href="#{anchor(package, item.name)}">{escape(item.name)}</a></li>')
    out.append("</ul>")
    return "\n".join(out)


def page(sections, side):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API reference — Numerical MoonBit Library</title>
<meta name="description" content="Every public type, function, method and value in moonvec, generated from the package interfaces.">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,600;1,8..60,600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/base.css">
<link rel="stylesheet" href="assets/docs.css">
<script>
  try {{
    const entry = performance.getEntriesByType("navigation")[0];
    if (entry && entry.type === "reload") sessionStorage.removeItem("moonvec-theme");
    else {{
      const saved = sessionStorage.getItem("moonvec-theme");
      if (saved) document.documentElement.dataset.theme = saved;
    }}
  }} catch {{}}
</script>
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>

<header class="topbar">
  <div class="wrap topbar-inner">
    <a class="brand" href="index.html">
      <svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="7" fill="currentColor"></rect>
        <g fill="var(--paper)">
          <rect x="6" y="6" width="5" height="5"></rect><rect x="13.5" y="6" width="5" height="5"></rect><rect x="21" y="6" width="5" height="5"></rect>
          <rect x="6" y="13.5" width="5" height="5"></rect><rect x="13.5" y="13.5" width="5" height="5"></rect><rect x="21" y="13.5" width="5" height="5"></rect>
          <rect x="6" y="21" width="5" height="5"></rect><rect x="13.5" y="21" width="5" height="5"></rect>
        </g>
        <rect x="21" y="21" width="5" height="5" fill="var(--teal)"></rect>
      </svg>
      <span class="brand-name">moonvec</span>
      <span class="brand-sub">numerical arrays</span>
    </a>
    <nav class="nav scroller" aria-label="Primary">
      <a href="index.html">Overview</a>
      <a href="guide.html">Guide</a>
      <a href="api.html" aria-current="page">API</a>
      <a href="playground.html">Playground</a>
    </nav>
    <div class="nav-tools">
      <button class="icon-btn" id="theme-toggle" type="button" aria-label="Switch theme">
        <svg class="i-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.2A8.6 8.6 0 1 1 9.8 3.5a7 7 0 0 0 10.7 10.7Z"></path></svg>
        <svg class="i-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.1"></circle><path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4 17 7M7 17l-1.6 1.6"></path></svg>
      </button>
      <a class="icon-btn" href="https://github.com/moonbitstack/moonvec" aria-label="Source on GitHub">
        <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path></svg>
      </a>
    </div>
  </div>
</header>

<main id="main">
  <header class="page-head">
    <div class="wrap">
      <p class="eyebrow">API reference</p>
      <h1>Everything public, in one place.</h1>
      <p class="lead">Generated from the three package interfaces and the doc comment beside each
        declaration. Use the filter to narrow by name or by a word in the description.</p>
    </div>
  </header>

  <div class="wrap api-shell">
    <aside class="api-side scroller" aria-label="Package index">
      <label class="sr-only" for="api-filter">Filter the reference</label>
      <input class="api-search" id="api-filter" type="search" placeholder="filter…" autocomplete="off" spellcheck="false">
{side}
    </aside>
    <div class="api-main">
{sections}
      <p class="api-empty" id="api-empty">No item matches that filter.</p>
    </div>
  </div>
</main>

<footer class="footer">
  <div class="wrap footer-inner">
    <div>
      <a class="brand" href="index.html">
        <svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
          <rect width="32" height="32" rx="7" fill="currentColor"></rect>
          <g fill="var(--paper)">
            <rect x="6" y="6" width="5" height="5"></rect><rect x="13.5" y="6" width="5" height="5"></rect><rect x="21" y="6" width="5" height="5"></rect>
            <rect x="6" y="13.5" width="5" height="5"></rect><rect x="13.5" y="13.5" width="5" height="5"></rect><rect x="21" y="13.5" width="5" height="5"></rect>
            <rect x="6" y="21" width="5" height="5"></rect><rect x="13.5" y="21" width="5" height="5"></rect>
          </g>
          <rect x="21" y="21" width="5" height="5" fill="var(--teal)"></rect>
        </svg>
        <span class="brand-name">moonvec</span>
      </a>
      <p class="footer-note">Numerical arrays for MoonBit. Apache-2.0.</p>
    </div>
    <nav class="footer-links" aria-label="Footer">
      <a href="guide.html">Guide</a>
      <a href="api.html">API</a>
      <a href="playground.html">Playground</a>
      <a href="https://mooncakes.io/docs/moonbitstack/moonvec">mooncakes</a>
      <a href="https://github.com/moonbitstack/moonvec">GitHub</a>
    </nav>
  </div>
</footer>

<script src="assets/site.js"></script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/api.html")
    args = parser.parse_args()

    sections = []
    side = []
    totals = [0, 0, 0]
    for package, directory, note in PACKAGES:
        built, types, functions, values, errors, traits = build_package(
            package, ROOT / directory, note
        )
        sections.append(built)
        side.append(sidebar_package(package, types, functions, values, errors, traits))
        totals[0] += len(types) + len(functions) + len(values) + len(errors) + len(traits)
        totals[1] += sum(len(item.methods) for item in types.values())
        totals[2] += sum(len(item.impls) for item in types.values())

    output = page("\n".join(sections), "\n".join(side))
    path = ROOT / args.out
    path.write_text(output, encoding="utf-8", newline="\n")
    entries = output.count('<article class="entry"')
    print(
        f"{totals[0]} named items, {totals[1]} methods and {totals[2]} operator impls "
        f"written to {args.out} ({entries} entries)"
    )


if __name__ == "__main__":
    main()
