# vendored from sfxnz/forge kit @ 3d294ff
# Regenerate the marked blocks of run.sh and README.md from recipe.yaml.
#
#   python3 kit/render.py            rewrite the generated blocks in place
#   python3 kit/render.py --check    exit 1 with a unified diff if any block is stale
#   python3 kit/render.py --strict   also exit 1 when a measured row has no evidence file
#
# recipe.yaml is the source of truth. Nothing outside these markers is touched:
#   run.sh     # BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
#              # END generated
#   README.md  <!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
#              <!-- END generated defaults -->
#              <!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
#              <!-- END generated measured -->
#
# Inside the run.sh block every NAME="${NAME:-value}" line takes its value from serve.env; comment and
# derived lines are kept verbatim. serve.env must list exactly those names, in run.sh order. README
# defaults rows may use {NAME} placeholders for serve.env values. Needs python3 and PyYAML.
import argparse
import difflib
import re
import sys
from pathlib import Path

import yaml

RUN_BEGIN = "# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py"
RUN_END = "# END generated"
DEFAULT_LINE = re.compile(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-.*\}"$')
PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
DEFAULTS_HEADER = ["| Setting | Value |", "|---|---|"]
MEASURED_HEADER = [
    "| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |",
    "|---|---|---:|---:|---:|",
]
NO_EVIDENCE = ("", "null", "~")  # BaseLoader keeps `null` as the string "null"


def md_markers(name):
    return (
        f"<!-- BEGIN generated {name} from recipe.yaml — edit recipe.yaml and run kit/render.py -->",
        f"<!-- END generated {name} -->",
    )


def find_block(lines, begin, end, path):
    """Return (start, stop) so that lines[start:stop] is the body between the marker lines."""
    starts = [i for i, line in enumerate(lines) if line == begin]
    if len(starts) != 1:
        sys.exit(f"render: {path}: need exactly one `{begin}` line, found {len(starts)}")
    try:
        stop = lines.index(end, starts[0] + 1)
    except ValueError:
        sys.exit(f"render: {path}: `{begin}` has no `{end}`")
    return starts[0] + 1, stop


def render_run_sh(text, env):
    lines = text.split("\n")
    start, stop = find_block(lines, RUN_BEGIN, RUN_END, "run.sh")
    body, seen = [], []
    for line in lines[start:stop]:
        m = DEFAULT_LINE.match(line)
        if not m:
            body.append(line)
            continue
        name = m.group(1)
        if name not in env:
            sys.exit(f"render: run.sh sets {name} inside the generated block but serve.env has no {name}")
        seen.append(name)
        body.append(f'{name}="${{{name}:-{env[name]}}}"')
    if seen != list(env):
        sys.exit(
            "render: serve.env must list the generated run.sh defaults in run.sh order\n"
            f"  run.sh:      {' '.join(seen)}\n  recipe.yaml: {' '.join(env)}"
        )
    return "\n".join(lines[:start] + body + lines[stop:])


def fill(template, env):
    def value(m):
        name = m.group(1)
        if name not in env:
            sys.exit(f"render: README row uses {{{name}}} but serve.env has no {name}")
        return env[name]

    return PLACEHOLDER.sub(value, template)


def render_readme(text, recipe):
    env = recipe["serve"]["env"]
    lines = text.split("\n")
    start, stop = find_block(lines, *md_markers("defaults"), "README.md")
    rows = [f"| {fill(k, env)} | {fill(v, env)} |" for row in recipe["readme"]["defaults"] for k, v in row.items()]
    lines[start:stop] = DEFAULTS_HEADER + rows
    start, stop = find_block(lines, *md_markers("measured"), "README.md")
    rows = [
        f"| {r['phase']} | {r['concurrency']} | {r['decode']} | {r['aggregate']} | {r['ttft_p50']} s |"
        for r in recipe["measured"]["decode"]["rows"]
    ]
    lines[start:stop] = MEASURED_HEADER + rows
    return "\n".join(lines)


def evidence_gaps(recipe, repo):
    none, missing = [], []
    for r in recipe["measured"]["decode"]["rows"]:
        label = f"measured.decode {r['phase']} c={r['concurrency']}"
        path = r.get("evidence", "")
        if path in NO_EVIDENCE:
            none.append(label)
        elif not (repo / path).exists():
            missing.append(f"{label}: {path}")
    return none, missing


def main():
    ap = argparse.ArgumentParser(description="Regenerate run.sh and README.md blocks from recipe.yaml.")
    ap.add_argument("--check", action="store_true", help="print a diff and exit 1 instead of writing")
    ap.add_argument("--strict", action="store_true", help="a row without an evidence file is an error")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    recipe = yaml.load((repo / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)
    renderers = {
        "run.sh": lambda text: render_run_sh(text, recipe["serve"]["env"]),
        "README.md": lambda text: render_readme(text, recipe),
    }
    stale = False
    for name, render in renderers.items():
        old = (repo / name).read_text()
        new = render(old)
        if new == old:
            continue
        if args.check:
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))
            stale = True
        else:
            (repo / name).write_text(new)
            print(f"render: wrote {name}")
    none, missing = evidence_gaps(recipe, repo)
    for label in none:
        print(f"render: no evidence yet: {label}")
    for label in missing:
        print(f"render: WARNING evidence file missing: {label}")
    if stale:
        sys.exit("render: generated blocks are stale. Run: python3 kit/render.py")
    if args.strict and (none or missing):
        sys.exit("render: --strict: every measured row needs an existing evidence file")


if __name__ == "__main__":
    main()
