#!/usr/bin/env python3
"""Self-verify id.md against en.md per translation rules."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fences(t):
    out, cur = [], None
    for line in t.split("\n"):
        s = line.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            if cur is None:
                cur = [line]
            else:
                cur.append(line)
                out.append("\n".join(cur))
                cur = None
        elif cur is not None:
            cur.append(line)
    return out


def urls(t):
    return sorted(set(re.findall(r"https?://[^\s)]+", t)))


def meta_lines(t):
    return [
        l
        for l in t.split("\n")
        if l.startswith("**Type:**")
        or l.startswith("**Languages:**")
        or l.startswith("**Prerequisites:**")
        or l.startswith("**Time:**")
    ]


def check(en_path, id_path):
    errs = []
    src = en_path.read_text(encoding="utf-8")
    dst = id_path.read_text(encoding="utf-8")
    fe, fi = fences(src), fences(dst)
    if fe != fi:
        errs.append(f"fences differ (en={len(fe)}, id={len(fi)})")
        for k, (a, b) in enumerate(zip(fe, fi)):
            if a != b:
                errs.append(f"  first diff fence {k}:\n  EN: {a[:220]!r}\n  ID: {b[:220]!r}")
                break
    ms, mi = meta_lines(src), meta_lines(dst)
    if ms != mi:
        errs.append(f"metadata differ:\n  EN: {ms}\n  ID: {mi}")
    us, ui = set(urls(src)), set(urls(dst))
    if us - ui:
        errs.append(f"missing urls: {sorted(us - ui)}")
    if ui - us:
        errs.append(f"extra urls: {sorted(ui - us)}")
    # display math blocks
    if src.count("$$") != dst.count("$$"):
        errs.append("$$ count differs")
    return errs


def lesson_list():
    ens = sorted((ROOT / "phases" / "12-multimodal-ai").glob("[0-9][0-9]-*/docs/en.md")) + sorted(
        (ROOT / "phases" / "13-tools-and-protocols").glob("[0-9][0-9]-*/docs/en.md")
    )
    return ens


if __name__ == "__main__":
    only = sys.argv[1:] or None
    n_ok = n_bad = n_miss = 0
    for e in lesson_list():
        parts = e.parts
        tgt = ROOT / "i18n" / "id" / parts[0] / parts[1] / parts[2] / "docs" / "id.md"
        rel = f"{parts[1]}/{parts[2]}"
        if only and not any(o in rel for o in only):
            continue
        if not tgt.is_file():
            print(f"MISS  {rel}")
            n_miss += 1
            continue
        errs = check(e, tgt)
        if errs:
            print(f"FAIL  {rel}")
            for x in errs:
                print(f"      {x}")
            n_bad += 1
        else:
            print(f"OK    {rel}")
            n_ok += 1
    print(f"== ok={n_ok} fail={n_bad} miss={n_miss}")
