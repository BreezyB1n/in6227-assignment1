"""Stage 5 - render narrative.json + results.json into a LaTeX/PDF report.

Numbers never come from the LLM's prose: the narrative refers to facts with
tokens such as [[R/models/lr/test/average_precision|3]] which are resolved
from results.json (R/), profile.json (P/) or plan.json (L/). Tables are built
directly from results.json. Unresolvable tokens abort the render.

Usage:
    python render_report.py --run runs/x [--max-pages 2]
Reads  runs/x/{profile,plan,results,narrative}.json, fig_results.pdf, optional reflection.tex
Writes runs/x/report.tex, report.pdf, numbers_audit.md
Exit code 3 if the main report exceeds --max-pages.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

TOKEN = re.compile(r"\[\[([A-Z])/([^\]|]+)(?:\|([a-z0-9+]+))?\]\]")
LITERAL_NUMBER = re.compile(r"(?<![\w/\[])(\d+\.\d+|\d{2,}(?:,\d{3})*|\d+\\?%)(?![\w\]])")
TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "templates" / "report.tex"


def tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def lookup(root: dict, path: str):
    cur = root
    for part in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def fmt(v, spec: str | None) -> str:
    if spec in (None, ""):
        spec = "3"
    if spec == "raw" or isinstance(v, str):
        return tex_escape(v) if isinstance(v, str) else str(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(fmt(x, spec) for x in v)
    if spec == "int":
        return f"{int(round(v)):,}".replace(",", "{,}")
    if spec.startswith("pct"):
        d = int(spec[3:] or 1)
        return f"{100 * v:.{d}f}\\%"
    if spec.startswith("+"):
        return f"{v:+.{int(spec[1:] or 3)}f}"
    if spec == "p":  # p-value
        return "$<$0.001" if v < 0.001 else f"{v:.3f}"
    if spec == "g":
        return f"{v:.3g}"
    return f"{v:.{int(spec)}f}"


class Resolver:
    def __init__(self, sources: dict):
        self.sources, self.used, self.missing = sources, [], []

    def __call__(self, text: str) -> str:
        def sub(m):
            src, path, spec = m.group(1), m.group(2), m.group(3)
            try:
                val = lookup(self.sources[src], path)
            except (KeyError, IndexError, ValueError):
                self.missing.append(m.group(0))
                return "\\textbf{??}"
            out = fmt(val, spec)
            self.used.append((m.group(0), val, out))
            return out
        return TOKEN.sub(sub, text)


# ----------------------------------------------------------------- auto tables
def ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def results_table(R: dict) -> str:
    ids = list(R["models"])
    comp = R.get("comparison", {})
    cis = comp.get("bootstrap_ci", {})
    if R["task"] == "binary":
        pm = R["cv"]["primary_metric"]
        pm_name = {"average_precision": "AP", "roc_auc": "ROC-AUC", "f1_macro": "Macro-F1",
                   "balanced_accuracy": "Bal.\\ Acc."}.get(pm, tex_escape(pm))
        head = (f"Model & CV {pm_name} (mean$\\pm$sd) & Test AP [95\\% CI] & ROC-AUC & $t$ & Prec. & Recall & F1 & "
                "Macro-F1 & Bal.\\ Acc.\\\\")
        rows = []
        b = R["baseline"]["test"]
        rows.append(f"Prior baseline & -- & {b['average_precision']:.3f} & 0.500 & -- & -- & 0.000 & 0.000 & "
                    f"{b['f1_macro']:.3f} & 0.500\\\\")
        for k in ids:
            m = R["models"][k]
            t, op = m["test"], m["test"]["at_tuned"]
            cvp = m["cv"][pm]
            rows.append(f"{tex_escape(m['label'])} & {cvp['mean']:.3f}$\\pm${cvp['std']:.3f} & "
                        f"{t['average_precision']:.3f} {ci(cis[k]['average_precision']) if k in cis else ''} & "
                        f"{t['roc_auc']:.3f} & {op['threshold']:.2f} & {op['precision']:.3f} & {op['recall']:.3f} & "
                        f"{op['f1']:.3f} & {op['f1_macro']:.3f} & {op['balanced_accuracy']:.3f}\\\\")
        cols = "l" + "c" * 9
        note = ("Precision/recall/F1 are for the positive class at threshold $t$, tuned on out-of-fold training "
                "predictions; AP = average precision (PR-AUC). CIs: 1000 test-set bootstrap resamples.")
    else:
        pm = R["cv"]["primary_metric"]
        head = (r"Model & CV " + tex_escape(pm) + r" (mean$\pm$sd) & Test Macro-F1 [95\% CI] & Bal.\ Acc. & Accuracy & ROC-AUC (OvR) & Log-loss\\")
        rows = []
        b = R["baseline"]["test"]
        rows.append(f"Prior baseline & -- & {b['f1_macro']:.3f} & {b['balanced_accuracy']:.3f} & {b['accuracy']:.3f} & "
                    f"{b['roc_auc_ovr']:.3f} & {b['log_loss']:.3f}\\\\")
        for k in ids:
            m, t = R["models"][k], R["models"][k]["test"]
            c = m["cv"][pm]
            rows.append(f"{tex_escape(m['label'])} & {c['mean']:.3f}$\\pm${c['std']:.3f} & {t['f1_macro']:.3f} "
                        f"{ci(cis[k]['f1_macro']) if k in cis else ''} & {t['balanced_accuracy']:.3f} & "
                        f"{t['accuracy']:.3f} & {t['roc_auc_ovr']:.3f} & {t['log_loss']:.3f}\\\\")
        cols = "l" + "c" * 6
        note = "Class predictions are the arg-max of predicted probabilities. CIs: 1000 test-set bootstrap resamples."
    return ("\\begin{table}[!htb]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\n"
            f"\\caption{{Cross-validated and held-out test performance. {note}}}\\label{{tab:results}}\n"
            f"\\begin{{tabular}}{{{cols}}}\\toprule\n{head}\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def space_str(spec) -> str:
    if isinstance(spec, list):
        return "$\\in\\{$" + ", ".join("none" if v is None else tex_escape(v) for v in spec) + "$\\}$"
    (kind, (lo, hi)), = spec.items()
    return f"[{lo:g}, {hi:g}]" + (" log" if kind == "loguniform" else "")


def config_table(R: dict) -> str:
    rows = []
    for k, m in R["models"].items():
        bp = ", ".join(f"{tex_escape(p)}={v:.3g}" if isinstance(v, float) else f"{tex_escape(p)}={tex_escape(v)}"
                       for p, v in m["best_params"].items())
        space = ", ".join(f"{tex_escape(p)} {space_str(v)}" for p, v in m["search_space"].items()) or "--"
        fit = m["fit"]
        stop = (f"{fit['n_iter']}/{fit['max_iter']} iter." if "n_iter" in fit and "max_iter" in fit else
                f"depth {fit['tree_depth']}" if "tree_depth" in fit else "--")
        bnd = (" (boundary: " + ", ".join(tex_escape(b) for b in m["params_at_search_boundary"]) + ")") \
            if m["params_at_search_boundary"] else ""
        rows.append(f"{tex_escape(m['label'])} & {m['n_candidates']} cand.\\ over {space} & {bp}{bnd} & {stop} & "
                    f"{m['n_features_after_preprocessing']} & {m['overfit_gap_primary']:+.3f}\\\\")
    return ("\\begin{table}[!htb]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Hyper-parameter search (randomised, stratified CV, refit on the primary metric) and fitted models. "
            "Gap = CV train $-$ validation primary metric (over-fitting indicator).}\\label{tab:config}\n"
            "\\begin{tabularx}{\\linewidth}{l Y Y c c c}\\toprule\n"
            "Model & Search & Selected & Stopping & \\#inputs & Gap\\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabularx}\\end{table}\n")


def ablation_table(R: dict) -> str:
    if not R.get("ablations"):
        return ""
    pm = tex_escape(R["cv"]["primary_metric"])
    rows = []
    for a in R["ablations"]:
        t = a["corrected_ttest"]
        p = "--" if t["p"] is None else ("$<$0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")
        rows.append(f"{tex_escape(a['id'])} & {tex_escape(a['question'])} & "
                    f"{'0.0000' if abs(a['delta']) < 5e-5 else format(a['delta'], '+.4f')} & {a['delta_fold_sd']:.4f} & {p}\\\\")
    return ("\\begin{table}[!htb]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            f"\\caption{{Ablations: one decision reversed, tuned hyper-parameters kept (so $\\Delta$ is conservative for the variant). "
            f"$\\Delta$ = change in CV {pm} on the same folds; sd = sd of paired fold differences ($|\\Delta|<$ sd: no detectable effect); "
            f"$p$: Nadeau--Bengio corrected $t$-test.}}"
            "\\label{tab:ablation}\n"
            "\\begin{tabularx}{\\linewidth}{l Y r r r}\\toprule\n"
            f"ID & Question & $\\Delta$ {pm} & sd & $p$\\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabularx}\\end{table}\n")


def issues_table(rows: list, resolve) -> str:
    body = "\n".join(f"{resolve(r[0])} & {resolve(r[1])} & {resolve(r[2])}\\\\" for r in rows)
    return ("\\begin{table}[!htb]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Data issues found by the profiler and the treatment chosen (all statistics learned inside CV folds).}"
            "\\label{tab:issues}\n"
            "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{2.6cm} Y Y}\\toprule\n"
            "Issue & Evidence & Treatment and rationale\\\\\\midrule\n" + body +
            "\n\\bottomrule\\end{tabularx}\\end{table}\n")


FIGURE = ("\\begin{figure}[!ht]\\centering\\includegraphics[width=\\linewidth]{fig_results.pdf}\n"
          "\\caption{%s}\\label{fig:results}\\end{figure}\n")

SECTIONS = [("data", "Data exploration and cleaning"), ("features", "Feature selection and engineering"),
            ("training", "Model training"), ("evaluation", "Evaluation and comparison"),
            ("findings", "Findings and discussion")]


def build(run: pathlib.Path) -> tuple[str, Resolver]:
    load = lambda n: json.loads((run / n).read_text())
    P, L, R, N = load("profile.json"), load("plan.json"), load("results.json"), load("narrative.json")
    res = Resolver({"P": P, "L": L, "R": R})
    meta = N["meta"]
    header = (
        "\\begin{center}\n"
        f"{{\\large\\bfseries {meta.get('title', 'Classification report')}}}\\\\[2pt]\n"  # title is LaTeX
        f"\\textbf{{Matric No.:}} {tex_escape(meta.get('matric', '<MATRIC NO>'))} \\quad "
        f"\\textbf{{Name:}} {tex_escape(meta.get('name', '<FULL NAME>'))} \\quad "
        f"\\textbf{{{tex_escape(meta.get('course_line', 'IN6227-Assignment-1'))}}} \\quad "
        f"\\textbf{{{tex_escape(meta.get('variant', 'Variant-2'))}}}\\\\[1pt]\n"
        "{\\footnotesize Generated by the SKILL \\texttt{tabular-classification-report} \\textbar{} "
        f"Model: \\texttt{{{tex_escape(meta['llm_model'])}}} \\textbar{{}} Interface: {tex_escape(meta['llm_interface'])} "
        f"\\textbar{{}} Code: \\url{{{meta.get('repo_url', '')}}}}}\n"
        "\\end{center}\n\\vspace{-4pt}\n")
    if N.get("abstract"):
        header += "{\\small " + res(N["abstract"]) + "}\n"
    blocks = {"issues_table": issues_table(N.get("issues_table", []), res) if N.get("issues_table") else "",
              "results_table": results_table(R), "config_table": config_table(R),
              "ablation_table": ablation_table(R),
              "figure": FIGURE % res(N.get("figure_caption", "Test-set results."))}
    body = []
    for i, (key, title) in enumerate(SECTIONS, 1):
        sec = N[key]
        text = res(sec if isinstance(sec, str) else sec["text"])
        # floats are placed where the narrative asks: %%issues_table%% etc.
        for name, tex in blocks.items():
            text = text.replace(f"%%{name}%%", tex)
        body.append(f"\\section{{{tex_escape(N.get('titles', {}).get(key, title))}}}\n{text}\n")
    reflection = ""
    if (run / "reflection.tex").exists():
        reflection = ("\\clearpage\n\\section*{Reflection {\\normalfont\\small(not counted toward the two-page limit)}}\n"
                      + res((run / "reflection.tex").read_text()))
    tex = TEMPLATE.read_text().replace("%%HEADER%%", header).replace("%%BODY%%", "\n".join(body)) \
        .replace("%%REFLECTION%%", reflection)
    return tex, res


def literal_numbers(run: pathlib.Path) -> list[str]:
    """Numbers typed directly into the narrative (not via tokens) - the reviewer must check each."""
    raw = (run / "narrative.json").read_text()
    if (run / "reflection.tex").exists():
        raw += (run / "reflection.tex").read_text()
    raw = TOKEN.sub(" ", raw)
    return sorted(set(m.group(0) for m in LITERAL_NUMBER.finditer(raw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    tex, res = build(run)
    (run / "report.tex").write_text(tex)
    audit = ["# Numbers audit", "", "## Token-resolved values (source of truth: JSON files)", "",
             "| token | value | rendered |", "|---|---|---|"]
    audit += [f"| `{t}` | {v} | {o} |" for t, v, o in res.used]
    audit += ["", "## Literal numbers typed in prose (verify by hand)", ""]
    audit += [f"- {x}" for x in literal_numbers(run)] or ["- none"]
    (run / "numbers_audit.md").write_text("\n".join(audit) + "\n")
    if res.missing:
        print("UNRESOLVED TOKENS:", *res.missing, sep="\n  ")
        sys.exit(2)
    if a.no_compile:
        return
    if not shutil.which("tectonic"):
        print("tectonic not found; report.tex written but not compiled")
        sys.exit(1)
    # 1) main report alone -> its page count; 2) full document (with reflection) -> report.pdf
    main_tex = tex.replace(tex[tex.index("\\label{endofreport}"):tex.index("\\end{document}")], "")
    (run / "report_main.tex").write_text(main_tex)
    pages = None
    for name in ("report_main.tex", "report.tex"):
        proc = subprocess.run(["tectonic", "-k", "--chatter", "minimal", name], cwd=run,
                              capture_output=True, text=True)
        if proc.returncode:
            print(proc.stdout[-3000:], proc.stderr[-3000:])
            sys.exit(1)
        if name == "report_main.tex":
            m = re.search(r"\\gdef\s*\\@abspage@last\{(\d+)\}", (run / "report_main.aux").read_text())
            pages = int(m.group(1)) if m else None
    warnings = [l for l in proc.stderr.splitlines() if "Overfull" in l or "undefined" in l.lower()]
    print(f"report.pdf written; main report has {pages} page(s) (limit {a.max_pages}); "
          f"{len(res.used)} tokens; {len(literal_numbers(run))} literal numbers to verify")
    for w in warnings[:10]:
        print("  latex:", w)
    if pages is None or pages > a.max_pages:
        print(f"PAGE LIMIT EXCEEDED: shorten the narrative and re-render")
        sys.exit(3)


if __name__ == "__main__":
    main()
