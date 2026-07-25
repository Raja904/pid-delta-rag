"""
eval/run_eval.py
Evaluation harness. Prints a scorecard.
Usage: python eval/run_eval.py
  OR:  make eval
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from eval.metrics import (
    delta_precision_recall_f1,
    groundedness_score_heuristic,
)
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_ground_truth(pair_dir: Path) -> dict:
    gt_file = pair_dir / "ground_truth.json"
    if not gt_file.exists():
        return {}
    return json.loads(gt_file.read_text())


def eval_delta_pair(pair_dir: Path) -> dict | None:
    gt = load_ground_truth(pair_dir)
    if not gt:
        console.print(f"[yellow]Skipping {pair_dir.name}: no ground_truth.json[/]")
        return None

    doc_a_path = pair_dir / gt.get("doc_a", "doc_a.pdf")
    doc_b_path = pair_dir / gt.get("doc_b", "doc_b.pdf")

    if not doc_a_path.exists() or not doc_b_path.exists():
        console.print(f"[yellow]Skipping {pair_dir.name}: documents not found[/]")
        return None

    console.print(f"[blue]Evaluating pair:[/] {pair_dir.name}")

    from src.ingest.base import AdapterRegistry
    from src.ingest.pdf_native import NativePDFAdapter
    from src.delta import engine as delta_engine

    registry = AdapterRegistry()
    registry.register(NativePDFAdapter())

    try:
        doc_a = registry.ingest(doc_a_path, pid=doc_a_path.stem, revision="A")
        doc_b = registry.ingest(doc_b_path, pid=doc_b_path.stem, revision="B")
        predicted = delta_engine.run(doc_a, doc_b)
    except Exception as e:
        console.print(f"[red]Error running delta: {e}[/]")
        return None

    expected_changes = gt.get("expected_changes", [])
    p, r, f1 = delta_precision_recall_f1(
        predicted=[{"change_type": i.change_type, "content_type": i.content_type,
                    "old_value": i.old_value or "", "new_value": i.new_value or ""}
                   for i in predicted],
        expected=expected_changes,
    )
    return {
        "pair":      pair_dir.name,
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "predicted": len(predicted),
        "expected":  len(expected_changes),
        "failures":  gt.get("known_failures", []),
    }


def eval_chat_pair(pair_dir: Path) -> list[dict]:
    qa_file = pair_dir / "qa_pairs.json"
    if not qa_file.exists():
        return []

    qa_pairs = json.loads(qa_file.read_text())
    results = []

    from src.chat.answer import answer as get_answer

    for qa in qa_pairs:
        question = qa["question"]
        expected = qa["expected_answer_keywords"]
        try:
            result   = get_answer(question)
            score    = groundedness_score_heuristic(result.answer, result.citations, expected)
            results.append({
                "question":    question,
                "score":       score,
                "has_citations": len(result.citations) > 0,
                "answer_preview": result.answer[:120],
            })
        except Exception as e:
            results.append({
                "question": question,
                "score": 0.0,
                "has_citations": False,
                "error": str(e),
            })
    return results


def main():
    console.rule("[bold]delta-chat Evaluation Scorecard")

    pair_dirs = [d for d in DATASETS_DIR.iterdir() if d.is_dir()]
    if not pair_dirs:
        console.print("[yellow]No eval pairs found in eval/datasets/. "
                      "Add document pairs with ground_truth.json to run eval.[/]")
        _print_synthetic_demo()
        return

    delta_results = []
    for pair_dir in sorted(pair_dirs):
        result = eval_delta_pair(pair_dir)
        if result:
            delta_results.append(result)

    # Delta scorecard
    console.print()
    tbl = Table(title="Delta Quality (Precision / Recall / F1)", box=box.ROUNDED)
    tbl.add_column("Pair"); tbl.add_column("P", justify="right")
    tbl.add_column("R", justify="right"); tbl.add_column("F1", justify="right")
    tbl.add_column("Predicted", justify="right"); tbl.add_column("Expected", justify="right")

    for r in delta_results:
        color = "green" if r["f1"] >= 0.6 else ("yellow" if r["f1"] >= 0.3 else "red")
        tbl.add_row(
            r["pair"],
            f"[{color}]{r['precision']:.2f}[/]",
            f"[{color}]{r['recall']:.2f}[/]",
            f"[{color}]{r['f1']:.2f}[/]",
            str(r["predicted"]),
            str(r["expected"]),
        )
    console.print(tbl)

    # Failure table
    for r in delta_results:
        if r.get("failures"):
            console.print(f"\n[bold red]Known failures for {r['pair']}:[/]")
            for f in r["failures"]:
                console.print(f"  • {f}")

    # Chat eval (only if index exists)
    chroma_dir = Path(".chroma_db")
    if chroma_dir.exists():
        console.print()
        chat_results = []
        for pair_dir in sorted(pair_dirs):
            chat_results += eval_chat_pair(pair_dir)

        if chat_results:
            chat_tbl = Table(title="Chat Quality (Groundedness / Correctness)", box=box.ROUNDED)
            chat_tbl.add_column("Question"); chat_tbl.add_column("Score", justify="right")
            chat_tbl.add_column("Citations?", justify="center")
            for r in chat_results:
                score = r.get("score", 0.0)
                color = "green" if score >= 0.6 else ("yellow" if score >= 0.3 else "red")
                chat_tbl.add_row(
                    r["question"][:60] + "..." if len(r["question"])>60 else r["question"],
                    f"[{color}]{score:.2f}[/]",
                    "✅" if r.get("has_citations") else "❌",
                )
            console.print(chat_tbl)

    avg_f1 = sum(r["f1"] for r in delta_results) / max(len(delta_results),1)
    console.print(f"\n[bold]Overall delta F1:[/] {avg_f1:.2f}")
    console.rule("[bold green]Scorecard complete")


def _print_synthetic_demo():
    """Print demo scorecard when no real pairs exist yet."""
    console.print("\n[dim]Demo run with synthetic results (no real eval pairs found):[/dim]")
    tbl = Table(title="Demo Scorecard", box=box.ROUNDED)
    tbl.add_column("Pair"); tbl.add_column("P"); tbl.add_column("R"); tbl.add_column("F1")
    tbl.add_row("pair_01 (synthetic)", "0.80", "0.75", "0.77")
    tbl.add_row("pair_02 (synthetic)", "0.65", "0.70", "0.67")
    console.print(tbl)
    console.print("[yellow]Add eval/datasets/pair_XX/ground_truth.json to run real eval.[/]")


if __name__ == "__main__":
    main()
