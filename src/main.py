"""
main.py
CLI entrypoint for delta-chat.
Usage:
  python -m src.main ingest  --pid-a path/a.pdf --pid-b path/b.pdf
  python -m src.main chat    --question "what changed on page 2?"
  python -m src.main run-all --pid-a path/a.pdf --pid-b path/b.pdf
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

load_dotenv()
console = Console()


@click.group()
def cli():
    """delta-chat: Document Delta & Grounded Chat"""


@cli.command("ingest")
@click.option("--pid-a", required=True, type=click.Path(exists=True), help="Path to document A (base)")
@click.option("--pid-b", required=True, type=click.Path(exists=True), help="Path to document B (revised)")
@click.option("--rev-a", default="A", help="Revision label for doc A")
@click.option("--rev-b", default="B", help="Revision label for doc B")
@click.option("--scanned-b", is_flag=True, default=False, help="Treat doc B as scanned PDF (use OCR)")
@click.option("--output-dir", default="reports", help="Output directory for reports")
def ingest_cmd(pid_a, pid_b, rev_a, rev_b, scanned_b, output_dir):
    """Ingest two documents, compute delta, write report, build chat index."""
    from src.observability.tracer import start_trace, finish_trace, trace_stage
    from src.observability.logger import log, new_request_id, set_request_id
    from src.ingest.base import AdapterRegistry
    from src.ingest.pdf_native import NativePDFAdapter
    from src.ingest.pdf_scanned import ScannedPDFAdapter
    from src.ingest.dwg import DWGAdapter
    from src.delta import engine as delta_engine
    from src.delta.report import render as render_report
    from src.chat.index import build_index

    rid = new_request_id()
    trace = start_trace("ingest_and_delta", {"pid_a": pid_a, "pid_b": pid_b})
    set_request_id(rid)

    registry = AdapterRegistry()
    registry.register(NativePDFAdapter())
    registry.register(DWGAdapter())

    try:
        console.rule("[bold blue]Ingesting documents")

        with trace_stage("ingest_a"):
            path_a = Path(pid_a)
            doc_a  = registry.ingest(path_a, pid=path_a.stem, revision=rev_a)
            console.print(f"[green]OK Doc A:[/] {doc_a.pid} - {len(doc_a.pages)} pages, "
                          f"{len(doc_a.all_blocks())} blocks ({doc_a.format})")

        with trace_stage("ingest_b"):
            path_b = Path(pid_b)
            if scanned_b or _looks_scanned(path_b):
                adapter_b = ScannedPDFAdapter()
                doc_b = adapter_b.ingest(path_b, pid=path_b.stem, revision=rev_b)
            else:
                doc_b = registry.ingest(path_b, pid=path_b.stem, revision=rev_b)
            console.print(f"[green]OK Doc B:[/] {doc_b.pid} - {len(doc_b.pages)} pages, "
                          f"{len(doc_b.all_blocks())} blocks ({doc_b.format})")

        console.rule("[bold blue]Running delta engine")
        with trace_stage("delta"):
            items = delta_engine.run(doc_a, doc_b)

        # Print summary table
        tbl = Table(title=f"Delta Summary ({len(items)} changes)")
        tbl.add_column("Type");  tbl.add_column("Count", justify="right")
        for ct in ("added", "removed", "modified"):
            n = sum(1 for i in items if i.change_type == ct)
            tbl.add_row(ct, str(n))
        console.print(tbl)

        with trace_stage("report"):
            md_path, json_path = render_report(items, doc_a, doc_b, run_id=rid)
            console.print(f"[green]OK Report:[/] {md_path}")
            console.print(f"[green]OK JSON:  [/] {json_path}")

        with trace_stage("index"):
            build_index(doc_a, doc_b, items)
            console.print("[green]OK Chat index built[/]")

        # Cache doc references for chat session
        _save_session(doc_a.pid, doc_b.pid, str(json_path))

        trace_path = finish_trace()
        console.print(f"[dim]Trace: {trace_path}[/dim]")
        console.rule("[bold green]Done - run: .venv\\Scripts\\python -m src.main chat")

    except Exception as e:
        finish_trace(error=str(e))
        console.print(f"[red]ERROR:[/] {e}")
        raise SystemExit(1)


@cli.command("chat")
@click.option("--question", "-q", default=None, help="Question to ask (if not provided, enters interactive mode)")
@click.option("--n-results", default=6, help="Number of retrieved chunks")
def chat_cmd(question, n_results):
    """Grounded chat over ingested documents."""
    from src.chat.answer import answer as get_answer
    from src.observability.tracer import start_trace, finish_trace, trace_stage
    from src.observability.logger import new_request_id

    def _ask(q: str):
        rid = new_request_id()
        start_trace("chat", {"question": q[:80]})
        try:
            with trace_stage("chat_full"):
                result = get_answer(q, n_results=n_results)
            finish_trace()
            console.print(Panel(result.answer, title="[bold green]Answer", border_style="green"))
            console.print("\n[bold]Citations:[/]")
            for cit in result.citations:
                console.print(f"  • [{cit.ref}] {cit.text[:80]}...")
        except Exception as e:
            finish_trace(error=str(e))
            console.print(f"[red]Error:[/] {e}")

    if question:
        _ask(question)
    else:
        console.print("[bold blue]Interactive chat mode. Type 'exit' to quit.[/]")
        while True:
            try:
                q = console.input("\n[bold]Question > [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                _ask(q)


@cli.command("run-all")
@click.option("--pid-a", required=True, type=click.Path(exists=True))
@click.option("--pid-b", required=True, type=click.Path(exists=True))
@click.option("--question", default="What changed between the two documents?")
@click.option("--scanned-b", is_flag=True, default=False)
def run_all_cmd(pid_a, pid_b, question, scanned_b):
    """Single command: ingest → delta → report → chat answer."""
    from click.testing import CliRunner
    runner = CliRunner(mix_stderr=False)
    args = ["--pid-a", pid_a, "--pid-b", pid_b]
    if scanned_b:
        args.append("--scanned-b")
    result = runner.invoke(ingest_cmd, args, catch_exceptions=False)
    console.print(result.output)
    result2 = runner.invoke(chat_cmd, ["--question", question], catch_exceptions=False)
    console.print(result2.output)


def _looks_scanned(path: Path) -> bool:
    """Heuristic: if native PDF has very few text chars per page, treat as scanned."""
    try:
        import fitz
        doc = fitz.open(str(path))
        total_chars = sum(len(p.get_text()) for p in doc)
        pages = max(len(doc), 1)
        doc.close()
        return (total_chars / pages) < 50
    except Exception:
        return False


def _save_session(pid_a: str, pid_b: str, report_json: str):
    import json
    Path(".session.json").write_text(
        json.dumps({"pid_a": pid_a, "pid_b": pid_b, "report_json": report_json})
    )


if __name__ == "__main__":
    cli()
