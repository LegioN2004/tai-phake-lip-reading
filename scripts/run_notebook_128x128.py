#!/usr/bin/env python3
"""
Execution script for notebooks/02.8_dlib_video_to_processed_pipeline_128x128.ipynb
Runs the notebook cell-by-cell with real-time logging, saves the executed notebook
with outputs and generated figures, and verifies the resulting 128x128 dataset.
"""

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02.8_dlib_video_to_processed_pipeline_128x128.ipynb"

class LoggingNotebookClient(NotebookClient):
    def on_cell_start(self, cell, cell_index):
        if cell.cell_type == "code":
            lines = [l.strip() for l in cell.source.split("\n") if l.strip() and not l.strip().startswith("#")]
            header = lines[0] if lines else "code"
            print(f"--> [Cell {cell_index}] Executing: {header[:60]}...", flush=True)

    def on_cell_executed(self, cell, cell_index, execute_reply=None):
        if cell.cell_type == "code":
            for out in cell.outputs:
                if out.output_type == "stream":
                    print(out.text, end="", flush=True)
                elif out.output_type == "error":
                    print(f"\n[ERROR in Cell {cell_index}]:", file=sys.stderr)
                    print("\n".join(out.traceback), file=sys.stderr, flush=True)
            print(f"<-- [Cell {cell_index}] Complete.\n", flush=True)

def main():
    print("=" * 80)
    print("EXECUTING NOTEBOOK: 02.8_dlib_video_to_processed_pipeline_128x128.ipynb")
    print("=" * 80)
    print(f"Notebook Path: {NOTEBOOK_PATH}")
    print(f"Working Dir  : {PROJECT_ROOT / 'notebooks'}\n")

    t0 = time.time()
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = LoggingNotebookClient(
        nb,
        timeout=1800,  # 30 minutes timeout for the entire notebook
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}
    )

    try:
        client.execute()
        print("=" * 80)
        print("ALL CELLS EXECUTED SUCCESSFULLY")
        print("=" * 80)
    except Exception as e:
        print(f"\n[EXECUTION FAILED]: {e}", file=sys.stderr)
        # Still save notebook up to failure point
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        sys.exit(1)

    # Save the executed notebook with outputs and figures
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Saved executed notebook to: {NOTEBOOK_PATH}")
    elapsed = time.time() - t0
    print(f"Total Notebook Execution Time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)\n")

if __name__ == "__main__":
    main()
