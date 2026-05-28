"""Export every run in the W&B `gipfelsturm` project as CSV + config JSON.

Usage:
    pip install wandb pandas
    export WANDB_API_KEY=<your-key>
    python scripts/export_wandb.py <entity>           # e.g. "course_00179" or your team
    python scripts/export_wandb.py <entity> <project> # default project: gipfelsturm
"""
import json
import sys
from pathlib import Path

import wandb

entity = sys.argv[1]
project = sys.argv[2] if len(sys.argv) > 2 else "gipfelsturm"
out = Path("wandb_exports")
out.mkdir(exist_ok=True)

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")
print(f"Found {len(runs)} runs in {entity}/{project}")

for run in runs:
    safe = run.name.replace("/", "_")
    rundir = out / f"{safe}__{run.id}"
    rundir.mkdir(exist_ok=True)

    run.history(samples=100000, pandas=True).to_csv(rundir / "history.csv", index=False)

    (rundir / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str))
    (rundir / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, default=str))
    (rundir / "meta.json").write_text(json.dumps({
        "id": run.id, "name": run.name, "state": run.state,
        "created_at": str(run.created_at), "url": run.url,
        "tags": run.tags,
    }, indent=2))
    print(f"  {run.name} -> {rundir}")

print(f"\nDone. {len(runs)} runs exported to {out}/")
