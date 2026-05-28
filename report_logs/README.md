# Report Logs

SLURM logs for every run referenced in the project report. Each filename's
final `validation loss at iteration ...` line is what we cite in the report.

## 30 min — 4-node baselines (`30min_4nodes/`)

| Run    | Source log                                | Final val loss |
|--------|-------------------------------------------|----------------|
| 125M   | `gipfel-train-125m-1800s-4n-2345282.log`  | 2.34 |
| 350M   | `gipfel-train-350m-1550s-4n-2345280.log`  | 2.34 |
| 760M   | `gipfel-train-760m-1800s-4n-2345281.log`  | 2.30 |

## 30 min — 760M learning-rate sweep on 8 nodes (`lr_sweep_760m_8nodes/`)

All four runs use the same 1819-step (cosine) schedule; report numbers come from the in-training validation at the 30-min mark.

| LR     | Source log                                          | Final val loss |
|--------|-----------------------------------------------------|----------------|
| 1e-4   | `gipfel-train-760m-1819s-8n-lr1e-4-2366458.log`     | 2.36 |
| 2e-4   | `gipfel-train-760m-1819s-8n-2355608.log` (default)  | 2.21 |
| 3e-4   | `gipfel-train-760m-1819s-8n-lr3e-4-2366459.log`     | 2.14 |
| 4e-4   | `gipfel-train-760m-1819s-8n-lr4e-4-2366460.log` ⭐  | 2.09 |

## 1 hr — 8 nodes (`1hr_8nodes/`)

| Model          | Source log                                  | Final val loss |
|----------------|---------------------------------------------|----------------|
| 760M (best) ⭐ | `gipfel-train-760m-3638s-8n-2366461.log`    | 1.99 |
| 1.5B           | `gipfel-train-1.5b-2293s-8n-2366463.log`    | 2.10 |
| 3B             | `gipfel-train-3b-2819s-8n-2366462.log`      | 2.08 |

## 2 hr — 8 nodes (`2hr_8nodes/`)

| Model          | Source log                                  | Final val loss |
|----------------|---------------------------------------------|----------------|
| 760M (best) ⭐ | `gipfel-train-760m-7277s-8n-2366465.log`    | 1.83 |
| 1.5B           | `gipfel-train-1.5b-4587s-8n-2355658.log`    | 1.91 |
| 3B             | `gipfel-train-3b-5639s-8n-2355614.log`      | 1.90 |
| 8B             | `gipfel-train-8b-12298s-8n-2366466.log`     | 1.98 |
