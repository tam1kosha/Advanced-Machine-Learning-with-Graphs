# Training on the GPU server

The interface runs independent first-k experiments on allocated GPUs. It does not split one model
across eight cards. Each process keeps its own historical graph in host RAM; only sampled batches
and the neural network go to the GPU. Eight cards do not provide one combined VRAM pool.

## Fresh Linux installation

Clone the branch containing this server audit, then enter the repository. Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **CUDA-enabled PyTorch first**, using the Linux / Pip / Python command from the
[official selector](https://pytorch.org/get-started/locally/) for the installed NVIDIA driver.
For example, if that driver supports the CUDA 12.8 wheel build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
python -m pip check
nvidia-smi
bash run_server.sh check
bash run_server.sh smoke
python -m pytest -q tests
python -m pip freeze > outputs/logs/server-requirements.txt
```

The CUDA wheel index is an example, not a claim about the driver on your server. The preflight
checks the CUDA build, visible cards, and an actual GPU operation/backward pass. The smoke test
trains homogeneous and heterogeneous GraphSAGE with binary and regression targets on **each selected
GPU**, reloads saved checkpoints, and fits XGBoost in its separate worker. It needs no trade data.
These GraphSAGE implementations need no compiled PyG extensions; basic `torch_geometric` is enough
([PyG installation documentation](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html)).

If XGBoost cannot load `libgomp.so.1` on Ubuntu/Debian, install the distribution's `libgomp1` package.
On macOS, the equivalent is `brew install libomp`. Missing native libraries are caught before downloads.
CUDA absence is an error for GPU stages; `PMGRAPH_ALLOW_CPU=1` explicitly enables CPU debugging.

## Prepare, pilot, then scale out

Use `tmux` or the server's job scheduler so disconnecting SSH does not terminate training. Do not
start two launchers using the same tag and output files at the same time.

```bash
# CPU/data stages. About 8 GB of raw download; allow substantial extra space for derived data/spills.
bash run_server.sh data prep diag

# One real experiment before committing all eight GPUs.
CUDA_VISIBLE_DEVICES=0 NGPU=1 SPLITS=train KS=1 PMGRAPH_TAG=pilot bash run_server.sh edge

# Check logs and peak CPU RAM / VRAM from the pilot, then run the full comparison.
NGPU=8 bash run_server.sh edge oot

# Optional: four predeclared GNN settings on the earlier period.
NGPU=8 bash run_server.sh grid
```

Alternatively, `bash run_server.sh` runs `data prep diag edge oot` in order, with all visible GPUs.
The data/prep/diag stages are not automatically skipped on restart. Once they complete, restart at
`edge oot`, not at the download stage. A grid rerun must report every setting; choosing a winner
using the CV test folds and reporting its score as unbiased would be invalid.

Resource planning: measure the **process peak RSS** of a real pilot and budget at least the number
of concurrent workers times that amount, plus OS/other jobs. The old train graph alone was about
3 GB per process; the later graph and larger sampled batches may need more. GPUs cannot compensate
for exhausted host RAM. Use `NGPU=2` or `NGPU=4` when host RAM or CPU sampling is the bottleneck.
Disk needs raw data, processed graphs, predictions, checkpoints, and DuckDB spill space; a generous
working allowance is 100 GB free (an allowance, not a measured requirement).

Allocation from a scheduler is preserved. For example:

```bash
CUDA_VISIBLE_DEVICES=2,5 NGPU=2 bash run_server.sh edge
```

Each worker sees exactly one of those cards as `cuda:0`. GPU UUID allocations work too. Do not
override `device` with a physical index in `PMGRAPH_GNN`.

## Options and outputs

| Setting | Default | Use |
|---|---|---|
| `PYTHON` | `python` | Active environment's interpreter, or an explicit path |
| `NGPU` | All visible cards | Maximum parallel workers; cannot exceed allocation |
| `SPLITS` | `train test` | Space-separated splits for prep, diagnostics, and edge; grid is train-only and oot always needs both |
| `KS` | `1 3 5 10 20 50` | Space-separated prepared k values; subsets are pilots, not the full graph claim |
| `METHODS` | `hist,xgb_own,xgb_nbr,sage,hgnn` | Comma-separated methods; retain both baselines for the full comparison |
| `PMGRAPH_TAG` | `server` | Prediction/checkpoint/report suffix; isolates new graph runs from committed historical results |
| `PMGRAPH_GNN` | Config | JSON overrides: e.g. `'{"hidden":128,"batch":1024,"eval_batch":512}'` |
| `PMGRAPH_RESUME` | `1` | Skip completed k × method jobs only when code/config/input identities match |

`KS` changes which experiments run; preparation still builds all configured k values. Grid profiles
replace `PMGRAPH_GNN` with their declared overrides. Default inference batch size follows training
batch size; set `eval_batch` lower when inference runs out of VRAM. Profiling defaults are unchanged
except this safer inference batching, which can change sampled predictions. New server scores are
tagged separately. If you change settings under an existing tag, jobs are retrained rather than
silently reused; prefer a new tag to retain both settings.

Outputs for a default server run:

- `outputs/logs/preflight_*.json`: interpreter/package versions, GPU names/VRAM, CPU count,
  Linux host RAM, disk space, commit, and selected options.
- `outputs/logs/09_edge_server_*.log`: training progress and per-fit resource records.
- `data/processed/checkpoints/exp4_server/*_fold*.pt`: best-validation GNN state, model dimensions,
  objective, target normalization, and fit metadata. These are inference weights, not optimizer/RNG
  snapshots for resuming halfway through an epoch.
- Matching checkpoint `.json`: parameter count, device, epochs/best epoch, validation score,
  training-plus-evaluation elapsed seconds, process-lifetime peak RSS, peak PyTorch CUDA allocated
  and reserved bytes. CUDA counters exclude unrelated processes and driver overhead.
- `*_preprocessing.pt`: feature order and fitted node/edge scalers. Keep the corresponding prepared
  graph/query data and source code as well; a checkpoint is not a standalone wallet-to-score service.
- `data/processed/exp4_preds_server/`: per-wallet predictions and completed-job manifests.
- `outputs/tables/exp4_{train,test}_graph_server*.csv` and corresponding figures: graph comparison.
- `outputs/tables/exp5_train_to_test_oot.csv`: out-of-time XGBoost results.

Resume granularity is a completed k × method (all five folds). An interrupted method restarts its
folds. Completed jobs are invalidated by source/config changes, changed input size/mtime, or missing
checkpoints. A report rejects missing requested methods/k/folds. A pilot cannot satisfy the full claim.
The full-data preparation, diagnostics and out-of-time outputs retain their established filenames;
only the graph comparison and GNN artifacts are tag-isolated. Old committed outputs were not changed
by this audit.

Keep `data/processed/checkpoints/` on persistent storage if you want the models. Git ignores `data/`.
Commit the new result tables, figures and a dated NOTES entry after the real server run. Preserve the
requirements snapshot and logs alongside model artifacts for reproducibility.

## Scope of the audit and scientific limitations

The audit runs CPU tests and small prepared-data integration tests, not the full downloaded dataset
or your remote GPUs. The server smoke and real-data pilot are still necessary. PyTorch/PyG package
versions are recorded, not locked across every Linux/CUDA combination.

Two layers with hidden size 64 are a sensible starting capacity for tens of thousands of labelled
wallets; the existing width-128 / larger-fanout grid gives a useful capacity check. A large graph does
not require a parameter per wallet: GraphSAGE shares its transformations across nodes
([original paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)).
Use independent GPUs for periods/k/settings rather than increasing the network to fill VRAM.

Before treating a null graph result as final, address coverage and scientific controls: many new
wallets trade in markets without frozen graph nodes; a self-only MLP ablation would distinguish graph
information from neural-network versus tree differences; trees fit on train+validation wallets while
GNNs reserve validation for stopping. CPU sampling can dominate runtime despite idle GPUs.

There are also existing temporal-validity questions outside this interface change: `hours_to_close`
uses the realised `close_time`, and label windows restrict trade times without enforcing that every
label market resolved before the end of the window. Thus script 15's claim that *all training labels
are resolved before the scoring period* is not enforced by the preparation code. Audit/replace the
time feature with an as-of observable deadline and explicitly define/enforce resolution availability
before making a strict deployment/out-of-time claim. This audit does not silently redefine historical
features/labels or rewrite their result tables.
