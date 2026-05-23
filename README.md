# MiB-SCFL and Federated Learning Baselines

This repository contains a Jupyter notebook for evaluating **MiB-SCFL ** against several federated learning baselines on a fixed 15-client non-IID partition.

The main notebook is:

```text
Our MiB-SCFL and all baselines.ipynb
```

## Overview

The notebook builds a reproducible federated learning experiment with:

- A fixed **15-client** partition.
- A non-IID data split controlled by a Dirichlet-style class allocation.
- Three label clusters for MNIST:
  - `C1_digits_0_1_2_3`
  - `C2_digits_4_5_6_7`
  - `C3_digits_8_9`
- Standard CNN-based MNIST models for baseline methods.
- Communication and computation reporting, including parameter and FLOP-based cost estimates.
- CSV exports for round-level metrics, per-class metrics, cluster metrics, and communication-cost summaries.

The notebook also includes a HAR-oriented MiB-SCFL cell using activity clusters:

- `C1_dynamic_walking_upstairs_downstairs`
- `C2_static_sitting_standing`
- `C3_laying`

## Implemented Methods

The notebook includes the following methods:

| Method | Description |
|---|---|
| FedAvg | Standard federated averaging baseline. |
| FedProx | FedAvg with proximal regularization. |
| CFL | Clustered Federated Learning baseline using client update similarity. |
| Bayesian-CFL | Soft / probabilistic clustered FL variant. |
| FedMTL-style | Multi-task-learning-style baseline utilities. |
| FedRFC-style | Recursive fuzzy clustering baseline. |
| FedSoft-style | Soft clustered FL baseline. |
| FedSPD-style | Soft personalized / distribution-style baseline. |
| Multi-Center FL | Multi-center personalized FL baseline. |
| SoftCluster-FL | Soft cluster federated learning baseline. |
| pFedCAM-style | Personalized cluster assignment / mixture baseline. |
| MiB-SCFL  | MNIST resource-bounded clustered FL implementation. |


> Note: the pFedCAM-style cell appears twice in the notebook. You can usually run only one copy unless you intentionally want to compare duplicated runs.

## Requirements

Install the core Python dependencies:

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn jupyter
```

Recommended environment:

- Python 3.9+
- PyTorch with CUDA support, if GPU training is desired
- Jupyter Notebook or JupyterLab

For exact CPU-based reproducibility, several cells expose:

```python
FULL_DETERMINISM = True
USE_CPU_FOR_EXACT_REPRODUCIBILITY = True
```

GPU execution is faster but may introduce small nondeterministic differences depending on CUDA/cuDNN behavior.

## Dataset Setup

### MNIST

MNIST is loaded through `torchvision.datasets.MNIST`. The dataset will be downloaded automatically if it is not already available.

The fixed split is saved as:

```text
fixed_mnist_15client_cluster_split_seed7.npz
```

By default, the notebook reuses this file if it exists:

```python
REBUILD_FIXED_SPLIT = False
```

Set it to `True` to rebuild the fixed client partition.


## Experiment Configuration

The common MNIST configuration is:

```python
SEED = 3
NUM_CLIENTS = 15
NUM_CLASSES = 10
VAL_RATIO = 0.1
NONIID_ALPHA = 0.1
ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
CLIENT_PARTICIPATION_RATE = 1.0
BASE_LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 0.001
```

The proposed resource-bounded method uses global, local, and exploration subnetworks with ratios similar to:

```python
GLOBAL_RATIO = 0.50
LOCAL_RATIO = 0.40
EXPLORE_RATIO = 0.10
CLIENT_TOTAL_WIDTH_BUDGET = 1.0   # MNIST version
BUDGET_GAMMA = 0.5
MIN_WIDTH = 1
```

```python
CLIENT_TOTAL_WIDTH_BUDGET = 0.99
```

## How to Run

1. Open the notebook:

   ```bash
   jupyter notebook "Our MiB-SCFL and all baselines.ipynb"
   ```

2. Run the first cell to create/load the fixed client split and shared utilities.

3. Run the baseline cells you want to evaluate.

4. Run the Resource-Bounded Cluster FL / MiB-SCFL cells after the shared setup cell has completed.

5. Review generated tables and saved CSV files in the notebook output and local working directory.

Suggested execution order:

```text
Cell 0: shared split, metrics, export utilities
Cell 1: FedAvg
Cell 2: FedProx
Cell 3: CFL
Cell 4: Bayesian-CFL
Cell 6: FedRFC-style
Cell 7: FedSoft-style
Cell 8: FedSPD-style
Cell 9: Multi-Center FL
Cell 10: SoftCluster-FL
Cell 11 or 12: pFedCAM-style
Cell 13:MiB-SCFL on MNIST

```

Cell 5 contains FedMTL-style utilities, but the inspected notebook does not show a final `run_*` call for that cell.

## Outputs

The shared export function saves baseline results under a `run/` directory using method-specific filenames. Typical outputs include:

```text
run/<method>_metrics.csv
run/<method>_per_digit_metrics.csv
run/<method>_cluster_precision_recall_f1.csv
run/<method>_<extra_table>.csv
```

The resource-bounded MNIST experiment also saves:

```text
resource_bounded_cluster_fl_round_log.csv
resource_bounded_cluster_fl_participation.csv
resource_bounded_cluster_fl_curves_shaded.png
```

The HAR resource-bounded experiment saves analogous HAR outputs, including round logs, participation logs, and shaded learning-curve figures.

## Metrics Reported

The notebook reports several evaluation and cost metrics, including:

- Test loss
- Test accuracy
- Precision, recall, and F1-score
- Per-digit metrics
- Cluster-level precision, recall, and F1-score
- Best validation-selected round
- Parameter counts
- FLOP estimates
- Communication-cost columns
- Client participation and cluster-level results for selected methods

## Reproducibility Notes

The notebook sets seeds for Python, NumPy, and PyTorch. For the most deterministic results:

```python
FULL_DETERMINISM = True
USE_CPU_FOR_EXACT_REPRODUCIBILITY = True
```

Some cells default to faster GPU-friendly settings:

```python
FULL_DETERMINISM = False
USE_CPU_FOR_EXACT_REPRODUCIBILITY = False
```

When comparing methods, use the same fixed split file and keep the experiment settings aligned across runs.

## Project Structure

A minimal project layout is:

```text
.
├── Our MiB-SCFL and all baselines.ipynb
├── README.md
├── fixed_mnist_15client_cluster_split_seed7.npz   # generated/reused split file
├── run/                                           # exported baseline CSVs
├── resource_bounded_cluster_fl_round_log.csv
├── resource_bounded_cluster_fl_participation.csv
└── resource_bounded_cluster_fl_curves_shaded.png
```
