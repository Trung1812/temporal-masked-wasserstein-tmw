# Temporal Masked Wasserstein (TMW) Project

This repository implements the Temporal Masked Wasserstein (TMW) distance and related optimal transport (OT) solvers for time series analysis, benchmarking, and hyperparameter tuning.

## Project Structure

```
tmw_project/
├── data/
│   ├── raw/
│   └── processed/          # Z-normalized, train/test splits (one folder per dataset)
│
├── src/
│   ├── tmw/                # Main TMW implementation
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Z-normalization, loading, splitting
│   │   ├── mask.py             # Build temporal mask matrices for window w
│   │   ├── sinkhorn.py         # Entropic OT solver
│   │   ├── network_simplex.py  # Exact OT solver via LEMON or OR-tools
│   │   └── evaluation.py       # 1-NN classifier, accuracy, timing, Optuna tuning
│   │
│   ├── benchmarks/         # Baseline OT/DTW methods
│   │   ├── dtw.py
│   │   ├── otw.py
│   │   ├── pow.py
│   │   ├── opw.py
│   │   └── taot.py
│   │
│   └── utils.py             # Logging, config parsing, plotting helpers
│
├── experiments/
│   ├── configs/             # Experiment configs (YAML)
│   │   ├── base.yaml        # Dataset paths, hardware specs
│   │
│   ├── logs/                # Stdout/stderr from runs, Optuna logs
│   └── results/             # CSVs of (w,λ)→accuracy, timing, plots
│
├── notebooks/
│   ├── 01_explore_datasets.ipynb
│   ├── 02_parameter_tuning.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                   # Unit tests
│   ├── test_sinkhorn.py
│   └── test_network_simplex.py
│
├── requirements.txt         # Python dependencies (NumPy, SciPy, pandas, matplotlib, POT, python-lemon-binding, etc.)
├── setup.py                 # Installable package setup
├── scripts/                 # Shell scripts for running experiments and setup
│   ├── run_all_tuning.sh    # Run tuning for all datasets
│   ├── setup_google_colab.sh
│   └── setup_kaggle.sh
└── README.md
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Trung1812/temporal-masked-wasserstein-tmw.git
   cd temporal-masked-wasserstein-tmw
   ```
2. **Install dependencies:**

### With conda (Recommended)
    ```bash
    conda env create -f environment.yml
    conda activate viettel
    ```
### With pip
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Running Experiments

### 1. Hyperparameter Tuning & Evaluation (All Datasets)

Use the provided script to run Optuna-based tuning and evaluation for all datasets in `data/processed/`:

```sh
bash scripts/run_all_tuning.sh
```
- Results and logs will be saved in `logs/<DatasetName>/`.
- You can adjust the number of trials and datasets in the script.

### 2. Single Dataset Example

To run tuning and evaluation for a single dataset (e.g., `BeetleFly`):

```sh
python src/tmw/evaluation.py \
  --dataset_dir data/processed/BeetleFly \
  --n_trials 50 \
  --log_dir logs
```

### 3. Jupyter Notebooks

Explore and analyze results using the notebooks in the `notebooks/` folder.

## Configuration

- Edit YAML files in `experiments/configs/` to control dataset paths, hardware, and parameter search ranges.
- Example: `experiments/configs/base.yaml` for global paths and hardware, `tuning_lambda.yaml` for lambda search space.

## Testing

Run all unit tests with:
```sh
pytest tests/
```

## Google Colab / Kaggle Setup

- Use `scripts/setup_google_colab.sh` or `scripts/setup_kaggle.sh` to set up the environment in Colab or Kaggle, respectively. Edit the repo URL in those scripts as needed.

## Citation
If you use this codebase, please cite the POT library and any relevant papers.

---

For questions or contributions, please open an issue or pull request.