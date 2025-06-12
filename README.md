
```
tmw_project/
├── data/
│   ├── raw/                # downloaded UCR .tsv/.txt files
│   └── processed/          # z-normalized, train/test splits
│
├── src/
│   ├── tmw/                
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # z-normalization, loading, splitting :contentReference[oaicite:0]{index=0}
│   │   ├── mask.py             # build temporal mask matrices for window w
│   │   ├── sinkhorn.py         # entropic OT solver (ε fixed to 1e-7) :contentReference[oaicite:1]{index=1}
│   │   ├── network_simplex.py  # exact OT solver via LEMON or OR-tools
│   │   └── evaluation.py       # 1-NN classifier, accuracy, timing
│   │
│   ├── benchmarks/         
│   │   ├── dtw.py
│   │   ├── otw.py
│   │   ├── pow.py
│   │   ├── opw.py
│   │   └── taot.py
│   │
│   └── utils.py             # logging, config parsing, plotting helpers
│
├── experiments/
│   ├── configs/             
│   │   ├── base.yaml        # dataset paths, hardware specs :contentReference[oaicite:2]{index=2}
│   │   ├── tuning_w_coarse.yaml
│   │   └── tuning_lambda.yaml
│   │
│   ├── logs/                # stdout/stderr from runs
│   └── results/             # CSVs of (w,λ)→accuracy, timing, plots
│
├── notebooks/
│   ├── 01_explore_datasets.ipynb
│   ├── 02_parameter_tuning.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sinkhorn.py
│   └── test_network_simplex.py
│
├── requirements.txt         # NumPy, SciPy, pandas, matplotlib, POT, python-lemon-binding :contentReference[oaicite:3]{index=3}
├── setup.py
└── README.md
```