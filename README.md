# Hydrate NODE Model

A Neural Ordinary Differential Equation (NODE) framework for predicting
**specific hydrate mass concentration in oil at topside** from OLGA
multiphase flow simulation data.

> **Status:** Research prototype — trained and evaluated on OLGA simulation
> cases of a subsea well-riser-topside system.

---

## Table of contents

- [Overview](#overview)
- [Model](#model)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Citation](#citation)

---

## Overview

Hydrate formation in subsea pipelines is a critical flow-assurance challenge.
This work frames hydrate mass prediction as a continuous-time dynamical system
and trains a NODE to learn the governing dynamics directly from simulation
data, using well, riser-base, and topside sensor signals as exogenous inputs.

The model ingests 16 physical features (pressures, temperatures, and mass flow
rates at three locations) and integrates a learned ODE from the initial
condition to produce a full time-series prediction of hydrate concentration.

---

## Model

The NODE learns the latent dynamics:

$$\frac{dy}{dt} = f_\theta\bigl(y(t),\, u(t)\bigr)$$

where:

- $y(t)$ — standardised hydrate mass in oil at topside [kg/m³]
- $u(t)$ — linearly interpolated feature vector at time $t$
- $f_\theta$ — a 2-hidden-layer MLP with Tanh activations

Integration is performed with the `dopri5` adaptive-step solver via
[`torchdiffeq`](https://github.com/rtqichen/torchdiffeq), using the
adjoint method for memory-efficient backpropagation.

**Target transform:** `log1p` followed by z-score standardisation
(fit on training cases only) to handle the heavy-tailed target distribution.

---

## Project structure

```
hydrate-node-model/
├── config.py               # All hyperparameters, paths, column map
├── main.py                 # Entry point — runs the full pipeline
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_loader.py      # File reading, spike detection, case loading
│   ├── preprocessing.py    # Train/val/test split, log1p transform, scaling
│   ├── dataset.py          # TrajectoryDataset wrapper
│   ├── model.py            # NODEFunc, FeatureInterpolator, run_trajectory
│   ├── train.py            # Training loop, early stopping, derivative reg
│   ├── evaluate.py         # predict_dataset, compute_metrics
│   └── visualize.py        # Training history, trajectory, parity, residual plots
│
├── data/
│   └── raw/                # OLGA .txt / .csv files (gitignored — see Data section)
│
├── images/                 # Result plots committed to the repo
├── models/                 # Saved model weights (gitignored)
└── outputs/                # Prediction CSVs, training history (gitignored)
```

---

## Setup

```bash
git clone https://github.com/your-username/hydrate-node-model.git
cd hydrate-node-model
pip install -r requirements.txt
```

Tested on Python 3.10+. GPU training is supported automatically if CUDA is
available; the model falls back to CPU otherwise.

---

## Usage

Place your OLGA simulation files (`.txt` or `.csv`) inside `data/raw/`, then:

```bash
python main.py
```

All hyperparameters and paths are controlled from `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `HIDDEN_DIM` | 32 | MLP hidden layer width |
| `LR` | 1e-4 | AdamW learning rate |
| `NUM_EPOCHS` | 60 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |
| `SPIKE_ABS_THRESHOLD` | 80.0 | Max target value before a case is flagged spiky |
| `ODE_METHOD` | `dopri5` | ODE solver |

---

## Data

The model is trained on OLGA dynamic multiphase flow simulations of a
subsea well-riser-topside system. Each simulation case produces a
time-series of pressures, temperatures, and mass flow rates at three
measurement points:

| Location | Signals |
|---|---|
| Well (P-WELL-B2) | Pressure, temperature, gas/oil/water/total mass flow |
| Riser base (P-RISER-BASE) | Pressure, temperature, gas/oil/water mass flow |
| Topside (P-TOPSIDE) | Pressure, temperature, gas/oil/water mass flow |

**Target:** `HYDMASSOIL P-TOPSIDE` — specific hydrate mass in the oil
layer at topside [kg/m³].

> The raw simulation files are not included in this repository as they
> were generated using the licensed OLGA simulator. To reproduce the
> dataset, OLGA (Schlumberger/SLB) and the corresponding pipeline model
> are required. Contact the authors for access inquiries.

---

## Results

Plots are saved automatically to `images/` when `main.py` is run.


---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{opeyemibisiriyu2025hydratenode,
  author       = {OPEYEMI BISIRIYU},
  title        = {Hydrate {NODE} Model: Neural ODE for Hydrate Mass Prediction},
  year         = {2026},
  howpublished = {\url{https://github.com/captaincode21/hydrate-node-flowassurance}}
}
```
