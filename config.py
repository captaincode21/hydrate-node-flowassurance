import os
import random
import numpy as np
import torch

# -------------------------
# REPRODUCIBILITY
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# PATHS
# -------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# -------------------------
# SPIKE FILTERING
# -------------------------
AUTO_REMOVE_SPIKY_CASES = True
SPIKE_ABS_THRESHOLD     = 80.0
SPIKE_RATIO_THRESHOLD   = 3.0

# -------------------------
# OPTIONAL FLAGS
# -------------------------
APPLY_TRIMMING      = False
USE_DERIVATIVE_REG  = False
DERIV_LAMBDA        = 0.05

# -------------------------
# MODEL / TRAINING
# -------------------------
HIDDEN_DIM   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 60
PATIENCE     = 10
GRAD_CLIP    = 0.5

# -------------------------
# ODE SOLVER
# -------------------------
ODE_METHOD = "dopri5"
ODE_RTOL   = 1e-4
ODE_ATOL   = 1e-6

# -------------------------
# COLUMN MAP
# -------------------------
RENAME_DICT = {
    "TIME[s]": "time_s",

    "PT P-WELL-B2 Pressure[bara]":                                          "well_pressure",
    "TM P-WELL-B2 Fluid temperature[C]":                                    "well_temp",
    "GG P-WELL-B2 Gas mass flow[kg/s]":                                     "well_gas_flow",
    "GLTHL P-WELL-B2 Mass flow rate of oil[kg/s]":                          "well_oil_flow",
    "GLTWT P-WELL-B2 Mass flow rate of water excluding vapour[kg/s]":       "well_water_flow",
    "GT P-WELL-B2 Total mass flow[kg/s]":                                   "well_total_flow",

    "PT P-RISER-BASE Pressure[bara]":                                       "riser_base_pressure",
    "TM P-RISER-BASE Fluid temperature[C]":                                 "riser_base_temp",
    "GG P-RISER-BASE Gas mass flow[kg/s]":                                  "riser_base_gas_flow",
    "GLTHL P-RISER-BASE Mass flow rate of oil[kg/s]":                       "riser_base_oil_flow",
    "GLTWT P-RISER-BASE Mass flow rate of water excluding vapour[kg/s]":    "riser_base_water_flow",

    "PT P-TOPSIDE Pressure[bara]":                                          "topside_pressure",
    "TM P-TOPSIDE Fluid temperature[C]":                                    "topside_temp",
    "GG P-TOPSIDE Gas mass flow[kg/s]":                                     "topside_gas_flow",
    "GLTHL P-TOPSIDE Mass flow rate of oil[kg/s]":                          "topside_oil_flow",
    "GLTWT P-TOPSIDE Mass flow rate of water excluding vapour[kg/s]":       "topside_water_flow",

    "HYDMASSOIL P-TOPSIDE Specific hydrate mass in oil layer[kg/m3]":       "target_hyd_oil_topside",
}

SELECTED_COLS = list(RENAME_DICT.keys())

FEATURE_COLS = [
    "well_pressure", "well_temp", "well_gas_flow", "well_oil_flow",
    "well_water_flow", "well_total_flow",
    "riser_base_pressure", "riser_base_temp", "riser_base_gas_flow",
    "riser_base_oil_flow", "riser_base_water_flow",
    "topside_pressure", "topside_temp", "topside_gas_flow",
    "topside_oil_flow", "topside_water_flow",
]

TARGET_COL = "target_hyd_oil_topside"
