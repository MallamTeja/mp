import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib


DATASET_PATH = Path("dataset.json")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)
