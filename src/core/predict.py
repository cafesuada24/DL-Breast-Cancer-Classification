from typing import Union, Optional
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from lib import MODEL_FILENAME, STD_SCALER_FILENAME


def transform_data(scaler: StandardScaler, data: pd.DataFrame):
    return scaler.transform(data)


def predict(model_path: Path, input: Path, output: Optional[Path] = None) -> pd.Series:
    model = keras.models.load_model(model_path.joinpath(MODEL_FILENAME))
    data = pd.read_csv(input, index_col=0)
    print(data.head())
    with open(model_path.joinpath(STD_SCALER_FILENAME), "rb") as f:
        std_scaler = load(f)

    data = transform_data(std_scaler, data)
    pred = model.predict(data)
    pred = pd.Series(np.argmax(pred, axis=1), name="Type")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(output)
    else:
        print(pred)
