from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from lib import MODEL_FILENAME, STD_SCALER_FILENAME

random_state = 2024


def train(dataset_path: Path, output_dir: Path):
    df = pd.read_csv(dataset_path, index_col=0)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = keras.Sequential(
        [
            keras.Input((X.shape[1],)),  # input layer
            keras.layers.Flatten(),
            keras.layers.Dense(25, activation="relu"),  # hidden layer
            keras.layers.Dense(2, activation="sigmoid"),  # output layer
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=10, validation_split=0.1)

    loss, accuracy = model.evaluate(X_test, y_test)

    print(
        f"""
        TEST RESULT
        Loss:     {loss}
        Accuracy: {accuracy}
    """
    )

    model.save(output_dir.joinpath(MODEL_FILENAME))
    with open(output_dir.joinpath(STD_SCALER_FILENAME), "wb") as f:
        dump(scaler, f)
