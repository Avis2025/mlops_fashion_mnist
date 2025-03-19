import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from skmultiflow.drift_detection.adwin import ADWIN

# Load Fashion MNIST and reduce dataset size for efficiency
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:10000], y_train[:10000]  # Reduce to 10K samples
x_test, y_test = x_test[:2000], y_test[:2000]  # Reduce test set for efficiency

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define CNN model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Initialize MLflow experiment
mlflow.set_experiment("FashionMNIST-Tracking")

# Initialize ADWIN for drift detection across runs
adwin = ADWIN()
previous_acc = None

# Train & log model performance over multiple runs
for run in range(5):  # Train 5 times to observe drift
    with mlflow.start_run():
        model = create_model()
        history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=1)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        print(f"Run {run+1}: Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")

        # Add accuracy to ADWIN drift detector
        adwin.add_element(test_acc)
        
        # Check for drift compared to previous runs
        if previous_acc is not None and adwin.detected_change():
            print(f"⚠️ Drift detected in Run {run+1}! Consider retraining the model.")
        
        previous_acc = test_acc  # Update previous accuracy

# MLflow UI command (Run this in your terminal to view logs)
print("\nRun the following command in your terminal to view MLflow logs:\n")
print("mlflow ui --host 0.0.0.0 --port 5000")
print("\nThen navigate to http://localhost:5000 on your browser.")