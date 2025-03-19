import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt

# Load Fashion MNIST
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten images to 1D vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test

# Feature Engineering (Normalization)
def preprocess_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train a simple model for explainability
def train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
    return model

def explainability_analysis(model, X_train):
    explainer = shap.Explainer(model, X_train[:1000])  # Use a subset for efficiency
    shap_values = explainer(X_train[:100], max_evals=2000)  # Increase max_evals
    
    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_train[:100])

# Main function
def main():
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    print("Running explainability analysis...")
    explainability_analysis(model, X_train_scaled)
    
if __name__ == "__main__":
    main()
