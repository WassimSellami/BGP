import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn import linear_model
import warnings
import pickle
warnings.filterwarnings("ignore")

from constants import Constants

TRAINING_DATA_FILE = 'train_data/rrc12-ma-5-g3.csv'

plt.style.use('default')
plt.rcParams["figure.figsize"] = (9, 8)

tf.random.set_seed(Constants.SEED)
np.random.seed(Constants.SEED)

def create_time_features(df, target=None):
    # Drop the target from features to avoid data leakage
    df_1 = pd.DataFrame(df, columns=[Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W, Constants.FEATURE_NB_A_MA, Constants.FEATURE_NB_W_MA])
    X = df_1
    
    if target:
        y = df[target].copy()
        return X, y
    return X

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    # Load data
    df = pd.read_csv(TRAINING_DATA_FILE, sep=',', header=0, low_memory=False, 
                    infer_datetime_format=True, parse_dates=True)

    train_size = int(len(df) * 0.8)
    df_training = df[1:train_size]  # Keep original indexing
    df_test = df[train_size:]
    print(f"{len(df_training)} training samples\n{len(df_test)} testing samples")

    # Prepare features
    X_train_df, y_train_A = create_time_features(df_training, target=Constants.FEATURE_NB_A)
    X_test_df, y_test_A = create_time_features(df_test, target=Constants.FEATURE_NB_A)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)

    # Train model with correct parameters
    reg_A = linear_model.BayesianRidge(
        max_iter=300,
        tol=1e-6
    )
    reg_A.fit(X_train, y_train_A)
    nb_A_pred = reg_A.predict(X_test)

    # Save model and scaler
    with open('model/br_model.pkl', 'wb') as f:
        pickle.dump(reg_A, f)
    with open('scaler/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved successfully")

    # Select a window of 200 samples for visualization
    window_start = 0
    window_size = 200
    
    # Get the actual values and predictions for the window
    actual_values = df_test[Constants.FEATURE_NB_A].values[window_start:window_start+window_size]
    predicted_values = nb_A_pred[window_start:window_start+window_size]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(actual_values, label='Original', color='black', linewidth=2.0)
    plt.plot(predicted_values, label='Bayesian Ridge', color='#00FF00', linewidth=2.0, linestyle='--')
    
    plt.xlabel('Time steps', fontsize=23, fontweight="bold")
    plt.ylabel('Number of Announcements', fontsize=27, fontweight="bold")
    plt.yscale("log")
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=28, loc='upper left')
    plt.title('nb_A Predictions', fontsize=25, fontweight="bold")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('prediction_results/br_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nMetrics for nb_A:")
    print("RMSE : ", np.sqrt(mean_squared_error(df_test[Constants.FEATURE_NB_A], nb_A_pred)), end=",     ")
    print("MAE: ", mean_absolute_error(df_test[Constants.FEATURE_NB_A], nb_A_pred), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(df_test[Constants.FEATURE_NB_A], nb_A_pred), end=",     ")
    print("r2 : ", r2_score(df_test[Constants.FEATURE_NB_A], nb_A_pred))

if __name__ == "__main__":
    main()