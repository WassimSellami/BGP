import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
from constants import (
    FEATURE_NB_A,
    FEATURE_NB_A_MA,
    FEATURE_NB_A_W,
    FEATURE_NB_W,
    FEATURE_NB_W_MA,
    SEQUENCE_LENGTH,
    MODEL_PATH,
    SCALER_PATH,
    TEST_DATA_DIR
)

SAMPLE_SIZE = 500

def create_time_features(df, target=None):
    df_1 = pd.DataFrame(df, columns=[FEATURE_NB_A, FEATURE_NB_W, FEATURE_NB_A_W, FEATURE_NB_A_MA, FEATURE_NB_W_MA])
    X = df_1
    
    if target:
        if isinstance(target, list):
            y = df[target]
            X = X.drop(target, axis=1)
        else:
            y = df[target]
            X = X.drop([target], axis=1)
        return X, y
    return X

def window_data(X, window=SEQUENCE_LENGTH):
    x = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
    return np.array(x)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict_from_file(model_path, test_file_path, scaler_path, sequence_length=SEQUENCE_LENGTH):
    df_test = pd.read_csv(test_file_path, sep=',', header=0, low_memory=False,
                         parse_dates=True)
    
    X_test_df, y_test = create_time_features(df_test, target=[FEATURE_NB_A, FEATURE_NB_W])
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test = scaler.transform(X_test_df)
    
    X_test_w = window_data(X_test, window=sequence_length)
    
    model = load_model(model_path)
    predictions = model.predict(X_test_w)
    nb_A_pred = predictions[:, 0]
    nb_W_pred = predictions[:, 1]
    
    original_nb_A = df_test[FEATURE_NB_A].values[sequence_length-1:]
    original_nb_W = df_test[FEATURE_NB_W].values[sequence_length-1:]
    
    start_idx = max(0, len(nb_A_pred) - SAMPLE_SIZE)
    end_idx = len(nb_A_pred)
    
    plt.figure(figsize=(15, 6))
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Announcements')
    plt.title('BGP Updates Prediction - Test Set')
    
    plt.plot(original_nb_A[start_idx:end_idx], 
             label='Actual nb_A', linewidth=1.0, color='blue', alpha=0.8)
    plt.plot(nb_A_pred[start_idx:end_idx], 
             label='Predicted nb_A', linewidth=1.0, color='orange', alpha=0.8)
    
    plt.legend()
    plt.tight_layout()
    
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    
    if not os.path.exists('prediction_results'):
        os.makedirs('prediction_results')
    
    plt.savefig(f'prediction_results/{test_file_name}_nb_A.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 6))
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Withdrawals')
    plt.title('BGP Updates Prediction - Test Set')
    
    plt.plot(original_nb_W[start_idx:end_idx], 
             label='Actual nb_W', linewidth=1.0, color='blue', alpha=0.8)
    plt.plot(nb_W_pred[start_idx:end_idx], 
             label='Predicted nb_W', linewidth=1.0, color='orange', alpha=0.8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'prediction_results/{test_file_name}_nb_W.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Metrics for Announcements:")
    print("RMSE : ", np.sqrt(mean_squared_error(original_nb_A, nb_A_pred)), end=",     ")
    print("MAE: ", mean_absolute_error(original_nb_A, nb_A_pred), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(original_nb_A, nb_A_pred), end=",     ")
    print("r2 : ", r2_score(original_nb_A, nb_A_pred))

    print("\nMetrics for Withdrawals:")
    print("RMSE : ", np.sqrt(mean_squared_error(original_nb_W, nb_W_pred)), end=",     ")
    print("MAE: ", mean_absolute_error(original_nb_W, nb_W_pred), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(original_nb_W, nb_W_pred), end=",     ")
    print("r2 : ", r2_score(original_nb_W, nb_W_pred))
    
    results_df = pd.DataFrame({
        'original_nb_A': original_nb_A,
        'predicted_nb_A': nb_A_pred,
        'original_nb_W': original_nb_W,
        'predicted_nb_W': nb_W_pred
    })
    
    return results_df

if __name__ == "__main__":
    model_path = MODEL_PATH
    scaler_path = SCALER_PATH
    test_file = os.path.join(TEST_DATA_DIR, 'test_rrc12-ma-1-g3.csv')
    
    results_df = predict_from_file(model_path, test_file, scaler_path)
    