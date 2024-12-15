import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
import matplotlib.pyplot as plt
import pickle

def predict_from_file(model_path, test_file_path, scaler_path, sequence_length=10):
    """
    Make predictions using a saved model on new test data
    
    Args:
        model_path: Path to the saved model (.h5 file)
        test_file_path: Path to the test features CSV file
        scaler_path: Path to the saved scaler (.pkl file)
        sequence_length: Length of input sequences
        
    Returns:
        DataFrame with original and predicted values
    """
    # Load the model with proper custom objects
    custom_objects = {
        'loss': MeanSquaredError(),
        'mse': MSE()
    }
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Read and prepare test data
    df = pd.read_csv(test_file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Scale the features
    feature_columns = ['nb_A', 'nb_W', 'nb_A_W']
    scaled_data = scaler.transform(df[feature_columns])
    
    # Create sequences
    X = []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
    X = np.array(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    
    # Create results DataFrame
    results_df = df.copy()
    
    # Add predicted values (shifted by sequence_length due to the sequence requirement)
    results_df.loc[sequence_length:, 'predicted_nb_A'] = predictions[:, 0]
    results_df.loc[sequence_length:, 'predicted_nb_W'] = predictions[:, 1]
    
    return results_df

if __name__ == "__main__":
    # Prediction mode
    test_file = 'test_features.csv'
    results_df = predict_from_file('bgp_lstm_model.h5', test_file, 'scaler.pkl')
    
    output_file = "predictions_"+test_file
    results_df.to_csv(output_file, index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(results_df['nb_A'].iloc[10:], label='Actual nb_A', alpha=0.5)
    plt.plot(results_df['predicted_nb_A'].iloc[10:], label='Predicted nb_A', alpha=0.5)
    plt.title('Actual vs Predicted Announcements')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Announcements')
    plt.legend()
    plt.savefig('test_predictions.png')
    plt.close() 