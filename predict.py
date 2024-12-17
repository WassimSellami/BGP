import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import os

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
    # Check if model and scaler files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    # Load the model without compilation
    model = load_model(model_path, compile=False)
    
    # Recompile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=['mse']
    )
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Read and prepare test data
    df = pd.read_csv(test_file_path)
    
    # Scale the features - update feature columns to match training data exactly
    feature_columns = ['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma']  # Removed 'nb_A_W' as it wasn't in training
    
    # If nb_A_W doesn't exist in the data, calculate it
    if 'nb_A_W' not in df.columns:
        df['nb_A_W'] = df['nb_A'] / (df['nb_W'] + 1e-10)  # Adding small epsilon to avoid division by zero
    
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
    # Define paths for model and scaler from prof directory
    model_path = os.path.join('prof', 'model', 'lstm_model.h5')
    scaler_path = os.path.join('prof', 'scaler', 'scaler.pkl')
    test_file = 'test_data/test_rrc12-ma-1-g3.csv'
    
    # Make predictions
    results_df = predict_from_file(model_path, test_file, scaler_path)
    
    # Save predictions to CSV
    # output_file = f"predictions_{test_file}"
    # results_df.to_csv(output_file, index=False)
    # print(f"Predictions saved to {output_file}")
    
    # Create and save visualization
    plt.figure(figsize=(15, 6))
    plt.plot(results_df['nb_A'].iloc[10:], label='Actual nb_A', alpha=0.5)
    plt.plot(results_df['predicted_nb_A'].iloc[10:], label='Predicted nb_A', alpha=0.5)
    plt.title('Actual vs Predicted Announcements')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Announcements')
    plt.legend()
    plt.savefig('prof_test_predictions.png')
    plt.close()
    print("Visualization saved as prof_test_predictions.png") 