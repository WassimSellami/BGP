import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from constants import Constants

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_time_features(df, target=None):
    df_1 = pd.DataFrame(df, columns=[
        Constants.FEATURE_NB_W, 
        Constants.FEATURE_NB_A_W, 
        Constants.FEATURE_NB_A_MA, 
        Constants.FEATURE_NB_W_MA
    ])
    X = df_1
    
    if target:
        y = df[target].copy()
        return X, y
    return X

def main():
    # Load the model and scaler
    with open('model/br_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load and prepare test data
    test_df = pd.read_csv('test_data/test_data1.csv', sep=',', header=0)
    
    # Create features and targets
    X_test_df, y_test = create_time_features(test_df, target=Constants.FEATURE_NB_A)
    
    # Scale features
    X_test = scaler.transform(X_test_df)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Debug prints to check alignment
    print("\nChecking data alignment:")
    print(f"Original data length: {len(test_df)}")
    print(f"Predictions length: {len(predictions)}")
    print(f"Target values length: {len(y_test)}")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save announcements predictions to CSV with time alignment information
    predictions_df = pd.DataFrame({
        'Time_Step': range(len(y_test)),
        'Actual_Announcements': y_test,
        'Predicted_Announcements': predictions
    })
    
    # Print first few rows to verify alignment
    print("\nFirst few predictions with alignment:")
    print(predictions_df.head())
    print("\nSample of prediction errors:")
    predictions_df['Error'] = abs(predictions_df['Actual_Announcements'] - predictions_df['Predicted_Announcements'])
    print(predictions_df[['Time_Step', 'Actual_Announcements', 'Predicted_Announcements', 'Error']].head())
    
    # Create larger, single plot for announcements
    plt.figure(figsize=(15, 10))
    
    # Plot Announcements with correct time alignment
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Number of Announcements', fontsize=14)
    time_points = predictions_df['Time_Step']
    
    # Plot actual values
    plt.plot(time_points, y_test, label='Actual', linewidth=2.5, color='blue', alpha=0.8)
    
    # Plot predictions
    plt.plot(time_points, predictions, label='Predicted', linewidth=2.5, color='orange', alpha=0.8)
    
    # Add error bars or shading to show prediction uncertainty
    plt.fill_between(time_points, 
                     predictions - predictions_df['Error'],
                     predictions + predictions_df['Error'],
                     color='orange', alpha=0.2, label='Prediction Error Range')
    
    plt.legend(fontsize=12, loc='upper right')
    plt.title('BGP Announcements: Actual vs Predicted (Bayesian Ridge)', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics_text = f'Metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('prediction_results/test1_comparison_br.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print metrics
    print("\nMetrics for Announcements:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R2:", r2)

if __name__ == "__main__":
    main()
