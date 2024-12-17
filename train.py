import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import pickle

# Load and prepare the data
def prepare_data(filename, feature_columns=['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'], sequence_length=10):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Convert timestamp to datetime
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df.set_index('Timestamp', inplace=True)
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and testing sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Build LSTM model
def create_model(sequence_length, n_features):
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(n_features)
    ])
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=[MeanSquaredError()]
    )
    return model

# Train and evaluate the model
def train_and_evaluate():
    # Parameters
    sequence_length = 10
    feature_columns = ['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma']
    epochs = 4
    batch_size = 32
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        'train_data/rrc12-ma-5-g3.csv',
        feature_columns=feature_columns,
        sequence_length=sequence_length
    )
    
    # Create and train model
    model = create_model(sequence_length, len(feature_columns))
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train)
    y_test_actual = scaler.inverse_transform(y_test)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_loss.png')
    plt.close()
    
    # Plot predictions vs actual for test set
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_actual[:, 0], label='Actual nb_A')
    plt.plot(test_predictions[:, 0], label='Predicted nb_A')
    plt.title('BGP Updates Prediction - Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Announcements')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()
    
    # Calculate and print metrics
    mse = np.mean((y_test_actual - test_predictions) ** 2)
    rmse = np.sqrt(mse)
    print(f'Test Set RMSE: {rmse}')
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

if __name__ == "__main__":
    # Training mode only
    model, scaler = train_and_evaluate()
    model.save('bgp_lstm_model.h5')