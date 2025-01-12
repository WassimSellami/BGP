import pandas as pd
from constants import Constants
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import pickle

INPUT_FILE = f'train_data/rrc12-ma-{Constants.TIME_WINDOW}-g3.csv'
FEATURE_COLUMNS = [Constants.FEATURE_NB_A, Constants.FEATURE_NB_W, Constants.FEATURE_NB_A_W]    

def prepare_data(filename, feature_columns):
    df = pd.read_csv(filename, nrows=Constants.MAX_TRAIN_ROWS)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    X, y = [], []
    for i in range(len(scaled_data) - Constants.SEQUENCE_LENGTH):
        X.append(scaled_data[i:(i + Constants.SEQUENCE_LENGTH)])
        y.append(scaled_data[i + Constants.SEQUENCE_LENGTH])
    
    X = np.array(X)
    y = np.array(y)
    
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

def train_and_evaluate():


    X_train, X_test, y_train, y_test, scaler = prepare_data(
        INPUT_FILE,
        FEATURE_COLUMNS
    )
    
    model = create_model(Constants.SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    history = model.fit(
        X_train, y_train,
        epochs=Constants.EPOCHS,
        batch_size=Constants.BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train)
    y_test_actual = scaler.inverse_transform(y_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'prediction_results/g3_train_loss_{Constants.TIME_WINDOW}.png')
    plt.close()
    
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_actual[:, 0], label='Actual nb_A')
    plt.plot(test_predictions[:, 0], label='Predicted nb_A')
    plt.title('BGP Updates Prediction - Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Announcements')
    plt.legend()
    plt.savefig(f'prediction_results/g3_train_predictions_{Constants.TIME_WINDOW}.png')
    plt.close()
    
    mse = np.mean((y_test_actual - test_predictions) ** 2)
    rmse = np.sqrt(mse)
    print(f'Test Set RMSE: {rmse}')
    
    with open(f'scalers/g3_scaler_{Constants.TIME_WINDOW}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_and_evaluate()
    model.save(f'models/g3_lstm_{Constants.TIME_WINDOW}.h5')