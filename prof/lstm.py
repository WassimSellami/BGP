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
import pickle
import warnings
warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 64
BUFFER_SIZE = 100
WINDOW_LENGTH = 24
EVALUATION_INTERVAL = 150
EPOCHS = 100
MODEL_DIR = 'prof/model'
SCALER_DIR = 'prof/scaler'

tf.random.set_seed(SEED)
np.random.seed(SEED)

plt.style.use('default')
plt.rcParams["figure.figsize"] = (9, 8)

def create_time_features(df, target=None):
    df_1 = pd.DataFrame(df, columns=['nb_A', 'nb_W', 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])
    X = df_1
    
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X

def window_data(X, Y, window=7):
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        y.append(Y[i])
    return np.array(x), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    url = 'train_data/rrc12-ma-5-g3.csv'
    df = pd.read_csv(url, sep=',', header=0, low_memory=False, 
                    infer_datetime_format=True, parse_dates=True)

    df_training, df_test = df[1:11453], df[11453:]
    print(f"{len(df_training)} days of training data\n{len(df_test)} days of testing data")

    df_training.to_csv('training.csv')
    df_test.to_csv('test.csv')

    X_train_df, y_train = create_time_features(df_training, target='nb_A_W')
    X_test_df, y_test = create_time_features(df_test, target='nb_A_W')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    if not os.path.exists(SCALER_DIR):
        os.makedirs(SCALER_DIR)
    scaler_path = os.path.join(SCALER_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    X_w = np.concatenate((X_train, X_test))
    y_w = np.concatenate((y_train, y_test))
    X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)
    
    X_train_w = X_w[:-len(X_test)]
    y_train_w = y_w[:-len(X_test)]
    X_test_w = X_w[-len(X_test):]
    y_test_w = y_w[-len(X_test):]

    train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=X_train_w.shape[-2:], dropout=0.0),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
    ])
    simple_lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    model_history = simple_lstm_model.fit(
        train_data, 
        epochs=EPOCHS,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data=val_data, 
        validation_steps=10
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    simple_lstm_model.save(model_path)
    print(f"Model saved to {model_path}")

    yhat = simple_lstm_model.predict(X_test_w).reshape(1, -1)[0]

    plt.figure()
    plt.xlabel('Time steps', fontsize=23, fontweight="bold")
    plt.ylabel('Number of Announcements', fontsize=27, fontweight="bold")
    plt.yscale("log")
    plt.plot(df_test['nb_A'].values[4708:4908], label='Original', linewidth=4.0, color='black')
    plt.plot(yhat[4708:4908], color='#FF1493', label='LSTM', linewidth=4.0)
    plt.xticks(fontsize=15, fontweight="bold")
    plt.yticks(fontsize=15, fontweight="bold")
    plt.legend(fontsize=28, loc='upper left')
    plt.tight_layout()
    plt.savefig('prof/prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("RMSE : ", np.sqrt(mean_squared_error(df_test['nb_A'], yhat)), end=",     ")
    print("MAE: ", mean_absolute_error(df_test['nb_A'], yhat), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(df_test['nb_A'], yhat), end=",     ")
    print("r2 : ", r2_score(df_test['nb_A'], yhat))

if __name__ == "__main__":
    main()