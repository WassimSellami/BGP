# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 20:43:52 2021

@author: Talaya Farasat
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:12:38 2021

@author: Talaya
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:35:09 2021

@author: Talaya
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:12:18 2021

@author: Talaya
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:44:11 2021

@author: Talaya
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:16:47 2021

@author: Talaya Farasat
"""

import pickle
from sklearn import metrics
import warnings
from math import sqrt
from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


import matplotlib as mpl
import numpy as np
import pandas as pd  # Basic library for all of our dataset operations

import tensorflow as tf
import xgboost as xgb
from sklearn import linear_model

from matplotlib import pyplot as plt
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.model import ARMA
#arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import sys
import numpy
#from metrics import evaluate
numpy.set_printoptions(threshold=sys.maxsize)



# We will use deprecated models of statmodels which throw a lot of warnings to use more modern ones
warnings.filterwarnings("ignore")


# Extra settings
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('default')
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['xtick.labelsize'] = 12
# mpl.rcParamsfa['ytick.labelsize'] = 12
# mpl.rcParams['text.color'] = 'k'
# mpl.rcParams['figure.figsize'] = 18, 8

mpl.rcParams["figure.figsize"] = (9,8)


print(tf.__version__)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib import pyplot

url='train_data/rrc12-ma-5-g3.csv'


#url='https://raw.githubusercontent.com/ahmadrathore/Datasets-MultiView/main/TelecomData/One_Month_Division/Tel_Data_Chittakong.csv'
#df = pd.read_csv(url, parse_dates=['Date'],)//2021-10-19 
#df.set_index('Date', inplace=True)         //2021-10-19


#url='Datasets/oneMonth_Dhaka_data.csv'

df =pd.read_csv(url, sep=',', header=0, low_memory=False, infer_datetime_format=True,  parse_dates=True)


#pd.set_option("display.max_rows", None, "display.max_columns", None)
#df=pd.read_csv(url, sep=',', header=0, low_memory=False, infer_datetime_format=True, index_col=['Processing_time'], parse_dates=True)
#df.hist()
#pyplot.show()
#df.head()
df.shape


resultsDict = {}
predictionsDict = {}

from sklearn import preprocessing


from sklearn import preprocessing
# df_le = preprocessing.LabelEncoder()
# df['Site ID'] = df_le.fit_transform(df['Site ID']) 
# df['Site Name'] = df_le.fit_transform(df['Site Name']) 
# df['Division'] = df_le.fit_transform(df['Division'])


#split_date = '2018-11-20'
#df_training = df.loc[df.index <= split_date]
#df_test = df.loc[df.index > split_date]

df_training,df_test = df[1:11453], df[11453:] 


print(f"{len(df_training)} days of training data \n {len(df_test)} days of testing data ")

df_training.to_csv('training.csv')
df_test.to_csv('test.csv')

df.head()

df_test['nb_A_W']

from sklearn.model_selection import cross_validate
import pandas as pd
def create_time_features(df, target=None):
    """
    Creates time series features from datetime index
    """
    
    df_1 = pd.DataFrame(df, columns = ['nb_A' , 'nb_W' , 'nb_A_W', 'nb_A_ma', 'nb_W_ma'])
    
    X = df_1
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X
X_train_df, y_train = create_time_features(
    df_training, target='nb_A_W')
X_test_df, y_test = create_time_features(df_test, target='nb_A_W')
scaler = StandardScaler()
scaler.fit(X_train_df)  # No cheating, never scale on the training+test!
X_train = scaler.transform(X_train_df)
X_test = scaler.transform(X_test_df)

X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)
X_test_df

BATCH_SIZE = 64
BUFFER_SIZE = 100
WINDOW_LENGTH = 24


def window_data(X, Y, window=7):
    '''
    The dataset length will be reduced to guarante all samples have the window, so new length will be len(dataset)-window
    '''
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        y.append(Y[i])
    return np.array(x), np.array(y)


# Since we are doing sliding, we need to join the datasets again of train and test
X_w = np.concatenate((X_train, X_test))
y_w = np.concatenate((y_train, y_test))

X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)
X_train_w = X_w[:-len(X_test)]
y_train_w = y_w[:-len(X_test)]
X_test_w = X_w[-len(X_test):]
y_test_w = y_w[-len(X_test):]

# Check we will have same test set as in the previous models, make sure we didnt screw up on the windowing
print(f"Test set equal: {np.array_equal(y_test_w,y_test)}")

train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
val_data = val_data.batch(BATCH_SIZE).repeat()



dropout = 0.0
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(
        128, input_shape=X_train_w.shape[-2:], dropout=dropout),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") #Support for tensorboard tracking!
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)





EVALUATION_INTERVAL = 150
#EVALUATION_INTERVAL = 5
EPOCHS = 1

model_history = simple_lstm_model.fit(train_data, epochs=EPOCHS,
                                      steps_per_epoch=EVALUATION_INTERVAL,
                                      validation_data=val_data, validation_steps=10)  # ,callbacks=[tensorboard_callback]) #Uncomment this line for tensorboard support
                                      #validation_data=val_data, validation_steps=10)  # ,callbacks=[tensorboard_callback]) #Uncomment this line for tensorboard support

# Create model directory if it doesn't exist
import os
model_dir = 'prof/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model_path = os.path.join(model_dir, 'lstm_model.h5')
simple_lstm_model.save(model_path)
print(f"Model saved to {model_path}")

# Continue with predictions
yhat = simple_lstm_model.predict(X_test_w).reshape(1, -1)[0]
#resultsDict['Tensorflow simple LSTM'] = evaluate(y_test, yhat)
predictionsDict['Tensorflow simple LSTM'] = yhat










plt.xlabel('Time steps', fontsize=23, fontweight="bold")
plt.ylabel('Number of Announcements', fontsize=27, fontweight="bold")
plt.yscale("log")





plt.plot(df_test['nb_A'].values[4708:4908], label='Original',linewidth=4.0,color='black')
plt.plot(yhat[4708:4908], color='#FF1493', label='LSTM ',linewidth=4.0)
plt.xticks(fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")
plt.legend(fontsize=28, loc='upper left')
plt.tight_layout()
plt.savefig('prof/prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#print('Original',resultsDict['XGBoost'])
#print('Predicted',yhat)
print("RMSE : ",np.sqrt(mean_squared_error(df_test['nb_A'],predictionsDict['Tensorflow simple LSTM'])),end=",     ")
print("MAE: ",mean_absolute_error(df_test['nb_A'],predictionsDict['Tensorflow simple LSTM']),end=",     ")
print("MAPE : ",mean_absolute_percentage_error(df_test['nb_A'],predictionsDict['Tensorflow simple LSTM']),end=",     ")
print("r2 : ",r2_score(df_test['nb_A'],predictionsDict['Tensorflow simple LSTM']),end=",     ")