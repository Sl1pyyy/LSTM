import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import kagglehub
import matplotlib.pyplot as plt


path = kagglehub.dataset_download('mnassrib/jena-climate')
csv_filename = 'jena_climate_2009_2016.csv'
csv_path = os.path.join(path, csv_filename)

df = pd.read_csv(csv_path)

df.index = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
temp = df['T (degC)']

def df_to_X_y(df, window_size = 5):
    """Create a two arrays of X's and y's as labels for upcoming workflow"""
    df_as_np = df.to_numpy()
    X = [df_as_np[i:i + 5].reshape(-1,1).tolist() for i in range(len(df_as_np) - window_size)]
    y = [df_as_np[i+5] for i in range(len(df_as_np) - window_size)]
    return np.array(X), np.array(y)

X, y = df_to_X_y(temp, window_size=5)

X_train, y_train = X[:336000], y[:336000]
X_val, y_val = X[336000:370000], y[336000:370000]
X_test, y_test = X[336000:], y[336000:]

model_lstm1 = Sequential()
model_lstm1.add(InputLayer((5,1)))
model_lstm1.add(LSTM(64))
model_lstm1.add(Dense(8, 'relu'))
model_lstm1.add(Dense(1, 'linear'))
model_lstm1.summary()

cp = ModelCheckpoint('model_lstm1.keras', save_best_only=True)

model_lstm1.compile(loss = 'mse',
                    optimizer = Adam(learning_rate= 0.0001),
                    metrics = [RootMeanSquaredError()]
                    )

model_lstm1.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=10, callbacks=[cp])

model_lstm1 = load_model('model_lstm1.keras')

train_predictions = model_lstm1.predict(X_train).flatten()
train_results = pd.DataFrame(data = {'Train predictions': train_predictions, 'Actual predictions': y_train})

val_predictions = model_lstm1.predict(X_val).flatten()
val_results = pd.DataFrame(data = {'Train predictions': val_predictions, 'Actual predictions': y_val})

test_predictions = model_lstm1.predict(X_test).flatten()
test_results = pd.DataFrame(data = {'Train predictions': test_predictions, 'Actual predictions': y_test})

plt.figure(figsize = (12,8))
plt.plot(train_results['Train predictions'][:100], label='Train predictions', color='blue')
plt.plot(train_results['Actual predictions'][:100], label='Actual predictions', color='red')

plt.plot(val_results['Train predictions'][:100], label='Train predictions', color='green')
plt.plot(val_results['Actual predictions'][:100], label='Actual predictions', color='yellow')

plt.plot(test_results['Train predictions'][:100], label='Train predictions', color='black')
plt.plot(test_results['Actual predictions'][:100], label='Actual predictions', color='orange')
plt.legend(loc = 'upper left')
plt.show()