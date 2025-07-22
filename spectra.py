# -*- coding: utf-8 -*-
"""Spectra.ipynb

This is my export of my working logic from my Jupyter notebook run on Colab.
It contains the exploratory data analysis of my training data and training runs and plots if various models.
It exports the artificial neural network created using fast ai as a saved model for future price predictions

Original file is located at
    https://colab.research.google.com/drive/1WkmfCcsuFGY6xeEDPBIfKP6cFj2BOg5A

### Mount Google Drive to Access Files
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""### Copy Dataset File from Google Drive to Colab"""

# Copy the files
!cp /content/drive/My\ Drive/Colab\ Notebooks/spectra/data/CryptocurrencyData.csv /content

# Check to make sure dataset file is on Google Colab
!ls -la /content

"""### Import Relevant Libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

"""###  Load and  View Dataset"""

df = pd.read_csv('/content/CryptocurrencyData.csv')
df.head(5)

df.shape

"""### Drop the follwing columns from the dataset
- Rank
- Coin Name
- Symbol
- 1h
- 24h
- 7d
- 30d
"""

def drop_columns(df):
  columns_to_drop = ["Rank", "Coin Name", "Symbol", "1h", "24h", "7d", "30d"]
  df = df.drop(columns_to_drop, axis=1)
  return df
df = drop_columns(df)
df.head()

"""#### Rename columns to remove whitespace"""

df = df.rename(columns=lambda x: x.strip().replace(' ', '_'))
df.head()

"""### Convert `Price`, `24_Volume`, `Circulating_Supply` and `Market_Cap` column values to floats"""

def convert_to_float(value):
  if pd.isna(value):
    return None
  value = value.replace('$', '').replace(',', '')
  if '-' in value:
    return None
  else:
    return float(value)

df['Price'] = df['Price'].apply(convert_to_float)
df['24h_Volume'] = df['24h_Volume'].apply(convert_to_float)
df['Circulating_Supply'] = df['Circulating_Supply'].apply(convert_to_float)
df['Market_Cap'] = df['Market_Cap'].apply(convert_to_float)
df.head()

"""### Convert `Total Supply` column to numeric values"""

def convert_supply_to_float(value):
  if pd.isna(value):
    return None
  value = value.replace(',', '')
  if 'Thousand' in value:
    return float(value.split()[0]) * 1000
  elif 'Million' in value:
    return float(value.split()[0]) * 1000000
  elif 'Billion' in value:
    return float(value.split()[0]) * 1000000000
  elif 'Trillion' in value:
    return float(value.split()[0]) * 1000000000000
  elif 'Quadrillion' in value:
    return float(value.split()[0]) * 1000000000000000
  elif '∞' in value:
    return None
  elif '-' in value:
    return None
  elif '. M' in value:
    return None
  else:
    return float(value)

df['Total_Supply'] = df['Total_Supply'].apply(convert_supply_to_float)
df.head(5)

"""### Drop all rows across the entire dataframe that have a `NaN` value in any column"""

df = df.dropna(subset=['Price', '24h_Volume', 'Circulating_Supply', 'Total_Supply', 'Market_Cap'])
df.head()

df.shape

"""### Save Dataframe for NN regression"""

df_nn = df.copy()
df_nn.shape

"""### Separate Variables into Independent and Dependent Variables Dataframes"""

y = df['Price']
X = df.drop('Price', axis=1)

y.shape

X.shape

"""### Apply scaling to the dataset to reduce the effect of very large or very small values in X"""

import pickle
from sklearn.preprocessing import StandardScaler

columns_to_scale = ['24h_Volume', 'Circulating_Supply', 'Total_Supply', 'Market_Cap']

scaler = StandardScaler()

X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X.head()

"""### Split the data into train and test sets"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Use Random Forest Regressor to fit the data"""

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

"""### Predict prices on the test set and show metrics"""

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

"""### Predicted vs Actual Prices Plot"""

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

"""### Feature Importance Plot"""

# Feature Importance Plot
feature_importances = model.feature_importances_
features = X.columns
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.show()

"""### Residual Plot"""

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

"""### Use a Linear Regression Model to predict prices"""

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print('Linear Regression - Mean Squared Error:', mse_linear)
print('Linear Regression - Root Mean Squared Error:', rmse_linear)
print('Linear Regression - R-squared:', r2_linear)

plt.scatter(y_test, y_pred_linear)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Linear Regression)')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.show()

"""### Residual Plot of Linear Regression Model"""

# Residual plot
residuals = y_test - y_pred_linear
plt.scatter(y_pred_linear, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression Model")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

!pip install fastai
from fastai.tabular.all import *

cont_names = ['24h_Volume', 'Circulating_Supply', 'Total_Supply', 'Market_Cap']

dls = TabularDataLoaders.from_df(df_nn, y_names='Price', cont_names=cont_names,
                                 procs=[FillMissing, Normalize])

y = dls.train.y
y.min(),y.max()

"""### Create the Learner"""

# Create a learner
learn = tabular_learner(dls, y_range=(0.01,36457), layers=[500, 250],
                        n_out=1, loss_func=F.mse_loss)

# learn.lr_find()

"""### Fit the Model with an appropriate learning rate"""

# Fit the model
learn.fit_one_cycle(300)

"""#  Calculate MSE from the Model"""

def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)

preds,targs = learn.get_preds()
r_mse(preds,targs)

from sklearn.metrics import mean_squared_error, r2_score

mse_nn = mean_squared_error(targs, preds)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(targs, preds)

print('Neural Network - Mean Squared Error:', mse_nn)
print('Neural Network - Root Mean Squared Error:', rmse_nn)
print('Neural Network - R-squared:', r2_nn)

plt.scatter(targs, preds)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices (Neural Network)')
plt.title('Actual vs Predicted Prices (Neural Network)')
plt.show()

"""### Residual Plot of NN Model"""

# Residual plot
residuals = targs - preds
plt.scatter(preds, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot - Neural Network Model")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

"""### Save the NN Model"""

# Use the format below to save the file to contain an optimizer state.

learn.export('nn_model.pkl')

"""### Make a Price Prediction from the Saved NN Model"""

learn = load_learner('nn_model.pkl')

new_record = pd.DataFrame({
    '24h_Volume': [110179781],
    'Circulating_Supply': [4413922778],
    'Total_Supply': [6449215559],
    'Market_Cap': [65768788]
})

dl = learn.dls.test_dl(new_record)
prediction, *_ = learn.get_preds(dl=dl)

print("Predicted Price:", prediction[0].item())

# Load the scaler
with open('scaler.pkl', 'rb') as f:
  loaded_scaler = pickle.load(f)

new_record = pd.DataFrame({
    '24h_Volume': [110179781],
    'Circulating_Supply': [4413922778],
    'Total_Supply': [6449215559],
    'Market_Cap': [65768788]
})

# Scale the features using the same scaler used for training
new_record[columns_to_scale] = scaler.transform(new_record[columns_to_scale])

# Make the prediction using the trained linear regression model
predicted_price = linear_model.predict(new_record)

print("Predicted Price (Linear Regression):", predicted_price[0])

"""### Actual Details
- Token Name: Alien Worlds
- Symbol: TLM
- Price: $0.01441
"""

