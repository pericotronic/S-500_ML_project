import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import seaborn as sns
import datetime
from finta import TA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import tensorflow as tf

import xgboost as xgb
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import pickle
import os
import gzip


data = yf.download(tickers = '^GSPC', start = '2005-01-03',end = '2023-02-08')




"""
Defining some constants for data mining
"""

NUM_DAYS = 10000     # The number of days of historical data to retrieve
INTERVAL = '1d'     # Sample rate of historical data
symbol = '^GSPC'      # Symbol of the desired stock

# List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']




"""
Next we pull the historical data using yfinance
Rename the column names because finta uses the lowercase names
"""

start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
end = datetime.datetime.today()

data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
print(data.head())

tmp = data.iloc[-60:]
#tmp['close'].plot()






"""
Next we clean our data and perform feature engineering to create new technical indicator features that our
model can learn from
"""

def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """
    
    return data.ewm(alpha=alpha).mean()

data = _exponential_smooth(data, 0.65)

tmp1 = data.iloc[-60:]
#plt.plot(tmp1['close'])
#plt.gcf().autofmt_xdate() # format the x-axis to show dates
















def _get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    del (data['Adj Close'])
    
    return data

data = _get_indicator_data(data)













#plt.figure(figsize=(10, 10))
# 
## As our concern is with the highly correlated features only so, we will visualize our heatmap as per that criteria only.
#sns.heatmap(data.corr() > 0.9, annot=True, cbar=False)
#plt.show()






def _produce_prediction(data, window):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """
    
    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    
    return data

data = _produce_prediction(data, window=15)
del (data['close'])
data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here








from sklearn.model_selection import TimeSeriesSplit

y = data['pred']
features = [x for x in data.columns if x not in ['pred']]
X = data[features]

tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]






def _train_random_forest(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    print(rf_gs.best_params_)
    
    prediction = rf_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return rf_best
    
rf_model = _train_random_forest(X_train, y_train, X_test, y_test)












def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}

    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)

    # Fit model to training data
    knn_gs.fit(X_train, y_train)

    # Save best model
    knn_best = knn_gs.best_estimator_
    
    # Check best n_neigbors value
    print(knn_gs.best_params_)

    prediction = knn_best.predict(X_test)

    print(classification_report(y_test, prediction, zero_division=1))
    print(confusion_matrix(y_test, prediction))

    return knn_best
    
    
knn_model = _train_KNN(X_train, y_train, X_test, y_test)















from xgboost import XGBClassifier
def _train_XGBoost(X_train, y_train, X_test, y_test):

    xgb = XGBClassifier()
    # Create a dictionary of all values we want to test for hyperparameters
    params_xgb = {'learning_rate': [0.01, 0.05, 0.1],
                  'max_depth': [3, 5, 7],
                  'n_estimators': [100, 500, 1000]}
    
    # Use gridsearch to test all values for hyperparameters
    xgb_gs = GridSearchCV(xgb, params_xgb, cv=5)
    
    # Fit model to training data
    xgb_gs.fit(X_train, y_train)
    
    # Save best model
    xgb_best = xgb_gs.best_estimator_
     
    # Check best hyperparameters value
    print(xgb_gs.best_params_)
    
    prediction = xgb_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return xgb_best
    
    
xgb_model = _train_XGBoost(X_train, y_train, X_test, y_test)










from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameters to search
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05,0.1,1],
    'max_depth': [0.05,1,5]
}

# Initialize the classifier
gbt_model = GradientBoostingClassifier(random_state=42)

# Perform the grid search
grid_search = GridSearchCV(gbt_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Use the best hyperparameters found by GridSearchCV to train the model
gbt_model = GradientBoostingClassifier(n_estimators=grid_search.best_params_['n_estimators'], 
                                        learning_rate=grid_search.best_params_['learning_rate'], 
                                        max_depth=grid_search.best_params_['max_depth'], 
                                        random_state=42)
gbt_model.fit(X_train, y_train)

# Print the best parameters and best score
#print("Best parameters:", grid_search.best_params_)
#print("Best score:", grid_search.best_score_)

prediction = gbt_model.predict(X_test)

#print(classification_report(y_test, prediction,zero_division=1))
#print(confusion_matrix(y_test, prediction))


















def _ensemble_model(rf_model, knn_model, gbt_model,xgb_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    estimators=[('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model),('Xgb',xgb_model)]
    
    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft')
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    #print(ensemble.score(X_test, y_test))
    
    #prediction = ensemble.predict(X_test)

    #print(classification_report(y_test, prediction,zero_division=1))
    #print(confusion_matrix(y_test, prediction))
    
    return ensemble
    
ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model,xgb_model, X_train, y_train, X_test, y_test)



# Get the current date and time
now = datetime.datetime.now()

# Format the date and time as yymmddhhmmss
fecha = now.strftime("%y%m%d%H%M%S")

# Define the path where you want to save the model
path = r'Alumno\3-Machine_Learning\Entregas\ML_project\src\model'

# Save the model in a file named model_fecha in the specified path
with gzip.open(os.path.join(path, "model_{}.pickle.gz".format(fecha)), 'wb') as f:
    pickle.dump(ensemble_model, f)

if not os.path.exists(path):
    os.makedirs(path)
















