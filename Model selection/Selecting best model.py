# -*- coding: utf-8 -*-

##### SELECT model #####

### Import packages ###

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

### Import data ###

df = pd.read_csv("""Path here""")
df = df[["""Variables here"""]]
df.relative_units = df.relative_units.astype("int")
df.total_units = df.total_units.astype("int")
df.warehouse = df.warehouse.astype("str")

##### Moving Average model #####

def predict_simple_avg(df, predict_col = 'dist', window = 12, metric = 'MAE'):
    periods = df.shape[0]
    train = df.iloc[0:window]
    test = df.iloc[window:]
    rolling = train[predict_col].values.astype(float)

    delta = []
    pred = []
    act = []
    tot = []
    abs_delta = []
    sqr_delta = []
    mape_delta = []
    for i in range(0,periods-window):
        avg = np.mean(rolling)

        if len(rolling) == window:
            rolling = rolling[1:]
        rolling = np.append(rolling,test.iloc[i].dist)
        actual = test.iloc[i].dist
        total = test.iloc[i].total_units

        pred.append(avg)
        act.append(actual)
        delta.append(avg-actual)
        abs_delta.append(np.abs(avg-actual))
        tot.append(total)
        sqr_delta.append((avg-actual)**2)
        mape_delta.append((avg-actual)/actual)

    MAE = {"prediction":np.mean(pred),"delta":np.mean(abs_delta),"actual":np.mean(act)}
    RMSE = {"prediction":np.mean(pred),"delta":np.sqrt(np.mean(sqr_delta)),"actual":np.mean(act)}
    MAPE = {"prediction":np.mean(pred),"delta":np.mean(mape_delta)*100,"actual":np.mean(act)}
    if metric == 'MAE':
        return pred, act, delta, tot, MAE
    elif metric == 'RMSE':
        return pred, act, delta, tot, RMSE
    elif metric == 'MAPE':
        return pred, act, delta, tot, MAPE

##### Linear Regression model #####
        
def predict_linear_regression(df, predict_col = 'dist', window=12, metric = 'MAE'):
    periods = df.shape[0]
    train = df.iloc[0:window]
    test = df.iloc[window:]
    rolling = train[predict_col].values.astype(float)
    
    delta = []
    pred = []
    act = []
    tot = []
    abs_delta = []
    sqr_delta = []
    mape_delta = []
    for i in range(0,periods-window):
        regressor = LinearRegression()
        regressor.fit(np.array(range(len(train))).reshape(-1, 1), rolling)
        forecast = float(regressor.predict(np.array(range(len(train)+i, len(train)+i+1)).reshape(-1, 1)))

        if len(rolling) == window:
            rolling = rolling[1:]
        rolling = np.append(rolling,test.iloc[i].dist)
        actual = test.iloc[i].dist
        total = test.iloc[i].total_units

        pred.append(forecast)
        act.append(actual)
        delta.append(forecast-actual)
        abs_delta.append(np.abs(forecast-actual))
        tot.append(total)
        sqr_delta.append((forecast-actual)**2)
        mape_delta.append((forecast-actual)/actual)

    MAE = {"prediction":np.mean(pred),"delta":np.mean(abs_delta),"actual":np.mean(act)}
    RMSE = {"prediction":np.mean(pred),"delta":np.sqrt(np.mean(sqr_delta)),"actual":np.mean(act)}
    MAPE = {"prediction":np.mean(pred),"delta":np.mean(mape_delta)*100,"actual":np.mean(act)}
    if metric == 'MAE':
        return pred, act, delta, tot, MAE
    elif metric == 'RMSE':
        return pred, act, delta, tot, RMSE
    elif metric == 'MAPE':
        return pred, act, delta, tot, MAPE

##### Auto Arima model #####
    
def predict_arima_auto(df, predict_col = 'dist', window = 12, metric = 'MAE'):                   
    periods = df.shape[0]
    train = df.iloc[0:window]
    test = df.iloc[window:]
    rolling = train[predict_col].values.astype(float)
    
    delta = []
    pred = []
    act = []
    tot = []
    abs_delta = []
    sqr_delta = []
    mape_delta = []
    for i in range(0,periods-window):
        try:
            arima_order = auto_arima(rolling).order
            model = ARIMA(np.array(rolling), order=arima_order)
            model_fit = model.fit(disp=0)
            forecast = float(model_fit.forecast(1)[0])
        except:
            model = ARIMA(np.array(rolling), order=(0,0,0))
            model_fit = model.fit(disp=0)
            forecast = float(model_fit.forecast(1)[0])

        if len(rolling) == window:
            rolling = rolling[1:]
        rolling = np.append(rolling,test.iloc[i].dist)
        actual = test.iloc[i].dist
        total = test.iloc[i].total_units

        pred.append(forecast)
        act.append(actual)
        delta.append(forecast-actual)
        abs_delta.append(np.abs(forecast-actual))
        tot.append(total)
        sqr_delta.append((forecast-actual)**2)
        mape_delta.append((forecast-actual)/actual)

    MAE = {"prediction":np.mean(pred),"delta":np.mean(abs_delta),"actual":np.mean(act)}
    RMSE = {"prediction":np.mean(pred),"delta":np.sqrt(np.mean(sqr_delta)),"actual":np.mean(act)}
    MAPE = {"prediction":np.mean(pred),"delta":np.mean(mape_delta)*100,"actual":np.mean(act)}
    if metric == 'MAE':
        return pred, act, delta, tot, MAE
    elif metric == 'RMSE':
        return pred, act, delta, tot, RMSE
    elif metric == 'MAPE':
        return pred, act, delta, tot, MAPE

##### GBM model #####

### Creating features ###
        
def add_lags(data,groupby,order_field,lags=[],columns=[],suffix='_lag_'):
    sort_order = [x for x in groupby]
    sort_order.append(order_field)
    
    _data = data.copy().sort_values(sort_order)

    grouped = _data.groupby(groupby)
    
    for i in lags:
        lagged = grouped.shift(i)
        for c in columns:
            series = lagged[[c]].rename(columns={c:c+suffix+str(i)})
            _data = pd.concat([_data,series],axis=1)
    return _data
                
def rolling_window_not_including(data,groupby,order_field,windows=[],columns=[],fn="mean",suffix='_window_'):
    sort_order = [x for x in groupby]
    sort_order.append(order_field)
    
    _data = data.copy().sort_values(sort_order)
    
    for c in columns:
        for w in windows:
            groupby_df = [data[x] for x in groupby]
            g = _data[c].groupby(groupby_df).shift(1).rolling(w)
            if fn=="mean":
                _data[c+suffix+fn+"_"+str(w)]= g.mean()
            if fn=="std":
                _data[c+suffix+fn+"_"+str(w)]= g.std()
    return _data
            
def rolling_window(data,groupby,order_field,windows=[],columns=[],fn="mean",suffix='_window_'):
    sort_order = [x for x in groupby]
    sort_order.append(order_field)
    
    _data = data.copy().sort_values(sort_order)
    grouped = _data.groupby(groupby)

    for w in windows:
        print("calculating window: ",str(w))
        r = grouped.rolling(w)[columns]
        if fn == "mean":
            curr = r.mean().reset_index()
        if fn == "std":
            curr = r.std().reset_index()
       
        rename_dic = {}
        for c in columns:
            rename_dic[c]  = c+suffix+fn+"_"+str(w)
        curr.rename(columns=rename_dic,inplace=True)
        curr[order_field] = np.array(_data[order_field].values)
        _data = pd.merge(_data,curr,how="left",on=sort_order)

    return _data

### Split training and testing set for GBM ###

def create_training_testing(df, lags = [1,2,3,4,5,6], windows = [1,2,3,4,5,6], validation = 12):
    data = add_lags(df, ["""Groupby variables"""], """Order field variable""", lags,
                    columns=["""Other variables"""])
    data = rolling_window_not_including(data, ["""Groupby variables"""], """Order field variable""", 
                                        windows, columns=["""Other variables"""])
    
    cols_delete = []
    for c in list(data.columns):
        if c.find("level") > -1:
            cols_delete.append(c)
    
    data = data.drop(cols_delete,axis=1)
    data['warehouse'] = data.warehouse.astype(str)
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in data.master_sku.unique():
        test_week = sorted(data[data.master_sku == i].ds_week.unique())[-validation:]
        train_week = sorted(data[data.master_sku == i].ds_week.unique())[:-validation]
        train_data = data[data.ds_week.isin(train_week)][data.master_sku == i]
        test_data = data[data.ds_week.isin(test_week)][data.master_sku == i]
        df_train = df_train.append(train_data)
        df_test = df_test.append(test_data)
        
    return df_train, df_test

### Prediction ###

def predict_gbm_single(df, predict_col = 'dist', window = 12, metric = 'MAE', tuning_hyperparameters = True):
    df_train, df_test = create_training_testing(df, lags = [1,2,3,4,5,6], windows = [1,2,3,4,5,6], validation = window)
    X = df_train.drop([predict_col], axis = 1)
    y = df_train[predict_col]
    X_test = df_test.drop([predict_col], axis = 1)
    y_test = df_test[predict_col]
    
    X.master_sku = X.master_sku.astype('category').cat.codes
    X.warehouse = X.warehouse.astype('category').cat.codes
    X.ds_week = X.ds_week.astype('category').cat.codes
    X = X.fillna(0)
    y = y.fillna(0)

    X_test.master_sku = X_test.master_sku.astype('category').cat.codes
    X_test.warehouse = X_test.warehouse.astype('category').cat.codes
    X_test.ds_week = X_test.ds_week.astype('category').cat.codes
    X_test = X_test.fillna(0)
    y_test = y_test.fillna(0)
    
    if tuning_hyperparameters == True:
        grid = GridSearchCV(
                estimator=ensemble.GradientBoostingRegressor(),
                param_grid={
                        'learning_rate': [0.1, 0.01, 0.001],
                        'max_depth': [3, 5, 7, 9],
                        'n_estimators': [30, 60, 90],
                        'subsample': [0.6, 0.8, 1],
                        'min_samples_split': [2, 3, 4, 5, 6]
                        },
                cv=8, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)

        grid_result = grid.fit(X, y)
        gbm_reg = ensemble.GradientBoostingRegressor(learning_rate = grid_result.best_params_['learning_rate'],
                                                     max_depth = grid_result.best_params_['max_depth'],
                                                     n_estimators = grid_result.best_params_['n_estimators'], 
                                                     subsample = grid_result.best_params_['subsample'], 
                                                     min_samples_split = grid_result.best_params_['min_samples_split'])
    else:
        gbm_reg = ensemble.GradientBoostingRegressor()
    
    gbm_reg.fit(X, y)
    preds = gbm_reg.predict(X_test)

    X_test['actual'] = y_test
    X_test = X_test.reset_index().iloc[:,1:]
    X_test['preds'] = pd.DataFrame(preds)
    X_test['delta'] = X_test['preds'] - X_test['actual']
    X_test['week'] = df_test.reset_index()['ds_week']
    X_test['master_sku'] = df_test.reset_index()['master_sku']
    X_test['warehouse'] = df_test.reset_index()['warehouse']
    
    X_test['preds'][X_test['preds']<0]  = 0
    X_test['abs_delta'] = abs(X_test.actual-X_test.preds)
    X_test['abs_relative'] = abs((X_test.actual-X_test.preds)/X_test.actual)*100
    X_test['sqr_relative'] = (X_test.actual-X_test.preds)**2

    MAE = {"prediction":np.mean(X_test['preds']),"delta":np.mean(X_test['abs_delta']),"actual":np.mean(X_test['actual'])}
    RMSE = {"prediction":np.mean(X_test['preds']),"delta":np.sqrt(np.mean(X_test['sqr_relative'])),"actual":np.mean(X_test['actual'])}
    MAPE = {"prediction":np.mean(X_test['preds']),"delta":np.mean(X_test['abs_relative']),"actual":np.mean(X_test['actual'])}
    if metric == 'MAE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], MAE
    elif metric == 'RMSE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], RMSE
    elif metric == 'MAPE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], MAPE    
    
### Pre-trained model using all traing data ###
   
def all_data_GBM(data, predict_col = 'dist', window = 12, tuning_hyperparameters = True):
    df_train, df_test = create_training_testing(data, lags = [1,2,3,4,5,6], windows = [1,2,3,4,5,6], validation = window)
    X = df_train.drop([predict_col], axis = 1)
    y = df_train[predict_col]
    X_test = df_test.drop([predict_col], axis = 1)
    y_test = df_test[predict_col]
    
    X.master_sku = X.master_sku.astype('category').cat.codes
    X.warehouse = X.warehouse.astype('category').cat.codes
    X.ds_week = X.ds_week.astype('category').cat.codes
    X = X.fillna(0)
    y = y.fillna(0)

    X_test.master_sku = X_test.master_sku.astype('category').cat.codes
    X_test.warehouse = X_test.warehouse.astype('category').cat.codes
    X_test.ds_week = X_test.ds_week.astype('category').cat.codes
    X_test = X_test.fillna(0)
    y_test = y_test.fillna(0)
    
    if tuning_hyperparameters == True:
        grid = GridSearchCV(
                estimator=ensemble.GradientBoostingRegressor(),
                param_grid={
                        'learning_rate': [0.1, 0.01, 0.001],
                        'max_depth': [3, 5, 7, 9],
                        'n_estimators': [30, 60, 90],
                        'subsample': [0.6, 0.8, 1],
                        'min_samples_split': [2, 3, 4, 5, 6]
                        },
                cv=8, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)

        grid_result = grid.fit(X, y)
        gbm_reg = ensemble.GradientBoostingRegressor(learning_rate = grid_result.best_params_['learning_rate'],
                                                     max_depth = grid_result.best_params_['max_depth'],
                                                     n_estimators = grid_result.best_params_['n_estimators'], 
                                                     subsample = grid_result.best_params_['subsample'], 
                                                     min_samples_split = grid_result.best_params_['min_samples_split'])
    else:
        gbm_reg = ensemble.GradientBoostingRegressor()
    return gbm_reg.fit(X, y)
     
def pretrain_GBM(df, data, predict_col = 'dist', window = 12, metric = 'MAE', tuning_hyperparameters = True):
    df_train, df_test = create_training_testing(df, lags = [1,2,3,4,5,6], windows = [1,2,3,4,5,6], validation = window)
    X = df_train.drop([predict_col], axis = 1)
    y = df_train[predict_col]
    X_test = df_test.drop([predict_col], axis = 1)
    y_test = df_test[predict_col]
    
    X.master_sku = X.master_sku.astype('category').cat.codes
    X.warehouse = X.warehouse.astype('category').cat.codes
    X.ds_week = X.ds_week.astype('category').cat.codes
    X = X.fillna(0)
    y = y.fillna(0)

    X_test.master_sku = X_test.master_sku.astype('category').cat.codes
    X_test.warehouse = X_test.warehouse.astype('category').cat.codes
    X_test.ds_week = X_test.ds_week.astype('category').cat.codes
    X_test = X_test.fillna(0)
    y_test = y_test.fillna(0)    
    
    gbm_reg = all_data_GBM(data, predict_col = predict_col, window = window, tuning_hyperparameters = tuning_hyperparameters)
    preds = gbm_reg.predict(X_test)

    X_test['actual'] = y_test
    X_test = X_test.reset_index().iloc[:,1:]
    X_test['preds'] = pd.DataFrame(preds)
    X_test['delta'] = X_test['preds'] - X_test['actual']
    X_test['week'] = df_test.reset_index()['ds_week']
    X_test['master_sku'] = df_test.reset_index()['master_sku']
    X_test['warehouse'] = df_test.reset_index()['warehouse']
    
    X_test['preds'][X_test['preds']<0]  = 0
    X_test['abs_delta'] = abs(X_test.actual-X_test.preds)
    X_test['abs_relative'] = abs((X_test.actual-X_test.preds)/X_test.actual)*100
    X_test['sqr_relative'] = (X_test.actual-X_test.preds)**2

    MAE = {"prediction":np.mean(X_test['preds']),"delta":np.mean(X_test['abs_delta']),"actual":np.mean(X_test['actual'])}
    RMSE = {"prediction":np.mean(X_test['preds']),"delta":np.sqrt(np.mean(X_test['sqr_relative'])),"actual":np.mean(X_test['actual'])}
    MAPE = {"prediction":np.mean(X_test['preds']),"delta":np.mean(X_test['abs_relative']),"actual":np.mean(X_test['actual'])}
    if metric == 'MAE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], MAE
    elif metric == 'RMSE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], RMSE
    elif metric == 'MAPE':
        return X_test['preds'], X_test['actual'], X_test['delta'], X_test['total_units'], MAPE

##### Select model with smallest average absolute error ##### 

def select_model(df, predict_col = 'dist', window = 12, metric = 'MAE', tuning_hyperparameters = True):
    df = df.fillna(0)
    avg = predict_simple_avg(df, predict_col, window, metric)[-1]['delta']
    linear = predict_linear_regression(df, predict_col, window, metric)[-1]['delta']
    auto = predict_arima_auto(df, predict_col, window, metric)[-1]['delta']
    GBM = predict_gbm_single(df, predict_col, window, metric, tuning_hyperparameters)[-1]['delta']
    choice = min(avg, linear, auto, GBM)
    if choice == avg:
        model = 'avg'
    elif choice == auto:
        model = 'auto'
    elif choice == GBM:
        model = 'GBM'
    else:
        model = 'linear'
    return choice, model


