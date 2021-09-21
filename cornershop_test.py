# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 00:44:30 2021

@author: phili
"""

import pandas as pd
import pandas_datareader as web
import tensorflow as tf
import numpy as np

from numpy import asarray
from numpy import savetxt

import matplotlib.pyplot as plt
import cv2
import folium
import seaborn as sn
import webbrowser
import geopy.distance
import datetime as dt
import time
from math import sin, cos, sqrt, atan2, radians
from sklearn import linear_model
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import f_regression
from keras.optimizers import RMSprop
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree


#%% exploratory analysis


def analisisExploratorio():
    
    
    data = pd.read_csv('orderFinal.csv')

    print("data shape:  ", data.shape,"----")
    
    print(data.info())


    #% limpieza de datos
    
    
    print("null Values:")

    print(data.isnull().sum())
    
    
    # replace data with median values 
    
    
    data['accepted_rate'].fillna(data['accepted_rate'].median(skipna=True), inplace=True)
    
    data['rating'].fillna(data['rating'].median(skipna=True), inplace=True)
    
    data['found_rate'].fillna(data['found_rate'].median(skipna=True),inplace=True) 
    

    print("null Values:")
    
    print(data.isnull().sum())
    
    print("order_id")
    print(data['order_id'])
    
    
 
    
    
    del data['store_id']
    
#    del data['order_id']
#    
    
    
    # clean data in case there are time values under 0


    data.loc[data['total_minutes']<0, ['total_minutes']] = np.nan 
    
#    data.loc[data['total_minutes']>300, ['total_minutes']] = np.nan 
    
    
    # delete values wwhere are not total_time values

    
    data = data.dropna(axis=0, how='any')
    
    
    print("")
    
    print("null Values:")
    
    print("")
    
    print(data.isnull().sum())
    
 

        
    # convert lat and lng into distances so it is possible to calculate the distance between delivery and order
    
    
    def haversine_vectorize(lon1, lat1, lon2, lat2):
        
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        newlon = lon2 - lon1
        
        newlat = lat2 - lat1

        haver_formula = np.sin(newlat/2.0)**2+ np.cos(lat1)* np.cos(lat2)* np.sin(newlon/2.0)**2
        
        dist = 2 * np.arcsin( np.sqrt(haver_formula) )
        
        km = 6367 * dist #6367 for distance in KM for miles use 3958
        
        return km
    
    
    data['distance_km'] = haversine_vectorize(data["lng_store"], 
        data["lat_store"], 
        data["lng_order"], 
        data["lat_order"]) 
    
    
    # create new dataset with distance included and delete distnaces in case are negative
    
    
    data.loc[ data['distance_km']<0,['distance_km']] = np.nan
    
    data = data.dropna(axis=0, how='any')
    
    
    # open a webbrowser to image map with coordinates 
    
    
    mapa=folium.Map(location=[-33.5017, -70.5794], zoom_start = 12) 
    
    folium.Marker(location=[-33.5017, -70.5794], popup='Default popup Marker1', tooltip='Click here to see Popup').add_to(mapa)
    
    mapa.save("mymap.html")
    
    #webbrowser.open('mymap.html', new=2)  # open in new tab
    
    
    # transform to categorical  so seniority can be classified
    
        
    data['seniority'] = data['seniority'].astype('category')
    
    data['seniority'] = data['seniority'].cat.codes
    
    data['seniority2'] = data['seniority']
    
    
#    data2= data.copy()
#    data = data2.copy()
#    data=pd.get_dummies(data,columns=['seniority'],drop_first =False)
    
    
    
    data['shopper_id'] = data['shopper_id'].astype('category')
    
    data['shopper_id'] = data['shopper_id'].cat.codes
    
    
    data['store_branch_id'] = data['store_branch_id'].astype('category')
    
    data['store_branch_id'] = data['store_branch_id'].cat.codes
    
    
    # plot to visualize data
    
    
    plt.figure()
    
    plt.title("On_Demand Histogram")
    
    data['on_demand'].value_counts().sort_index().plot.bar(figsize=(10, 6), color='.3')
    
    plt.xlabel("on_demand")
    
    plt.grid()
    
    plt.show()
    
    
    
    plt.figure()
    
    plt.title('seniority')
    
    data['seniority'].value_counts().sort_index().plot.bar(figsize=(10, 6),color = '.3')
    
    plt.xlabel('seniority/Xperience')
    
    plt.grid()
    
    
    
    
    plt.figure()
    
    plt.title("--- Analysis of total minutes to deliver ---")
    
    sn.boxplot(x="total_minutes", data=data, color=".3")
    
    sn.stripplot(x='total_minutes',data=data, color="orange")
    
    plt.grid()
    
    plt.show()
    
    
    # change data time type to ns[64]
    
    
    data['promised_time'] = pd.to_datetime(data['promised_time'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize(None)
    
    data['promised_time'] = pd.to_datetime(data['promised_time'], format="%H:%M:%S").dt.tz_localize(None)
    
    
    # bin time to catogories to split with period of the day in 4 time zones
    
    
    #bins1 = [0, 8, 16, 24]
    
    #labels1 = ['24hr - 08hr', '08hr - 16hr', '16hr -24hr']
    
    bins2 = [0, 6, 12, 18, 24]
    
    labels2 = ['24hr - 06hr', '06hr - 12hr', '12hr -18hr', '18hr-24hr']
    
    #bins3 = [0, 12, 24]
    
    #labels3 = ['24hr - 12hr', '12hr - 24hr']
    
    
    res = pd.cut(data['promised_time'].dt.hour, bins2, labels = labels2, right=False)
    
    data["Day_section"] = res
    
    
    # categoorize day time in parts  24-06-12-18 hrs
    
    
    data['Day_section2'] = data['Day_section'].astype('category')
    
    data['Day_period'] = data['Day_section2'].cat.codes
    
    
    # create dummy variables for categories
    
    
    data = pd.get_dummies(data, columns=['Day_period'], drop_first = False)
        
    data = pd.get_dummies(data, columns=['seniority'], drop_first = False)
    
    data['Day_category'] = data['Day_section2'].cat.codes  # extra categoriry to use in RegressionModels():



    plt.figure()
    
    plt.title('orders during Day/Night ')
    
    data['Day_section2'].value_counts().sort_index().plot.bar(figsize = (10, 6),color='.3')
    
    plt.xlabel('Day_section')
    
    plt.grid()
    
    
    # classify coordinates in groups, separte santiago in 4 areas to avoid overdimentionality
    
    
    cut_points1 = data['lat_order'].quantile([0,0.25,0.5,0.75,1.0])
    
    cut_points2 = data['lng_order'].quantile([0,0.25,0.5,0.75,1.0])
    
    
    labels1= [0, 1, 2, 3]
    
    labels2= [0, 1, 2, 3]



    data['lat_order_b'] = pd.cut(data['lat_order'], bins=cut_points1, labels=labels1, include_lowest=True) 
    
    data['lng_order_b'] = pd.cut(data['lng_order'], bins=cut_points2, labels=labels2, include_lowest=True)
    
    data['lat_store_b'] = pd.cut(data['lat_store'], bins=cut_points1, labels=labels1, include_lowest=True)
    
    data['lng_store_b'] = pd.cut(data['lng_store'], bins=cut_points2, labels=labels2, include_lowest=True)
    
    
    
#        
#    data=pd.get_dummies(data,columns=['lat_order_b'],drop_first =False)
#    
#    data=pd.get_dummies(data,columns=['lng_order_b'],drop_first =False)
#    
#    data=pd.get_dummies(data,columns=['lat_store_b'],drop_first =False)
#    
#    data=pd.get_dummies(data,columns=['lng_store_b'],drop_first =False)
    
    
    

    plt.figure()
    
    plt.title("Distribución Geográfica de ordenes y Tiendas en Santiago")
    
    plt.scatter(data['lat_order'], data['lng_order'])
    
    plt.scatter(data['lat_store'], data['lng_store'])
    
    plt.grid()
    
    plt.show()
    
   # calculate the time to get all products
    
    data['time_per_products'] = data['picking_speed']*data['productosdistintos']
    
    plt.figure()
    
    plt.title("--- picking produts up time ---")
    
    sn.boxplot(x='time_per_products', data = data, color=".3")
    
    sn.stripplot(x='time_per_products',data = data, color="cyan")
    
    plt.grid()
    
    plt.show()
    
    
    
    print(" --- Max and Min Values of picking products time ---")
    
    print("")
    
    print(data['time_per_products'].max(), 'max time')
    
    print(data['time_per_products'].min(), 'min time')
    
    
    # delete unnecessary data
    
    
    del data['accepted_rate']
    
    del data['lat_order']
    
    del data['lng_order']
    
    del data['lat_store']
    
    del data['lng_store']
    
    del data['Day_section']
    
    del data['Day_section2']
    
    del data['on_demand']  
    
    del data['promised_time']
    
    del data['shopper_id']
    
    del data['store_branch_id']
    
    del data['time_per_products']
    
    del data['lat_order_b']  
    
    del data['lng_order_b']
    
    del data['lat_store_b']
    
    del data['lng_store_b'] 
  
    #del data['Day_period']
    
    
    # split data and make correlations matrix
    
    
    order_id = data['order_id']
    
    
    data= data.drop('order_id', axis=1)
    
    x = data.drop('total_minutes', axis=1)

    
#    print("----- infoooo---- ")
#    print(x.info())
    
    y = data['total_minutes']
    
    print(y)
    
    corrmat = x.corr().abs()
    
    fx,ar= plt.subplots(figsize=(15, 10))
    
    sn.heatmap(corrmat, square=True)
    
    plt.show()
   
    
    # see more important coeffecients
    

    
    
    x_new = f_regression(x, y, center=True)
    
    print(list(x_new))
    
    X_new=SelectKBest(score_func=f_regression, k='all').fit(x, y)
    
    X_final = X_new.fit_transform(x, y)
    
    X_final.shape
    
    kBest = np.asarray(x.columns)[X_new.get_support()]   
    
    print(kBest)
    

    return data , order_id 




def NeuralRegression():
    
    
    data = DatosProcesados
    
    x  = data.drop('total_minutes', axis=1)
    
    y =  data['total_minutes']
    
    
    N =  np.array(DatosProcesados.drop('total_minutes', axis = 1))

    selected_features = ['picking_speed', 
                         'productosdistintos', 
                         'UNIDADES', 
                         'KG',
                         'rating',
                         'distance_km', 
                         'seniority_0', 
                         'seniority_1', 
                         'seniority_2', 
                         'seniority_3','','','']
    
    
    xdata = x  #data[selected_features]
    
    ydata = y  #data['total_minutes']
    


    scaler = MinMaxScaler()
    
    xdata_scaled = scaler.fit_transform(xdata)
    
    ydata = ydata.values.reshape(-1, 1)
    
    ydata_scaled = scaler.fit_transform(ydata)
    
    

    X_train,X_test,Y_train,Y_test = train_test_split(xdata_scaled,ydata_scaled,test_size=0.25)


    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(units=100, kernel_initializer='normal', activation='relu', input_shape=(N.shape[1], )))
    
    model.add(tf.keras.layers.Dense(units=100, kernel_initializer='normal', activation='relu'))
    
    model.add(tf.keras.layers.Dense(units = 100, kernel_initializer='normal', activation='relu'))
    
    model.add(tf.keras.layers.Dense(units = 100, kernel_initializer='normal', activation = 'relu'))
    
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))
  
    model.compile(optimizer='Adam', loss='mean_squared_error')


    epochs_hist=model.fit(X_train, Y_train, epochs=100, batch_size=50, validation_split=0.1)
    
    epochs_hist.history.keys()
    

    print(epochs_hist.history.keys())
    
    model.summary()
    
    

    plt.figure()
    
    plt.title("entrenamiento")
    
    plt.plot(epochs_hist.history['loss']) 
    
    plt.plot(epochs_hist.history['val_loss'])
    
    plt.xlabel("epochs")
    
    plt.ylabel("training and validation loss")
    
    plt.grid()
    
    
    plt.show()
    
    
    
    y_predictions1 = model.predict(xdata_scaled)
    
    y_predictions1 = scaler.inverse_transform(y_predictions1)
    
    
    
    y_pred = model.predict(X_test)
    
    score = r2_score(Y_test, y_pred)
    

    print("score:")
    
    print(score)

    a = np.array(ydata)
    
    b = np.array(y_predictions1)
    
    nn= np.hstack([a,b])
    
    Neural_preds = (np.vstack([ydata, y_predictions1 ])).T
   
    return nn





def RegressionModels():
    
    
    DatosProcesados
    
    numeric_list = ['found_rate',
                    'rating', 
                    'productosdistintos', 
                    'picking_speed', 
                    'UNIDADES', 
                    'KG', 
                    'distance_km']
    
    categoric_list = ['seniority2', 
                      'Day_category']
    
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    categoric_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    
    transformer = ColumnTransformer([('num', numeric_transformer, numeric_list), ('cat', categoric_transformer, categoric_list)])
    
    
#    print(transformer)
  
            
    Xdata  = DatosProcesados.drop('total_minutes', axis = 1)
#    Xdata  = DatosProcesados.drop('order_id', axis = 1)
    
    Ydata =  DatosProcesados['total_minutes']
    
    Ydata2 = np.array(Ydata)
    
    
    scaler = MinMaxScaler()
    
    xdata_scaled = scaler.fit_transform(Xdata)
    
    Ydata2 = Ydata2.reshape(-1, 1)
    
    ydata_scaled = scaler.fit_transform(Ydata2)
    

    X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, train_size=0.7)
    
    
    # just total_minutes values
    
    
    Tiempos = np.array(Ydata)
    
    
    # regresion lineal Regression
    
    
    linear_regression = LinearRegression()
    
    model_linear_regression = Pipeline([('transformer', transformer), ('linear_regression', linear_regression)])
    
    model_linear_regression.fit(X_train, Y_train)

    prediction_linear_regression = np.array(model_linear_regression.predict(Xdata))
    
    lin_reg_preds = (np.vstack([Tiempos, prediction_linear_regression])).T
    
    print("lin_reg_preds:")
    
    print(prediction_linear_regression)
    
    
    lin_scores = cross_val_score(model_linear_regression, Xdata, Ydata, scoring ='neg_mean_squared_error', cv=10)
    
    lin_mre = np.sqrt(-lin_scores)

    
    # decision tree Regressor
    
    
    tree_reg = DecisionTreeRegressor(max_depth=8)
    
    model_tree_reg = Pipeline(steps = [('transformer', transformer), ('tree_reg', tree_reg)] )    
    
    model_tree_reg.fit(X_train, Y_train)
    
    prediction_decision_Tree = np.array(model_tree_reg.predict(Xdata))
    
    prediction_decision_tree_regression = np.array(model_tree_reg.predict(Xdata))  # prediction
    
    tree_reg_preds = ( np.vstack([ Tiempos, prediction_decision_tree_regression])).T
    
    print("lin_reg_preds:")
    
    print(prediction_decision_tree_regression)
    
    tree_scores = cross_val_score(model_tree_reg, Xdata, Ydata, scoring ='neg_mean_squared_error', cv=10)
    
    tree_mre = np.sqrt(-tree_scores)
    
    
    # Random Forest Regressor 
    

    Random_reg = RandomForestRegressor(random_state=10)
    
    model_Random_reg = Pipeline(steps=[('transformer', transformer),('tree_reg', Random_reg)]) 
    
    model_Random_reg.fit(X_train, Y_train)
    
    model_Random_reg.predict(Xdata)
    
    prediction_random_forest = np.array(model_Random_reg.predict(Xdata))
    
    Random_reg_preds = (np.vstack([Tiempos, prediction_random_forest])).T
    
    
    print("Random_reg_preds:")
    
    print(prediction_random_forest)
    
    Random_scores = cross_val_score(model_Random_reg, Xdata, Ydata, scoring ='neg_mean_squared_error', cv = 10)
    
    Random_mre = np.sqrt(-Random_scores)



    def display_scores(score):
        
        
        print( "scores :", score)
        
        print( "scores mean:", score.mean())
    
        print( "scores std:" ,score.std())
        
    
    score_1 = display_scores(lin_mre)
    
    score_2 = display_scores(tree_mre)
    
    score_3 = display_scores(Random_mre)
     
    
#    n_estimators=[3, 10, 30, 50, 80, 100]
#    
#    min_samples_split=[2, 4, 6, 8, 10]
#    
#    max_depth=[2, 4, 6, 8, 10]
#    
#    params={'forest_reg__n_estimators':n_estimators, 'forest_reg__min_samples_split':min_samples_split, 'forest_reg__max_depth':max_depth}
#    
#    print(params)

#    grids = GridSearchCV(Random_reg,param_grid=params,cv=10,scoring='neg_mean_squared_error',n_jobs=-1,verbose=2) 
#    grids.fit(Xdata,Ydata)
    
#    All_preds = (np.vstack([ydata, lin_reg_preds, tree_reg_preds,Random_reg_preds])).T
    
    
    return Ydata,prediction_linear_regression, prediction_decision_tree_regression, prediction_random_forest
    
    







if __name__ == '__main__':
    

    DatosProcesados = analisisExploratorio()[0]
    orders_id = analisisExploratorio()[1]
    NN_preds = NeuralRegression()
    final_predictions_regressions = RegressionModels()
#    
    
    All_preds = (np.vstack([orders_id,
                            final_predictions_regressions[0], 
                            final_predictions_regressions[1], 
                            final_predictions_regressions[2],
                            final_predictions_regressions[3],
                            NN_preds[:,1]])).T
        
    df_preds = pd.DataFrame(data=All_preds, columns=['order_id'
                                                     ,'total_time', 
                                                     "linear_preds", 
                                                     "decisionTree_preds", 
                                                     "RandonForest_preds",
                                                     'NN_preds'])
        
#    df_preds.to_csv('All_predicions.csv',index=False)
#    
    print("")
#    
    print("predictions saved in file 'All_predictions' ")

