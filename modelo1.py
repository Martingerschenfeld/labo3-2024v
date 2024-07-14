# Importamos las librerias necesarias
import os
import pandas as pd
import numpy as np
import functions

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

import warnings

warnings.filterwarnings("ignore")

producto_final = []
pred_final = []

# Accedemos a los datos
ventas = pd.read_csv("datos/sell-in.txt", sep="\t")
prod_a_pred = pd.read_csv("datos/prod_a_pred.txt", sep="\t")

# Filtro los productos a predecir
ventas = ventas[ventas["product_id"].isin(list(prod_a_pred["product_id"]))].reset_index(
    drop=True
)
products_id = ventas["product_id"].unique()

"""
Armamos un diccionario que nos va a indicar para cada familia que productos incluyo
Luego con ese diccionario armamos un diccionario que nos indica para cada producto a que familia pertenece
"""

directorio = "familias"
archivos_csv = sorted([f for f in os.listdir(directorio) if f.endswith(".csv")])
familia_productos = {}
for archivo in archivos_csv:
    ruta_completa = os.path.join(directorio, archivo)
    df = pd.read_csv(ruta_completa, index_col=0)
    productos = [col.split("_")[-1] for col in df.columns if col.startswith("ventas_")]
    familia_productos[archivo] = productos

producto_familia = {}
for archivo in familia_productos:
    for producto in familia_productos[archivo]:
        producto_familia[producto] = archivo

intento = 1

"""   
-Recorremos cada producto y armamos los distintos dataframes para el training y la validación.
-Eliminamos el primer valor, ya que el diferencial de ventas va a ser NaN al no existir. 
-Los valores de X training son los valores previo a la fecha_limit ('2019-12-01'), mientras que el de validación es la fecha_limit
-Escalamos los valores usando el RobustScaler()
-Armamos el modelo pidiendo la predicción a dos meses.
 Van a haber tres casos:
 1) Cantidad de edad del producto menor o igual a 6 meses
 2) Cantidad de edad del producto mayor a 6 meses
 3) Predicción negativa
- En el caso 1) Vamos a hacer una regresión lineal con los pocos datos que hay e imputar la predicción
- En el caso 2) Vamos a usar un LSTM
- En el caso 3) Vamos a utilizar el promedio de los 3 últimos valores.
"""

for producto in products_id:
    csv = producto_familia[str(producto)]
    ruta_completa = os.path.join(directorio, csv)
    df = pd.read_csv(ruta_completa, index_col=0)
    cols_start_with_dif = [col for col in df.columns if col.startswith('diff')]
    df = df.dropna(subset=cols_start_with_dif)
    
    fecha_limit = '2019-12-01'
    X_train, y_train, X_val, y_val = functions.split_data(df, fecha_limit, producto)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1,1))
    y_val_scaled = scaler.transform(y_val.values.reshape(-1,1))
    
    n_forecast = 2
    
    if max(df[f'edad_ventas_{producto}']) >  6:
        n_lookback = 18
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        
        model_file_path = f"Models_params/{intento}/model_product_{producto}.keras"
        if os.path.exists(model_file_path):
            model = load_model(model_file_path)
            print(f"Modelo para el producto {producto} cargado.")
        else:
            # Armado del modelo
            model = Sequential([
                LSTM(50, activation = 'tanh', kernel_regularizer=l2(1e-6), input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True, kernel_initializer='he_uniform'),
                Dropout(0.1),
                #BatchNormalization(),
                LSTM(20, return_sequences=False),
                Dropout(0.1),
                #BatchNormalization(),
                Dense(n_forecast)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val_scaled))
            os.makedirs("Models_params", exist_ok=True)
            os.makedirs(f"Models_params/{intento}", exist_ok=True)
            model.save(model_file_path)
            print(f"Modelo para el producto {producto} entrenado y guardado.")
        # Predicción para noviembre 2019
        pred_val_scaled = model.predict(X_val_scaled)
        pred_val = scaler.inverse_transform(pred_val_scaled)  # Desescalar
        future_predictions = pred_val[0][1]
    else:
        n_lookback = 3    
        # Crear y entrenar un modelo de regresión lineal en una línea, usando los últimos 4 valores de 'columna_deseada'
        modelo_lineal = LinearRegression().fit(np.arange(n_lookback).reshape(-1, 1), df[f'ventas_{producto}'].tail(n_lookback))
        # Crear índices para las predicciones futuras desde n hasta n+N_forecast-1
        future_indices = np.arange(n_lookback, n_lookback + n_forecast).reshape(-1, 1)
        # Hacer predicciones utilizando el modelo
        future_predictions = modelo_lineal.predict(future_indices)
        future_predictions = future_predictions[1]

    if future_predictions < 0:
        future_predictions = df[f'ventas_{producto}'].tail(n_lookback).mean()
        
    pred_final.append(future_predictions)
    producto_final.append(producto)
        
    print("Predicción para Diciembre 2019:", future_predictions)
    
data = {'product_id':producto_final,
    'tn':pred_final}

df_final = pd.DataFrame(data)

df_final.to_csv('pred_final.csv')

