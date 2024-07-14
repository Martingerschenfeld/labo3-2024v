Modelo seleccionado utilizando LSTM

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
