# Importamos las librerias necesarias
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Función que reemplaza los valores Nan intermedios (asumo que son errores en base de datos) por el promedio de sus valores anteriores y posteriores.
def replace_nan_with_mean(series):
    for i in range(1, len(series) - 1):  # Evitar el primer y último elemento
        if (
            pd.isna(series[i])
            and not pd.isna(series[i - 1])
            and not pd.isna(series[i + 1])
        ):
            series[i] = (series[i - 1] + series[i + 1]) / 2
    return series


# Accedemos a los datos
ventas = pd.read_csv("datos/sell-in.txt", sep="\t")
prod_a_pred = pd.read_csv("datos/prod_a_pred.txt", sep="\t")
prod_nombres = pd.read_csv("datos/prod_nombres.txt", sep="\t")

# Filtro los productos a predecir
ventas = ventas[ventas["product_id"].isin(list(prod_a_pred["product_id"]))].reset_index(
    drop=True
)
# Agrupo por período para tener el totalizado de tn por periodo y producto
ventas_agrupadas = ventas.groupby(["periodo", "product_id"]).sum().reset_index()
# Dentro del df de los productos por categoría y tamaño de envase, me quedo con los productos a predecir
prod_nombres_agrupados = prod_nombres[
    prod_nombres["product_id"].isin(list(prod_a_pred["product_id"]))
].reset_index(drop=True)
# Unifico los df en uno solo
ventas_agrupadas_plus = pd.merge(
    ventas_agrupadas, prod_nombres_agrupados, on="product_id", how="left"
)
# Determino las familias de los productos
familias_con_marca = (
    ventas_agrupadas_plus[["cat1", "cat2", "cat3", "brand"]]
    .drop_duplicates(subset=["cat1", "cat2", "cat3", "brand"])
    .sort_values(by=["cat1", "cat2", "cat3", "brand"])
    .reset_index(drop=True)
)

# Crear rango de fechas desde enero de 2017 hasta diciembre de 2019
date_range = pd.date_range(start="2017-01-01", end="2019-12-31", freq="MS")
formatted_dates = date_range.strftime("%Y%m")
formatted_dates = formatted_dates.astype(int)
df = pd.DataFrame(formatted_dates, columns=["periodo"])

# Marca de control
a = 0

# Trabajo por familia de productos. Una familia se define como cat_1, cat_2,cat_3 y brand. Siendo los miembros de esta flia los sku_size y product_id distintos que cumplen la condición anterior.
for i in range(len(familias_con_marca)):

    cat_1 = familias_con_marca.cat1[i]
    cat_2 = familias_con_marca.cat2[i]
    cat_3 = familias_con_marca.cat3[i]
    cat_4 = familias_con_marca.brand[i]

    df_interno = ventas_agrupadas_plus[
        (ventas_agrupadas_plus["cat1"] == cat_1)
        & (ventas_agrupadas_plus["cat2"] == cat_2)
        & (ventas_agrupadas_plus["cat3"] == cat_3)
        & (ventas_agrupadas_plus["brand"] == cat_4)
    ]

    productos_unicos = df_interno["product_id"].unique()

    df_ventas_familia = []

    for producto in productos_unicos:

        df_interno_interno = df_interno[
            df_interno["product_id"] == int(producto)
        ].reset_index(drop=True)
        df_concat = pd.merge(df, df_interno_interno, on="periodo", how="left")
        df_ventas_familia.append(df_concat["tn"])

    productos_unicos = [
        f"ventas_{str(productos_unicos[j])}" for j in range(len(productos_unicos))
    ]
    df_ventas = pd.DataFrame(data=df_ventas_familia, index=productos_unicos).transpose()
    df_ventas["Fecha"] = df["periodo"]
    df_ventas["Fecha"] = pd.to_datetime(df_ventas["Fecha"].astype(str), format="%Y%m")

    # Reordeno las columnas del DF
    cols = df_ventas.columns.tolist()
    cols_2 = cols[-1:] + cols[:-1]
    df_ventas = df_ventas[cols_2]

    # Reemplazo los Nan Intermedios por el valor promedio de sus valores anterior y posterior
    df_ventas = df_ventas.apply(replace_nan_with_mean)

    # Diccionario para marcar productos como discontinuados
    productos_discontinuados = {
        col: (
            "Discontinuado"
            if np.isnan(df_ventas[col].iloc[-2]) and np.isnan(df_ventas[col].iloc[-1])
            else "No Discontinuado"
        )
        for col in df_ventas.columns
        if col != "Fecha"
    }

    # Filtrar y mostrar solo los productos marcados como "Discontinuado"
    discontinuados_filtrados = {
        k: v for k, v in productos_discontinuados.items() if v == "Discontinuado"
    }
    print("Productos discontinuados:", discontinuados_filtrados)
    ## No encontramos productos discontinuados ##

    # Agrego Columnas al DF (Cantidad total vendido por mes, y % de ventas x producto)
    df_ventas["totales_ventas"] = None
    for l in range(len(df_ventas)):
        df_ventas["totales_ventas"].iloc[l] = df_ventas[cols[:-1]].iloc[l].sum()

    # Agrego columnas nuevas para considerar las edades de los productos y su variación de ventas de forma mensual
    for columna in df_ventas.columns:
        # Ignoro la columna fecha
        if columna == "Fecha" or columna == "totales_ventas":
            continue

        else:
            # Aumento la marca de control
            a += 1

            # Asigno la edad del producto
            df_ventas[f"edad_{columna}"] = [
                (
                    None
                    if np.isnan(val)
                    else (idx - df_ventas[columna].first_valid_index() + 1)
                )
                for idx, val in enumerate(df_ventas[columna])
            ]

            # Creo una columna para cada variable que marque la diferencia de ventas del mes anterior
            df_ventas[f"diff_{columna}"] = [
                (
                    None
                    if m == 0
                    else (
                        (
                            df_ventas[f"{columna}"].iloc[m]
                            - df_ventas[f"{columna}"].iloc[m - 1]
                        )
                        / df_ventas[f"{columna}"].iloc[m - 1]
                        if pd.isna(df_ventas[f"{columna}"].iloc[m - 1]) == False
                        else None
                    )
                )
                for m in range(len(df_ventas))
            ]

            # df_ventas[f'diff_{columna}'] = df_ventas[f'{columna}'].diff()

            # Creo el % de ventas por producto x mes
            df_ventas[f"fraccion_{columna}"] = [
                (
                    df_ventas[f"{columna}"].iloc[m]
                    / df_ventas["totales_ventas"].iloc[m]
                    if df_ventas["totales_ventas"].iloc[m] > 0
                    else None
                )
                for m in range(len(df_ventas))
            ]

            # Check para que siempre tenga 780 productos a predecir
            print(a)

    # Guardo los DF para luego levantarlos como parte del modelo.
    if i < 100:
        numero_formateado = f"{i:03}"
        df_ventas.to_csv(f"familias/{numero_formateado}_familia.csv")
    else:
        df_ventas.to_csv(f"familias/{i}_familia.csv")
