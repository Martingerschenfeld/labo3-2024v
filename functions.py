def split_data(df, fecha_limit, product_id):
    """
    Split the data into training and validation sets based on a given fecha_limit and producto.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    fecha_limit (str): The date limit for splitting the data.
    producto (str): The product column to be predicted.

    Returns:
    tuple: A tuple containing X_train, y_train, X_val, y_val.
    """
    
    train_data = df[df['Fecha'] < fecha_limit].reset_index(drop=True)
    val_data = df[df['Fecha'] == fecha_limit].reset_index(drop=True)

    X_train = train_data.drop([f'ventas_{product_id}', 'Fecha'], axis=1)
    y_train = train_data[f'ventas_{product_id}']

    X_val = val_data.drop([f'ventas_{product_id}', 'Fecha'], axis=1)
    y_val = val_data[f'ventas_{product_id}']

    return X_train, y_train, X_val, y_val