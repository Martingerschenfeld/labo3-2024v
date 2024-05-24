import featuretools as ft
import pandas as pd

df = pd.read_csv('water_potability.csv')


feature_matrix, feature_defs = ft.dfs(entityset=df,
                                      target_entity="Portability",
                                      agg_primitives=["count"],
                                      trans_primitives=["month"],
                                      max_depth=1)