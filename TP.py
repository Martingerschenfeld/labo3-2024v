# Importo las librerias
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from autofeat import FeatureSelector
from pycaret.anomaly import setup, create_model, assign_model, plot_model, predict_model, save_model
from dataprep.eda import plot, plot_correlation, plot_missing
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport
import sweetviz
import dtale
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv('water_potability.csv')

#Potability es el target, lo vamos a llamar y
df.rename(columns={'Potability': 'y'}, inplace=True)

# TPot no tiene ningun preprocesamiento cuando la columna es de tipo string. Se tiene que realizar el preprocesamiento de forma manual
df_tpot = df

columnasObject = df.select_dtypes(include=np.object_).columns
for columna in columnasObject:
    df_tpot = pd.concat([df.drop(columna, axis=1), pd.get_dummies(df[columna])], axis=1)

df_tpot = df_tpot.fillna(0)
# Separo el modleo entre train y test
X_train, X_test, y_train, y_test = train_test_split(df_tpot.drop(['y'], axis=1), df_tpot.y, train_size=0.75, test_size=0.25, random_state=42)

generations=5
population_size=5
pipeline_optimizer = TPOTClassifier(
    generations=generations,
    population_size=population_size,
    verbosity=2,
    random_state=42,
    memory='auto',
    n_jobs=-1,
    scoring='f1')

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_export_default_pipeline.py')

#TPOT 
#Best pipeline: MultinomialNB(input_matrix, alpha=100.0, fit_prior=False)
#0.4492131616595136

#02 01 Data Prep #Me funcionó ayer pero hoy no
plot(df).show_browser()
plot_correlation(df).show_browser()
plot_missing(df).show_browser()


#02 02 AutoViz
AV = AutoViz_Class()
df_av = AV.AutoViz(
    'water_potability.csv',
    sep = ",", 
    depVar = 'Potability',
    chart_format='html',
    save_plot_dir='./autoviz_plots')


#02 03 Pandas Profiling
profile = ProfileReport(df, title="Pandas Profiling Report", html={'style':{'full_width': True}})
profile.to_file("pandas_profile.html")

#02 04 Sweet Viz
sweetviz_report  = sweetviz.analyze(df)
sweetviz_report.show_html('SweetViz.html')

#02 05  # Martin: A mi no me funcióno esta
#d = dtale.show(df) 
#d.open_browser()


fsel = FeatureSelector(verbose=1)
X,y = df[df.columns[df.columns != 'y']], df['y']
new_X = fsel.fit_transform(pd.DataFrame(X), pd.DataFrame(y))
new_X.columns = X.columns


# [featsel] Scaling data...done.
# 2024-04-24 22:48:41,208 INFO: [featsel] Feature selection run 1/5
# 2024-04-24 22:48:42,552 INFO: [featsel] Feature selection run 2/5
# 2024-04-24 22:48:42,584 INFO: [featsel] Feature selection run 3/5
# 2024-04-24 22:48:42,627 INFO: [featsel] Feature selection run 4/5
# 2024-04-24 22:48:42,659 INFO: [featsel] Feature selection run 5/5
# 2024-04-24 22:48:42,691 INFO: [featsel] 1 features after 5 feature selection runs
# 2024-04-24 22:48:42,691 INFO: [featsel] 1 features after correlation filtering
# 2024-04-24 22:48:42,699 INFO: [featsel] 0 features after noise filtering
# 2024-04-24 22:48:42,699 WARNING: [featsel] Not a single good features was found...
# 2024-04-24 22:48:42,700 WARNING: [FeatureSelector] No good features found; returning data unchanged. 

# PyCaret Anomaly Detection

data = df.sample(frac=0.95, random_state=786)
data_unseen = df.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

setup(data = data, session_id=1001)
iforest = create_model('iforest')
iforest_results = assign_model(iforest)
plot_model(iforest)
plot_model(iforest, plot = 'umap')
unseen_predictions = predict_model(iforest, data=data_unseen)
save_model(iforest,'./anomaly01')
