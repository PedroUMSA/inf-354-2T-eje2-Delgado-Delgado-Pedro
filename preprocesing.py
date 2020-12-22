
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


path = ""

datacsv = pd.read_csv(path+"cpu.csv")
header = datacsv.columns

data = np.array(datacsv)


# Remplazar datos faltantes con la media de todos los datos
imputacion = SimpleImputer(missing_values = np.nan, strategy="mean")
mt = imputacion.fit_transform(data[:,0:-1])
data[:,0:-1] = mt


# Remplazar con la moda el class dado que son datos que se repiten
imputacion = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
dt = data[:,-1:len(data)-1]
dt = imputacion.fit_transform(dt)
data[:,-1:len(data)-1] = dt


# Normalizacion: Normalizamos los datos al final pues ya no hay datos faltants 
scaler = StandardScaler()
scaler.fit(mt)
data_norm = scaler.transform(mt)


finaldata = []
for i,j in zip(data_norm, dt):
    i = np.append(i,j[0])
    finaldata.append(i)
finaldata = np.array(finaldata)


# Guardamos los datos preprocesados en otro archivo csv

df = pd.DataFrame(finaldata)
df.columns = header
print(df)

df.to_csv(path+"preprocessed_cpu.csv", index = False, header = True)
