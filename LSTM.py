import numpy as np
import pandas as pd
import os
import tensorflow as tf
import kagglehub

#Extracting dataset
path = kagglehub.dataset_download("mnassrib/jena-climate")
csv_filename = "jena_climate_2009_2016.csv"
csv_path = os.path.join(path, csv_filename)

df = pd.read_csv(csv_path)
print(df)

