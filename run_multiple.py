import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_predict,validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import scikitplot as skplt
import graphviz
import pydotplus
from itertools import *
import time
import tensorflow as tf
import os
#from process_datos import *

fromdate = "1963-1-1 01"
todate = "" 
columns = ["DST_Index"] 
#columns = ["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"]
split_percent = 0.6
step_hours = 1
number_regressor = 1
norm = True

os.system("./process_datos.py 1")
#subprocess.Popen("./process_datos.py 1", shell=True)

fromdate = "1963-1-1 01"
todate = "" 
columns = ["DST_Index"] 
#columns = ["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"]
split_percent = 0.6
step_hours = 1
number_regressor = 1
norm = False

subprocess.Popen("./process_datos.py 1", shell=True)