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
from methods import *

__author__ = "Camilo Jara Do Nascimento"
__email__ = "camilo.jara@ug.uchile.cl"

def setVariables(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs):
    return fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs
"""
################# INIT VARIABLES EXAMPLE #########################
fromdate = "1963-1-1 01"
todate = "" 
#todate = "2017-1-1 01"

columns = ["DST_Index"] 
#columns = ["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"]

split_percent = 0.6

step_hours = 1

number_regressor = 1

norm = True

learning_rate = 0.0005

epochs = 100
##################################################################
"""
##### 1 step-ahead ######
[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index"],0.6,1,1,False,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index"],0.6,1,1,True,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"],0.6,1,1,False,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"],0.6,1,1,True,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

##### 10 steps-ahead ######
[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index"],0.6,1,10,False,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index"],0.6,1,10,True,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"],0.6,1,10,False,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)

[fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs] = setVariables("1963-1-1 01","",["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"],0.6,1,10,True,0.0005,100)
print("Currently working on model: "+ str(number_regressor)+"step-ahead and norm=" + str(norm) + " using " + str(columns) + " as variables" )
run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs)
