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


__author__ = "Camilo Jara Do Nascimento"
__email__ = "camilo.jara@ug.uchile.cl"


# ===================================== METHODS ================================================

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def normalize(data,min_,max_):
    data1 = (data - min_) / (max_-min_)
    return data1

def denormalize(data_norm,min_,max_):
    y1 = data_norm*(max_.values-min_.values)+min_.values
    return y1

def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals) if v is not None)

# ===================================== SCRIPT ================================================
### Note that you can make some changes: 
### Change fromdate and todate to select the time range (used at 99 row)
### Change columns to select variables you want to train (used at 106 and 109 rows)
### Change split_percent to select the percent of train data, the rest is for test (used at 109 row)
### Change step_hours to select the slide step (used at 109 row)

################# MAKE THE CHANGES YOU WANT ###################

fromdate = "1963-1-1 01"
todate = "" 
#todate = "2017-1-1 01"

columns = ["DST_Index"] 
#columns = ["DST_Index","Electric_field","Bz_GSM","Flow_Pressure"]

split_percent = 0.6
step_hours = 1

norm = False

################################################################

###   Read OMNI dataset
df = pd.read_csv("./omni2_all_years.dat",sep="\s+", header=None, skiprows=1) #spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
df.columns = ["Year","Day","Hour","Bartels_rotation","ID_IMF" ,"ID_SW_plasma",
            "points_IMF_avg","points_plasma_avg","|B|_avg","Mag_avg_Field", 
            "Lat.ang_Field", "Long.ang_Field","Bx_GSE_GSM","By_GSE","Bz_GSE", 
            "By_GSM", "Bz_GSM" ,"sigma|B|", "sigma_B", "sigma_Bx", "sigma_By",
            "sigma_Bz","Proton_T","Proton_Density","Plasma_flow_speed",
            "Plasma_flow_long.ang", "Plasma_flow_lat.ang","Na/Np","Flow_Pressure",
            "sigma_T","sigma_N","sigma_V","sigma_phi_V","sigma_theta_V","sigma-Na/Np",
            "Electric_field","Plasma_beta","alfven_mach_number", "Kp", "R","DST_Index",
            "AE-index","Proton_flux","Proton_flux","Proton_flux","Proton_flux",
            "Proton_flux","Proton_flux","Flag(***)","","","","","","magnetosonic_mach",
            "Solar_Lyman"]
df.index = compose_date(df['Year'], days=df['Day'],hours=df['Hour'])

### Clean data
df = df.drop(df[(df["DST_Index"] == 99999) | (df["Electric_field"] == 999.99) | (df["Bz_GSM"] == 999.9) | (df["Flow_Pressure"] == 99.99)].index)

###  Select dates to model
if not todate:
    mask = df.index >= fromdate
else:
    mask = (df.index >= fromdate) & (df.index < todate)
df_date = df.loc[mask]

###  Select variables to model
df_var = df_date.loc[:,columns] 

###  Split dataset on train and test
x_train = df_var.iloc[0:int(df_var.shape[0]*split_percent)-2*step_hours:step_hours]
y_train = pd.DataFrame(df_var.loc[:,columns[0]].iloc[step_hours:int(df_var.shape[0]*split_percent)-1*step_hours:step_hours])
x_test = df_var.iloc[int(df_var.shape[0]*split_percent)-1*step_hours:df_var.shape[0]-2*step_hours:step_hours]
y_test = pd.DataFrame(df_var.loc[:,columns[0]].iloc[int(df_var.shape[0]*split_percent):df_var.shape[0]-1*step_hours:step_hours])
"""
mean_x = x_train.mean(axis=0)
std_x = x_train.std(axis=0)
x_train = (x_train - mean_x) / std_x
x_test = (x_test - mean_x) / std_x

mean_y = y_train.mean(axis=0)
std_y = y_train.std(axis=0)
y_train = (y_train - mean_y) / std_y
y_test = (y_test - mean_y) / std_y

"""
if norm:
    max_x = x_train.max(axis=0)
    min_x = x_train.min(axis=0)
    x_train = (x_train - min_x) / (max_x-min_x)
    x_test = (x_test - min_x) / (max_x-min_x)
    #x_train = normalize(x_train,min_x,max_x)
    #x_test = normalize(x_test,min_x,max_x)

    max_y = y_train.max(axis=0)
    min_y = y_train.min(axis=0)
    y_train = (y_train - min_y) / (max_y-min_y)
    y_test = (y_test - min_y) / (max_y-min_y)
    #y_train = normalize(y_train,min_y,max_y)
    #y_test = normalize(y_test,min_y,max_y)



###   If you want a default splitting uncomment the following lines
#X_dataset = df_var.iloc[0:df_var.shape[0]-2]
#Y_dataset = df_var.iloc[1:df_var.shape[0]-1]
#x_train, x_test, y_train, y_test = train_test_split(X_dataset,Y_dataset, test_size = 0.40, random_state=42)

###   Delete some variables to save RAM
#del df
#del X_dataset
#del Y_dataset

"""
#   Code that can be necessary later
#   Group variables by days in same row using list
df_group_list = df_var.groupby(df_var.index.hour).agg(lambda x: list(x))
df_group_list = df_var.groupby([df_var.index.year]).agg(lambda x: list(x))

#   Group columns on dataframe's dict (key->variable_name)
all_df = {col: pd.DataFrame(df_group_list[col].values.tolist()).add_prefix('data ') for col in columns}

#df3 = pd.DataFrame(df2['data'].values.tolist()).add_prefix('data').join(df['Time'])
"""

"""
#=============== MLP Regressor MODEL ====================
###   Prediction
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#nn.fit(x_train.values.reshape(-1,1), y_train.values.ravel())
nn.fit(x_train.values, y_train.values.ravel())
y_test_predict = nn.predict(x_test.values)
df_y_test_predict = pd.DataFrame(y_test_predict,index=y_test.index,columns=["DST_Index"])

###   Plot prediction
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_train["DST_Index"],label="Train data")
ax1.plot(y_test["DST_Index"],label="Validation data")
ax1.plot(df_y_test_predict["DST_Index"],label="NN Prediction")
plt.xlabel('Time')
plt.ylabel('DST Index')
plt.title('Prediction '+str(step_hours)+'-step ahead of DST Index with MLPRegressor using ' + ', '.join(columns) + ' as variables')
ax1.legend()

plt.show()

fig = plt.figure()
pd.DataFrame(nn.loss_curve_).plot()
plt.show()

### Plot learning curve, uncomment if you want
#title = 'Learning and validation curve '+str(step_hours)+'-step ahead using ' + ', '.join(columns) + ' as variables'
#grafico = plot_learning_curve(nn, title, x_test, y_test, cv=10)
#plt.show()
"""
######################### Import stuff ##########################
import tensorflow as tf

######################## Set learning variables ##################
learning_rate = 0.0005 #0.0005
epochs = 100
batch_size = 100
#n_hidden_1 = 1000

######################## Set some variables #######################
x = tf.placeholder(tf.float32, [None, x_train.shape[1]], name='x')  # features
y = tf.placeholder(tf.float32, [None, 1], name='y')  # 1 output

def kezhNet(input_data):
    with tf.name_scope('kezhNet'):
        with tf.name_scope('hidden_1'):
            #bn1_1 = tf.layers.batch_normalization(input_data)
            bn1_1 = input_data
            fc1 = tf.layers.dense(inputs=bn1_1,units=100)
            act_fc1 = tf.nn.relu(fc1)

        with tf.name_scope('hidden_2'):
            #bn1_2 = tf.layers.batch_normalization(act_fc1)
            bn1_2 = act_fc1
            fc2 = tf.layers.dense(inputs=bn1_2,units=25)
            act_fc2 = tf.nn.relu(fc2)
        """
        with tf.name_scope('hidden_3'):
            #bn1_3 = tf.layers.batch_normalization(act_fc2)
            bn1_3 = act_fc2
            fc3 = tf.layers.dense(inputs=bn1_3,units=25)
            act_fc3 = tf.nn.relu(fc3)
            
        with tf.name_scope('hidden_4'):
            #bn1_4 = tf.layers.batch_normalization(act_fc3)
            bn1_4 = act_fc3
            fc4 = tf.layers.dense(inputs=bn1_4,units=200)
            act_fc4 = tf.nn.relu(fc4)
        with tf.name_scope('hidden_5'):
            #bn1_5 = tf.layers.batch_normalization(act_fc4)
            bn1_5 = act_fc4
            fc5 = tf.layers.dense(inputs=bn1_5,units=100)
            act_fc5 = tf.nn.relu(fc5)
            """
        with tf.name_scope('output'):
            #bn1_out = tf.layers.batch_normalization(act_fc5)
            bn1_out = act_fc2
            fc_out = tf.layers.dense(inputs=bn1_out,units=1)
    return fc_out

####################### Prediction  #########################
y_ = kezhNet(x)

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)

####################### Optimizer  #########################
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

###################### Initialize and Run #################
# Initialize variables
init_op = tf.global_variables_initializer()

train_loss = []
epoch_time = []
# Run
with tf.Session() as sess:
    sess.run(init_op)
    t = time.clock()
    total_batch = int(len(y_train) / batch_size)
    print("Start Training...")
    for epoch in range(epochs):
        avg_cost = 0
        epoch_loss = []
        t_epoch = time.clock()
        for i in range(total_batch):
            batch_x  = x_train.values[i * batch_size:min(i * batch_size + batch_size, len(x_train)), :]
            batch_y = y_train.values[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
            _, l = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
            epoch_loss.append(l)
        if epoch % 10 == 0:
            elapsed_time = time.clock() - t_epoch
            epoch_time.append(elapsed_time)
            avg_loss = np.mean(epoch_loss)
            train_loss.append(avg_loss)
            print("Epoch: {} | Loss: {:.3f} | Elapsed Time: {:.2f} minutes".format(epoch + 1, avg_loss, elapsed_time / 60))
    print("Training Finished! Elapsed Time: {:.2f} minutes".format((time.clock() - t) / 60))
    pred_f = sess.run(y_, feed_dict={x: x_test})
    df_pred = pd.DataFrame(pred_f,index=y_test.index.values,columns=["DST_Index"])    
    if norm:
        #y_test_final = std_y.values*y_test+mean_y.values
        y_test_final = y_test*(max_y.values-min_y.values)+min_y.values

        #df_predic_final = std_y.values*df_pred+mean_y.values
        df_pred_final = df_pred*(max_y.values-min_y.values)+min_y.values
    else:
        y_test_final = y_test
        df_pred_final = df_pred
        
    # Prediction
    pred_fig = plt.figure()
    pred_title = 'Prediction '+str(step_hours)+'-step ahead with MLPRegressor using ' + ', '.join(columns) + ' as variables'
    plt.plot(y_test_final,label="Original Data")
    plt.plot(df_pred_final,label="Predicted Data")
    plt.xlabel('Time')
    plt.ylabel('DST Index')
    plt.title(pred_title)
    plt.legend()
    pred_fig.savefig("./results/plots/"+pred_title+".png", bbox_inches='tight')

    # Train Loss
    loss_fig = plt.figure()
    loss_title = 'Model Train Loss '+str(step_hours)+'-step ahead using ' + ', '.join(columns) + ' as variables'
    plt.plot(range(1, len(train_loss) + 1), train_loss)
    plt.title(loss_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['kezhNet'])
    loss_fig.savefig("./results/plots/"+loss_title+".png", bbox_inches='tight')

    # Error Histogram
    hist_fig, ax = plt.subplots()
    error = abs(df_pred_final - y_test_final)
    error.hist(bins=50,ax=ax)
    hist_title = 'Distribution Error '+str(step_hours)+'-step ahead using ' + ', '.join(columns) + ' as variables'
    plt.title(hist_title)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.xlim(0,80)
    hist_fig.savefig("./results/plots/"+hist_title+".png", bbox_inches='tight')
    plt.show()

"""
#================ KERAS MODEL =======================
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=3))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train.values.reshape((-1,3)), y_train.values,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test.values, y_test.values, batch_size=128)

"""