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

__author__ = "Camilo Jara Do Nascimento"
__email__ = "camilo.jara@ug.uchile.cl"


##################################################################
#################                                #################            
#################           METHODS              #################
#################                                #################
##################################################################

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


def read_and_split(fromdate,todate,columns,split_percent,step_hours):
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
    X_train = df_var.iloc[0:int(df_var.shape[0]*split_percent):step_hours]
    X_test = df_var.iloc[int(df_var.shape[0]*split_percent):df_var.shape[0]:step_hours]
    return X_train,X_test

def datasetWithRegressors(number_regressor,X_train,X_test,columns):
    x_train_dict = {}
    x_test_dict = {}
    for i in range(number_regressor):
        # Create train dict from regressors
        x_train_t = X_train.iloc[i:X_train.shape[0]-number_regressor+i].add_suffix('(t-'+str(number_regressor-i)+')')
        x_train_t_dict = x_train_t.to_dict()
        x_train_dict = {**x_train_dict, **x_train_t_dict} 
        # Create test dict from regressors
        x_test_t = X_test.iloc[i:X_test.shape[0]-number_regressor+i].add_suffix('(t-'+str(number_regressor-i)+')')
        x_test_t_dict = x_test_t.to_dict()
        x_test_dict = {**x_test_dict, **x_test_t_dict} 
    # Dicts to DataFrame
    df_x_train = pd.DataFrame(x_train_dict)
    df_x_test = pd.DataFrame(x_test_dict)
    df_y_train = pd.DataFrame(X_train.loc[:,columns[0]].iloc[number_regressor:X_train.shape[0]])
    df_y_test = pd.DataFrame(X_test.loc[:,columns[0]].iloc[number_regressor:X_test.shape[0]])
    return df_x_train, df_x_test, df_y_train, df_y_test

### Clean columns with NaN values and convert to numpy array
def clean_and_convert(df_x_train,df_x_test,df_y_train,df_y_test):
    # Drop nan values for every column
    df1_x_train = df_x_train.apply(lambda x: pd.Series(x.dropna().values))
    df1_x_test = df_x_test.apply(lambda x: pd.Series(x.dropna().values))
    # Convert DataFrame to array
    x_train = df1_x_train.values
    x_test = df1_x_test.values
    y_train = df_y_train.values
    y_test = df_y_test.values
    return x_train, x_test, y_train, y_test

def normalize_dataset(norm,x_train, x_test, y_train, y_test):
    max_x = x_train.max(axis=0)
    min_x = x_train.min(axis=0)
    max_y = y_train.max(axis=0)
    min_y = y_train.min(axis=0)
    if norm:
        x_train = (x_train - min_x) / (max_x-min_x)
        x_test = (x_test - min_x) / (max_x-min_x)
        
        y_train = (y_train - min_y) / (max_y-min_y)
        y_test = (y_test - min_y) / (max_y-min_y)

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
    return x_train, x_test, y_train, y_test, max_x, min_x, max_y, min_y


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
            fc3 = tf.layers.dense(inputs=bn1_3,units=1000)
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

def MLP(x_train,y_train,x_test,y_test,df_y_test,max_y,min_y,learning_rate,epochs,number_regressor,norm,columns):
    print("Setting variables for the Regression...")
    x = tf.placeholder(tf.float32, [None, x_train.shape[1]], name='x')  # features
    y = tf.placeholder(tf.float32, [None, 1], name='y')  # 1 output

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
        #total_batch = int(len(y_train) / batch_size)
        print("Start Training...")
        for epoch in range(epochs):
            epoch_loss = []
            t_epoch = time.clock()
            #for i in range(total_batch):
                #batch_x  = x_train[i * batch_size:min(i * batch_size + batch_size, len(x_train)), :]
                #batch_y = y_train[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
                #_, l = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
            _, l = sess.run([optimizer, mse],feed_dict={x: x_train, y: y_train})
            epoch_loss.append(l)
            if epoch % 10 == 0:
                elapsed_time = time.clock() - t_epoch
                epoch_time.append(elapsed_time)
                avg_loss = np.mean(epoch_loss)
                train_loss.append(avg_loss)
                print("Epoch: {} | Loss: {:.3f} | Elapsed Time: {:.2f} minutes".format(epoch + 1, avg_loss, elapsed_time / 60))
        print("Training Finished! Elapsed Time: {:.2f} minutes".format((time.clock() - t) / 60))
        pred = sess.run(y_, feed_dict={x: x_test})
        
        df_pred = pd.DataFrame(pred,index=df_y_test.index.values,columns=["DST_Index"])    
        
        ### Denormalize and calculate error
        if norm:
            #df_pred_final = std_y.values*df_pred+mean_y.values
            df_pred_final = df_pred*(max_y-min_y)+min_y
        else:
            df_pred_final = df_pred

        error = abs(df_pred_final - df_y_test)
        acc = np.mean((df_pred.values-y_test)**2)
        acc_denorm = np.mean((error)**2)

        # Prediction
        pred_fig = plt.figure()
        pred_title = 'Prediction '+str(number_regressor)+'-step ahead with MLPRegressor using ' + ', '.join(columns) + ' as variables'+", norm="+str(norm)
        plt.plot(df_y_test,label="Original Data")
        plt.plot(df_pred_final,label="Predicted Data")
        plt.xlabel('Time')
        plt.ylabel('DST Index')
        plt.title(pred_title)
        plt.legend()
        pred_fig.savefig("./results/plots/"+pred_title+" norm="+str(norm)+".png", bbox_inches='tight')

        # Train Loss
        loss_fig = plt.figure()
        loss_title = 'Model Train Loss '+str(number_regressor)+'-step ahead using ' + ', '.join(columns) + ' as variables'+", norm="+str(norm)
        plt.plot(range(1, len(train_loss) + 1), train_loss)
        plt.title(loss_title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['kezhNet'])
        loss_fig.savefig("./results/plots/"+loss_title+" norm="+str(norm)+".png", bbox_inches='tight')

        # Error Histogram
        hist_fig, ax = plt.subplots()
        error.hist(bins=50,ax=ax)
        hist_title = 'Distribution Error '+str(number_regressor)+'-step ahead using ' + ', '.join(columns) + ' as variables'+", norm="+str(norm)
        plt.title(hist_title)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.xlim(0,np.max(error).values)
        hist_fig.savefig("./results/plots/"+hist_title+" norm="+str(norm)+".png", bbox_inches='tight')

        # Error on time
        error_fig, ax = plt.subplots()
        plt.plot(error,label="Error")
        error_title = 'Error on time '+str(number_regressor)+'-step ahead using ' + ', '.join(columns) + ' as variables'+", norm="+str(norm)
        plt.title(error_title)
        plt.xlabel("Time")
        plt.ylabel("Error")
        error_fig.savefig("./results/plots/"+error_title+" norm="+str(norm)+".png", bbox_inches='tight')
        
        ### Cumulative Mean-Square Error
        error2 = error**2
        df_error_acum = error2.expanding(min_periods=1).mean()
        error_acum_fig, ax = plt.subplots()
        plt.plot(df_error_acum,label="Cumulative Mean-Square Error")
        error_title = 'Cumulative Mean-Square Error on time '+str(number_regressor)+'-step ahead using ' + ', '.join(columns) + ' as variables'+", norm="+str(norm)
        plt.title(error_title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Mean-Square Error")
        error_acum_fig.savefig("./results/plots/"+error_title+".png", bbox_inches='tight')
        
        print("Accuracy Test (MSE): ", acc,acc_denorm.values[0])
        #plt.show()
        plt.close('all')
        

def run_program(fromdate,todate,columns,split_percent,step_hours,number_regressor,norm,learning_rate,epochs):
    ### Splitted dataset
    print("Read and splitting dataset...")
    [X_train, X_test] = read_and_split(fromdate,todate,columns,split_percent,step_hours)

    ### Dataset with Regressors
    print("Creating dataset with regressors...")
    [df_x_train,df_x_test,df_y_train,df_y_test] = datasetWithRegressors(number_regressor,X_train,X_test,columns)

    ### Cleaned dataset on numpy array
    print("Cleaning dataset and transform to numpy array...")
    [x_train, x_test, y_train, y_test] = clean_and_convert(df_x_train,df_x_test,df_y_train,df_y_test)

    print("Normalizing dataset = " + str(norm) + "...")
    [x_train, x_test, y_train, y_test, max_x, min_x, max_y, min_y] = normalize_dataset(norm, x_train, x_test, y_train, y_test)

    MLP(x_train,y_train,x_test,y_test,df_y_test,max_y,min_y,learning_rate,epochs,number_regressor,norm,columns)
