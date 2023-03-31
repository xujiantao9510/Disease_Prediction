from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

plt.style.use("fivethirtyeight")

colors = ["Teal","Indigo","HotPink","DarkGoldenRod","Coral"]
print(os.listdir('/'))
df = pd.read_csv('data_arrhythmia.csv', delimiter=';')
df.dtypes

# df中全是0的列名
df.loc[:,(df==0).all()].columns
# df中存在？的列名
#df.loc[:,(df=='?').any()].columns
#df.head(20)

#inplace=True原地操作值，节省内存
#axis=1时，沿着每一行或者列标签向右执行。
df.drop(["J","R'_wave","S'_wave", "AB", "AC", "AD","AE", "AF", "AG", "AL", "AN", "AO", "AP", "AR", "AS", "AT", "AZ", "AB'", "BC", "BD", "BE", "BG", "BH", "BP", "BR", "BS", "BT", "BU",
          "CA", "CD", "CE", "Cf", "CG", "CH", "CI", "CM","CN","CP","CR","CS","CT","CU","CV","DE","DF","DG","DH","DI","DJ","DR","DS","DT","DU","DV","DY","EG",
          "EH", "EL", "ER", "ET", "EU", "EV", "EY", "EZ", "FA", "FE", "FF", "FH", "FI", "FJ", "FK", "FL", "FM", "FR", "FS", "FU", "FV", "FY", "FZ", "GA",
          "GB", "GG", "GH", "HD", "HE", "HO", "IA", "IB", "IK", "IL", "IY", "JI", "JS", "JT", "KF", "KO", "KP", "LB", "LC", "T", "P", "QRST", "heart_rate"], axis=1, inplace=True)


df.head()

df['height'].value_counts().sort_index()

df.loc[df["height"] == 608, "height"] = 61
df.loc[df["height"] == 780, "height"] = 78
df['height'].value_counts().sort_index()

norm_risk_list = []
for diagnose in df.diagnosis:
    if diagnose == 1:
        norm_risk_list.append("1")
    if diagnose == 2:
        norm_risk_list.append("2")
    if diagnose == 3:
        norm_risk_list.append("3")
    if diagnose == 4:
        norm_risk_list.append("4")
    if diagnose == 5:
        norm_risk_list.append("5")
    if diagnose == 6:
        norm_risk_list.append("6")
    if diagnose == 7:
        norm_risk_list.append("7")
    if diagnose == 8:
        norm_risk_list.append("8")
    if diagnose == 9:
        norm_risk_list.append("9")
    if diagnose == 10:
        norm_risk_list.append("10")
    if diagnose == 11:
        norm_risk_list.append("11")
    if diagnose == 12:
        norm_risk_list.append("12")
    if diagnose == 13:
        norm_risk_list.append("13")
    if diagnose == 14:
        norm_risk_list.append("14")
    if diagnose == 15:
        norm_risk_list.append("15")
    if diagnose == 16:
        norm_risk_list.append("16")
df["label"] = np.array(norm_risk_list)
df.drop(columns = ["diagnosis"],inplace = True)
#df.diagnosis.value_counts()
df.label.value_counts()


# -1是删除最后一列
X = df.drop(columns = [df.columns[-1]])
y = df[df.columns[-1]]

print(X.head(5))
print(y.shape)

from sklearn.neural_network import MLPClassifier

# Prepare the data
# convert the dataframe to numpy array
data = df.values
# split the data into features and labels
X = data[:,:-1]
y = data[:,-1]

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define the MLP model
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)

# train the model
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)
print(clf.predict([[65,1,180,64,81,174,420,149,39,25,37,-17,31,65,0,48,0,0,0,24,0,15,0,0,0,0,0,64,0,0,0,24,0,0,5,0,0,0,32,24,0,0,0,30,0,0,0,0,0,0,60,0,0,0,0,0,0,0,0,0,0,0,0,44,20,0,0,24,0,0,0,0,0,0,0,60,0,0,0,20,0,0,0,0,0,0,0,24,52,0,0,16,0,0,0,0,0,0,0,32,52,0,0,20,0,0,0,0,0,0,0,44,48,0,0,32,0,0,0,0,0,0,0,48,44,0,0,32,0,0,0,0,0,0,0,48,40,0,0,28,0,0,0,0,0,0,0,48,0,0,0,28,0,0,0,0,0,0,-0.6,0.0,7.2,0.0,0.0,0.0,0.4,1.5,17.2,26.5,0.0,0.0,5.5,0.0,0.0,0.0,0.1,1.7]]))
print("Accuracy: ", accuracy_score(y_test, y_pred))
