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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#MinMaxScaler归一化处理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = X_train_scaled
X_test = X_test_scaled

model_names = ["Logistic Regression",
               "K-Nearest Neighbors",
               "Decision Tree Classifier",
               "Random Forest Classifier",
               "Gaussian Naive Bayes"]
models = []
predictions = []
pred_probabilities = []
'''
log_model = LogisticRegression(random_state=0,solver = "saga")
log_reg1=LogisticRegression(multi_class="multinomial",solver="newton-cg")
ovr=OneVsRestClassifier(log_model)    #1-2进行多分类转换
ovo=OneVsOneClassifier(log_reg1)
models.append(log_model)
ovr.fit(X_train, y_train)            #1-3进行数据训练与预测
ovo.fit(X_train, y_train)
'''

log_reg=LogisticRegression()        #1-1定义一种二分类算法
log_reg1=LogisticRegression(multi_class="multinomial",solver="newton-cg")
ovr=OneVsRestClassifier(log_reg)    #1-2进行多分类转换
ovo=OneVsOneClassifier(log_reg1)
ovr.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovr.score(X_test,y_test))
ovo.fit(X_train,y_train)
print(ovo.score(X_test,y_test))

pred_probabilities.append(ovr.predict_proba(X_test))

# predictions.append(ovr.predict(X_test))
# predictions.append(ovo.predict(X_test))

print("The label that predicts the unknown data is：\n",ovr.predict([[56,1,165,64,81,174,401,149,39,25,37,-17,31,53,0,48,0,0,0,24,0,0,0,0,0,0,0,64,0,0,0,24,0,0,0,0,0,0,32,24,0,0,0,40,0,0,0,0,0,0,48,0,0,0,0,0,0,0,0,0,0,0,0,44,20,0,0,24,0,0,0,0,0,0,0,60,0,0,0,20,0,0,0,0,0,0,0,24,52,0,0,16,0,0,0,0,0,0,0,32,52,0,0,20,0,0,0,0,0,0,0,44,48,0,0,32,0,0,0,0,0,0,0,48,44,0,0,32,0,0,0,0,0,0,0,48,40,0,0,28,0,0,0,0,0,0,0,48,0,0,0,28,0,0,0,0,0,0,-0.6,0.0,7.2,0.0,0.0,0.0,0.4,1.5,17.2,26.5,0.0,0.0,5.5,0.0,0.0,0.0,0.1,1.7]]))
'''
knn_model = KNeighborsClassifier(n_neighbors=50)
models.append(knn_model)
knn_model.fit(X_train, y_train)
knn_predprob = knn_model.predict_proba(X_test)
pred_probabilities.append(knn_predprob)
knn_pred = knn_model.predict(X_test)
predictions.append(knn_pred)

#检查最优的n_neighbors值
best_score = -np.inf
best_n = np.inf
for n in range(1,df.columns.shape[0]):
   temp_model = KNeighborsClassifier(n_neighbors=n)
   print(n,end=" ")
   temp_model.fit(X_train, y_train)
   temp_predprob = temp_model.predict_proba(X_test)
   temp_score = roc_auc_score(y_test,temp_predprob[:, 1])
   if temp_score > best_score:
           best_score = temp_score
           best_n = n
print("Best performing number of n_neighbors is",best_n,"scoring",round(best_score * 100 , 2))


tree_model = DecisionTreeClassifier(random_state=0,max_depth = 8,max_features="auto")
models.append(tree_model)
tree_model.fit(X_train, y_train)
tree_predprob = tree_model.predict_proba(X_test)
pred_probabilities.append(tree_predprob)
tree_pred = tree_model.predict(X_test)
predictions.append(tree_pred)

#检查最优的max_depth值
best_score = -np.inf
best_n = np.inf
for n in range(1,df.columns.shape[0]):
   temp_tree_model = DecisionTreeClassifier(random_state=0,max_depth = n,max_features="auto")
   print(n,end=" ")
   temp_tree_model.fit(X_train, y_train)
   temp_tree_predprob = temp_tree_model.predict_proba(X_test)
   temp_score = roc_auc_score(y_test,temp_tree_predprob[:, 1])
   if temp_score > best_score:
           best_score = temp_score
           best_n = n
print("Best performing number of max_depth is",best_n,"scoring",round(best_score * 100 , 2))

rndfor_model = RandomForestClassifier(max_depth=9, random_state=0,n_estimators = 100)
models.append(rndfor_model)
rndfor_model.fit(X_train, y_train)
rndfor_predprob = rndfor_model.predict_proba(X_test)
pred_probabilities.append(rndfor_predprob)
rndfor_pred = rndfor_model.predict(X_test)
predictions.append(rndfor_pred)

#检查最优的max_depth值
best_score = -np.inf
best_n = np.inf
for n in range(1,df.columns.shape[0]):
   temp_tree_model = RandomForestClassifier(max_depth=n, random_state=0,n_estimators = 100)
   print(n,end=" ")
   temp_tree_model.fit(X_train, y_train)
   temp_tree_predprob = temp_tree_model.predict_proba(X_test)
   temp_score = roc_auc_score(y_test,temp_tree_predprob[:, 1])
   if temp_score > best_score:
           best_score = temp_score
           best_n = n
print("Best performing number of max_depth is",best_n,"scoring",round(best_score * 100 , 2))

nb_model = GaussianNB(var_smoothing = 0.00001)
models.append(nb_model)
nb_model.fit(X_train, y_train)
nb_predprob = nb_model.predict_proba(X_test)
pred_probabilities.append(nb_predprob)
nb_pred = nb_model.predict(X_test)
predictions.append(nb_pred)
nb_predprob.shape

#测试集的预测准确性
#AUC(area under the curve)曲线下面积，AUC在机器学习领域中是一种模型评估指标。
#ROC（receiver operating characteristic curve）受试者特征曲线
for name,pred in zip(model_names,predictions):
    print(name,"Accuracy:",round(accuracy_score(y_test,pred) * 100 , 2),"%")


for name,pred in zip(model_names,pred_probabilities):
    print(name,"AUROC:",round(roc_auc_score(y_test,pred[:, 1]) * 100 , 2),"%")

fprs = []
tprs = []
for i, pred in enumerate(pred_probabilities):
    fpr, tpr, thresholds = roc_curve(y_test, pred[:, 1],drop_intermediate = False)
    fprs.append(fpr)
    tprs.append(tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr, color=colors[i], lw=2)
    plt.plot([0,1],[0,1],linestyle='--', color='black', lw=.8)
    plt.title(model_names[i] + " ROC - AUC " + str(round(roc_auc_score(y_true=y_test,y_score=pred[:,1]) * 100,2)) + " % " + " 0-PCA",fontsize=10)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=8)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=8)
    plt.xticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.yticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.show()


from sklearn.decomposition import PCA
best_scores = []
best_ns = []

for i, model in enumerate(models):
    best_score = -np.inf
    best_n = np.inf
    for n in range(1,df.columns.shape[0]):
        pca = PCA(n_components=n,random_state = 0)
        temp_X_train = pca.fit_transform(X_train)
        temp_X_test = pca.transform(X_test)
        model.fit(temp_X_train,y_train)
        temp_pred = model.predict_proba(temp_X_test)
        temp_score = roc_auc_score(y_test,temp_pred[:, 1])
        if temp_score > best_score:
            best_score = temp_score
            best_n = n
    print("Best performing number of components for",model_names[i],"is",best_n,"scoring",round(best_score * 100 , 2))
    best_scores.append(best_score)
    best_ns.append(best_n)

from sklearn.decomposition import PCA

overall_best_score = -np.inf
overall_best_n = np.inf
for n in range(1,df.columns.shape[0]):
    #print(str(round(n / df.columns.shape[0] * 100,2)) + "%",end = " ")
    pca = PCA(n_components=n,random_state = 0)
    temp_X_train = pca.fit_transform(X_train)
    temp_X_test = pca.transform(X_test)
    temp_score = 0
    for i, model in enumerate(models):
        model.fit(temp_X_train,y_train)
        temp_pred = model.predict_proba(temp_X_test)
        model_score = roc_auc_score(y_test,temp_pred[:, 1])
        temp_score += model_score
    temp_score /= len(model_names)
    if temp_score > overall_best_score:
        overall_best_score = temp_score
        overall_best_n = n
print("Best performing number of components for all models is",best_n,"scoring",round(best_score * 100 , 2))

pca_pred_probs = []
for i, model in enumerate(models):
    pca = PCA(n_components=best_ns[i],random_state = 0)
    temp_X_train = pca.fit_transform(X_train)
    temp_X_test = pca.transform(X_test)
    model.fit(temp_X_train,y_train)
    temp_pred = model.predict_proba(temp_X_test)
    pca_pred_probs.append(temp_pred)

fprs = []
tprs = []
for i, pred in enumerate(pca_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, pred[:, 1],drop_intermediate = False)
    fprs.append(fpr)
    tprs.append(tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr, color=colors[i], lw=2)
    plt.plot([0,1],[0,1],linestyle='--', color='black', lw=.8)
    plt.title(model_names[i] + " ROC - AUC " + str(round(roc_auc_score(y_true=y_test,y_score=pred[:,1]) * 100,2)) + "%" + str(best_ns[i]) + "-PCA",fontsize=10)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=8)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=8)
    plt.xticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.yticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.show()


overall_pca_pred_probs = []
for i, model in enumerate(models):
    pca = PCA(n_components=overall_best_n,random_state = 0)
    temp_X_train = pca.fit_transform(X_train)
    temp_X_test = pca.transform(X_test)
    model.fit(temp_X_train,y_train)
    temp_pred = model.predict_proba(temp_X_test)
    overall_pca_pred_probs.append(temp_pred)

fprs = []
tprs = []
for i, pred in enumerate(overall_pca_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, pred[:, 1],drop_intermediate = False)
    fprs.append(fpr)
    tprs.append(tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr, color=colors[i], lw=2)
    plt.plot([0,1],[0,1],linestyle='--', color='black', lw=.8)
    plt.title(model_names[i] + " ROC - AUC " + str(round(roc_auc_score(y_true=y_test,y_score=pred[:,1]) * 100,2)) + "%" + str(overall_best_n) + "-PCA",fontsize=10)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=8)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=8)
    plt.xticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.yticks([x / 10 for x in range(0,11)], fontsize=10)
    plt.show()


best_model = RandomForestClassifier(max_depth=9, random_state=0,n_estimators = 100)

pca = PCA(n_components=132,random_state = 0)
best_X_train = pca.fit_transform(X_train)
best_X_test = pca.transform(X_test)
best_model.fit(best_X_train, y_train)
best_predprob = best_model.predict_proba(best_X_test)
best_pred = best_model.predict(best_X_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_model, best_X_test, y_test)
plt.grid(which = "major")
plt.show()
'''