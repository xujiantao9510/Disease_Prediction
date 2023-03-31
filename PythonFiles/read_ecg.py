
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data):
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('E:\\Files\\pythonFiles\\mit-bih-arrhythmia-database-1.0.0\\' + number, channel_names=['MLII'])  #源文件都放在ecg_data这个文件夹中了
    data = record.p_signal.flatten()
    #data=np.array(data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('E:\\Files\\pythonFiles\\mit-bih-arrhythmia-database-1.0.0\\' + number, 'atr')
    Rlocation = annotation.sample  #对应位置
    Rclass = annotation.symbol  #对应标签

    X_data.append(data)

    return

# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    for n in numberSet:
        getDataSet(n, dataSet)
    return dataSet

def main():
    dataSet = loadData()
    dataSet = np.array(dataSet)
    print(dataSet[5])
    #x_train, x_test, y_train, y_test = train_test_split(dataSet, random_state=22, test_size=0.2)

if __name__ == '__main__':
    main()
