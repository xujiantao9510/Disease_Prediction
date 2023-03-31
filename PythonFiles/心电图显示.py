# # 导入心电信号处理函数库
# import wfdb
# # 导入python的类matlab绘图函数库
# import matplotlib.pyplot as plt
# # 读取本地的100号记录，从0到25000，读取模拟信号,通道0
# record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100', sampfrom=0, sampto=25000, physical=True, channels=[0, ])
# # 读取，从第145个数据到第756个数据
# ventricular_signal = record.p_signal[144:756]
# # 打印标题
# plt.title("ventricular signal")
# # 打印信号
# plt.plot(ventricular_signal)
# # 显示图像
# plt.show()
#
#
#
#
# import wfdb
# signal_annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/106', "atr", sampfrom=0, sampto=1000)
# # 打印标注信息
# print("symbol: " + str(signal_annotation.symbol))
# print("sample: " + str(signal_annotation.sample))


import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/106', sampfrom=0, sampto=1000, physical=True, channels=[0, ])
signal_annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/106', "atr", sampfrom=0, sampto=1000)
# 打印标注信息
ECG = record.p_signal
plt.plot(ECG)
#按坐标在散点图上绘点
for index in signal_annotation.sample:
    plt.scatter(index, ECG[index], marker='*',s=200)
plt.show()




