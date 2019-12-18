import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans


def read_points(path):
    dataset = []
    with open(path,'r') as file:
        for line in file:
            if line == '\n':
                continue
            dataset.append(list(map(float,line.split('\t'))))
        file.close() 
        return  dataset

def start_cluster(data, t):
    zs = [data[0]]  # 聚类中心集，选取第一个模式样本作为第一个聚类中心Z1
    # 第2步：寻找Z2,并计算阈值T
    T = step2(data, t, zs)
    # 第3,4,5步，寻找所有的聚类中心
    get_clusters(data, zs, T)
    # 按最近邻分类
    result,sample_num = classify(data, zs, T)
    return result,sample_num


# 分类
def classify(data, zs, T):
    result = [[] for i in range(len(zs))]
    sample_num = [[] for i in range(len(zs))]
    num = 0
    for aData in data:
        min_distance = T
        index = 0
        for i in range(len(zs)):
            temp_distance = get_distance(aData, zs[i])
            if temp_distance < min_distance:
                min_distance = temp_distance
                index = i
        result[index].append(aData)
        sample_num[index].append(num)
        num = num + 1
    return result,sample_num


# 寻找所有的聚类中心
def get_clusters(data, zs, T):
    max_min_distance = 0
    index = 0
    for i in range(len(data)):
        min_distance = []
        for j in range(len(zs)):
            distance = get_distance(data[i], zs[j])
            min_distance.append(distance)
        min_dis = min(dis for dis in min_distance)
        if min_dis > max_min_distance:
            max_min_distance = min_dis
            index = i
    if max_min_distance > T:
        zs.append(data[index])
        # 迭代
        get_clusters(data, zs, T)


# 寻找Z2,并计算阈值T
def step2(data, t, zs):
    distance = 0
    index = 0
    for i in range(len(data)):
        temp_distance = get_distance(data[i], zs[0])
        if temp_distance > distance:
            distance = temp_distance
            index = i
    # 将Z2加入到聚类中心集中
    zs.append(data[index])
    # 计算阈值T
    T = t * distance
    return T


# 计算两个模式样本之间的欧式距离
def get_distance(data1, data2):
    distance = 0
    for i in range(len(data1)):
        distance += pow((data1[i]-data2[i]), 2)
    return math.sqrt(distance)


#聚类结果评价
def assess(label, label_pred):
    a_valence=0
    b_valence=0
    c_valence=0
    d_valence=0
    a_arousal=0
    b_arousal=0
    c_arousal=0
    d_arousal=0
    label = np.array(label)
    label_valence = [int(f) for f in label[:,0]]
    label_arousal = [int(f) for f in label[:,1]]
    for i in range(len(label)):
        for j in range(i+1,len(label)):
            if label_pred[i] == label_pred[j]:
                if label_valence[i] == label_valence[j]:
                    a_valence = a_valence + 1
                if label_valence[i] != label_valence[j]:
                    b_valence = b_valence + 1
                if label_arousal[i] == label_arousal[j]:
                    a_arousal = a_arousal + 1
                if label_arousal[i] != label_arousal[j]:
                    b_arousal = b_arousal + 1
            else:
                if label_valence[i] == label_valence[j]:
                    c_valence = c_valence + 1
                if label_valence[i] != label_valence[j]:
                    d_valence = d_valence + 1
                if label_arousal[i] == label_arousal[j]:
                    c_arousal = c_arousal + 1
                if label_arousal[i] != label_arousal[j]:
                    d_arousal = d_arousal + 1
    JC_valence = a_valence/(a_valence+b_valence+c_valence)
    FMI_valence = sqrt((a_valence/(a_valence+b_valence))*(a_valence/(a_valence+c_valence)))
    RI_valence = (2*(a_valence+b_valence))/(len(label)*(len(label)-1))
    JC_arousal = a_arousal/(a_arousal+b_arousal+c_arousal)
    FMI_arousal = sqrt((a_arousal/(a_arousal+b_arousal))*(a_arousal/(a_arousal+c_arousal)))
    RI_arousal = (2*(a_arousal+b_arousal))/(len(label)*(len(label)-1))    
    return JC_valence, FMI_valence, RI_valence,JC_arousal, FMI_arousal, RI_arousal



path_dataset = 'E:\python_program\cluster\数据\DEAP\EEG_feature.txt'
path_label = 'E:\\python_program\\cluster\\数据\\DEAP\\valence_arousal_label.txt'
dataset = read_points(path_dataset)
label = read_points(path_label)
count = 0
JC_valence = [0 for i in range(7)]
FMI_valence = [0 for i in range(7)]
RI_valence = [0 for i in range(7)]
JC_arousal = [0 for i in range(7)]
FMI_arousal = [0 for i in range(7)]
RI_arousal = [0 for i in range(7)]
label_pred = [0 for i in range(len(label))]

for t in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    result,sample_num = start_cluster(dataset, t)
    for i in range(len(result)):
        for j in sample_num[i]:
            label_pred[j] = i
    JC_valence[count], FMI_valence[count], RI_valence[count],JC_arousal[count], FMI_arousal[count], RI_arousal[count] = assess(label, label_pred)

    print(JC_valence[count], FMI_valence[count], RI_valence[count],JC_arousal[count], FMI_arousal[count], RI_arousal[count])
    count = count + 1
x = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.title('Max-Min-diatance DEAP Result Analysis')
plt.plot(x,JC_valence, 'ro-', label='JC_valence')
plt.plot(x,FMI_valence, 'go-', label='FMI_valence')
plt.plot(x,RI_valence, 'yo-', label='RI_valence')
plt.plot(x,JC_arousal,'rx-', label='JC_arousal',markersize=10)
plt.plot(x,FMI_arousal, 'gx-', label='FMI_arousal',markersize=10)
plt.plot(x,RI_arousal, 'yx-', label='RI_arousal',markersize=10)

plt.legend() # 显示图例

plt.xlabel('t')
plt.ylabel('assess_parameter')
plt.show()

'''print(JC_valence, FMI_valence, RI_valence,
          JC_arousal, FMI_arousal, RI_arousal)'''
        

'''print('\n\n---------------------------------最大最小距离聚类结果---------------------------------------\n\n')
for i in range(len(result)):
    print ("----------第" + str(i+1) + "个聚类----------")
    print (label_pred[i])'''
