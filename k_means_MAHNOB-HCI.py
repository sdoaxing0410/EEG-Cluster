from numpy import *
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans
np.set_printoptions(threshold=np.nan)

def read_points(path):
    dataset = []
    with open(path,'r') as file:
        for line in file:
            if line == '\n':
                continue
            dataset.append(list(map(float,line.split('\t'))))
        file.close() 
        return  dataset


def k_means(dataset, k):
    # 构造一个聚类数为k的聚类器
    estimator = KMeans(n_clusters=k)#构造聚类器
    estimator.fit(dataset)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    '''
    print('\n\n---------------------------------k-means聚类结果---------------------------------------\n\n')
    for kind in range(k):
        arr = np.where(label_pred == kind)
        print("----------第%d个聚类----------"%(kind+1))
        count=0
        for num in arr[0]:
            print(num,end=' ')
            count=count+1
            if count%25==0:
                print('\n')
        print('\n')
        '''
    return label_pred


#聚类结果评价
def assess(label, label_emotion, label_pred):
    a_valence=0
    b_valence=0
    c_valence=0
    d_valence=0
    a_arousal=0
    b_arousal=0
    c_arousal=0
    d_arousal=0
    a_emotion=0
    b_emotion=0
    c_emotion=0
    d_emotion=0
    label = np.array(label)
    label_emotion = np.array(label_emotion)
    label_valence = [int(f) for f in label[:,0]]
    label_arousal = [int(f) for f in label[:,1]]
    label_emotion = [int(f) for f in label_emotion]
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
                if label_emotion[i] == label_emotion[j]:
                    a_emotion = a_emotion + 1
                if label_emotion[i] != label_emotion[j]:
                    b_emotion = b_emotion + 1
            else:
                if label_valence[i] == label_valence[j]:
                    c_valence = c_valence + 1
                if label_valence[i] != label_valence[j]:
                    d_valence = d_valence + 1
                if label_arousal[i] == label_arousal[j]:
                    c_arousal = c_arousal + 1
                if label_arousal[i] != label_arousal[j]:
                    d_arousal = d_arousal + 1
                if label_emotion[i] == label_emotion[j]:
                    c_emotion = c_emotion + 1
                if label_emotion[i] != label_emotion[j]:
                    d_emotion = d_emotion + 1
    JC_valence = a_valence/(a_valence+b_valence+c_valence)
    FMI_valence = sqrt((a_valence/(a_valence+b_valence))*(a_valence/(a_valence+c_valence)))
    RI_valence = (2*(a_valence+b_valence))/(len(label)*(len(label)-1))
    JC_arousal = a_arousal/(a_arousal+b_arousal+c_arousal)
    FMI_arousal = sqrt((a_arousal/(a_arousal+b_arousal))*(a_arousal/(a_arousal+c_arousal)))
    RI_arousal = (2*(a_arousal+b_arousal))/(len(label)*(len(label)-1))
    JC_emotion = a_emotion/(a_emotion+b_emotion+c_emotion)
    FMI_emotion = sqrt((a_emotion/(a_emotion+b_emotion))*(a_emotion/(a_emotion+c_emotion)))
    RI_emotion = (2*(a_emotion+b_emotion))/(len(label)*(len(label)-1)) 
    return JC_valence, FMI_valence, RI_valence,JC_arousal, FMI_arousal, RI_arousal,JC_emotion, FMI_emotion, RI_emotion





def main():
    path_dataset = 'E:\python_program\cluster\数据\MAHNOB-HCI\EEG_feature.txt'
    path_label = 'E:\\python_program\\cluster\\数据\\MAHNOB-HCI\\valence_arousal_label.txt'
    path_emotion = 'E:\\python_program\\cluster\\数据\\MAHNOB-HCI\\EEG_emotion_category.txt'
    dataset = read_points(path_dataset)
    label = read_points(path_label)
    label_emotion = read_points(path_emotion)
    count = 0
    JC_valence = [0 for i in range(8)]
    FMI_valence = [0 for i in range(8)]
    RI_valence = [0 for i in range(8)]
    JC_arousal = [0 for i in range(8)]
    FMI_arousal = [0 for i in range(8)]
    RI_arousal = [0 for i in range(8)]
    JC_emotion = [0 for i in range(8)]
    FMI_emotion = [0 for i in range(8)]
    RI_emotion = [0 for i in range(8)]
    
    for k in range(2,10):
        label_pred = k_means(dataset,k)
        JC_valence[count], FMI_valence[count], RI_valence[count],JC_arousal[count], FMI_arousal[count], RI_arousal[count],JC_emotion[count], FMI_emotion[count], RI_emotion[count] = assess(label, label_emotion, label_pred)
        count = count + 1
    x = [2,3,4,5,6,7,8,9]
    plt.title('K_means MAHNOB-HCI Result Analysis')
    plt.plot(x,JC_valence, 'ro-', label='JC_valence')
    plt.plot(x,FMI_valence, 'go-', label='FMI_valence')
    plt.plot(x,RI_valence, 'yo-', label='RI_valence')
    plt.plot(x,JC_arousal,'rx-', label='JC_arousal',markersize=10)
    plt.plot(x,FMI_arousal, 'gx-', label='FMI_arousal',markersize=10)
    plt.plot(x,RI_arousal, 'yx-', label='RI_arousal',markersize=10)
    plt.plot(x,JC_emotion,'r|-', label='JC_emotion',markersize=10)
    plt.plot(x,FMI_emotion, 'g|-', label='FMI_emotion',markersize=10)
    plt.plot(x,RI_emotion, 'y|-', label='RI_emotion',markersize=10)
    
    plt.legend() # 显示图例

    plt.xlabel('k')
    plt.ylabel('assess_parameter')
    plt.show()
       
    '''print(JC_valence, FMI_valence, RI_valence,
          JC_arousal, FMI_arousal, RI_arousal)'''
        
if __name__ == "__main__":   
    main()

