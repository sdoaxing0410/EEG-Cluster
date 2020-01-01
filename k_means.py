#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans


def k_means(dataset, k):
    # 构造一个聚类数为k的聚类器
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dataset)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    print('\n\n-------------------------------k-means聚类结果---------------------------------\n\n')
    for kind in range(k):
        arr = np.where(label_pred == kind)
        print('----------------第%d个聚类-----------------' % (kind + 1))
        count = 0
        for num in arr[0]:
            print(num, end=' ')
            count = count + 1
            if count % 25 == 0:
                print('\n')
        print('\n')
    print(label_pred)


# 聚类结果评价
def assess(label, label_pred, source):
    # 外部指标评价 a=|SS|,b=|SD|,c=|DS|,d=|DD|
    # valence愉悦度
    a_valence = 0
    b_valence = 0
    c_valence = 0
    d_valence = 0
    # arousal唤醒度
    a_arousal = 0
    b_arousal = 0
    c_arousal = 0
    d_arousal = 0
    if source == 'MAHNOB':  # MAHNOB数据源还给了emotion标签
        a_emotion = 0
        b_emotion = 0
        c_emotion = 0
        d_emotion = 0





