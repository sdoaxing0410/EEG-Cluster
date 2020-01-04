#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
from visualization import MDS
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score


def k_means(dataset, k, label, source):
    # 构造一个聚类数为k的聚类器并聚类
    estimator = KMeans(n_clusters=k).fit(dataset)
    predicted_label = estimator.predict(dataset)
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和

    # 外部指标评价ARI
    print('================%s==================' % source)
    print('外部指标ARI:%s' % adjusted_rand_score(label, predicted_label))
    # 内部指标：每个样本距离他们各自的簇中心的距离的平均值
    num_samples = dataset.shape[0]
    print('内部指标avg center distance:%s' % (inertia/num_samples))
    print('内部指标DBI：%s' % davies_bouldin_score(dataset, predicted_label))
    MDS(dataset, predicted_label, label)


def gaussian_mixture(dataset, k, label, source):
    estimator = mixture.GaussianMixture(n_components=k).fit(dataset)  # 构造高斯混合聚类器聚类
    predicted_label = estimator.predict(dataset)

    # 外部指标评价ARI
    print('================%s==================' % source)
    print('外部指标ARI:%s' % adjusted_rand_score(label, predicted_label))
    # 内部指标：每个样本距离他们各自的簇中心的距离的平均值
    print('内部指标DBI：%s' % davies_bouldin_score(dataset, predicted_label))
    MDS(dataset, predicted_label, label)





