#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Clusterer import *
from config import *


if __name__ == '__main__':
    feature_list = []
    for i, _ in enumerate(range(1, 161)):
        feature_list.append('feature' + str(i + 1))
    df_feature = pd.read_table(MAHNOB_EEG_feature_file_path, names=feature_list, delim_whitespace=True)
    num_samples = df_feature.shape[0]
    df_emotion_category = pd.read_table(MAHNOB_EEG_emotion_category_file_path, names=['EMOTION_CATEGORY'],
                                        delim_whitespace=True)
    # 归一化
    scaler = StandardScaler().fit(df_emotion_category)
    scaler.transform(df_emotion_category)
    # subject、video
    df_subject_video = pd.read_table(MAHNOB_subject_video_file_path, names=['subjectID', 'videoID'],
                                     delim_whitespace=True)
    df_subject_ID = df_subject_video[['subjectID']]
    df_video_ID = df_subject_video[['videoID']]

    # valence、arousal
    df_valence_arousal_label = pd.read_table(MAHNOB_valence_arousal_label_file_path, names=['valence', 'arousal'],
                                             delim_whitespace=True)
    df_valence = df_valence_arousal_label[['valence']]
    df_arousal = df_valence_arousal_label[['arousal']]

    # KMeans
    print('----------------KMeans---------------')
    # 1、Emotion Category
    k_means(df_feature, 9, df_emotion_category['EMOTION_CATEGORY'], '1、Emotion Category')
    # 2、Subject ID
    k_means(df_feature, 27, df_subject_ID['subjectID'], '2、Subject ID')
    # 3、VideoID
    k_means(df_feature, 20, df_video_ID['videoID'], '3、VideoID')
    # 4、valence
    k_means(df_feature, 2, df_valence['valence'], '4、valence')
    # 5、arousal
    k_means(df_feature, 2, df_arousal['arousal'], '5、arousal')

    # GaussianMixture
    print('\n\n\n----------------GaussianMixture-----------------')
    # 1、Emotion Category
    gaussian_mixture(df_feature, 9, df_emotion_category['EMOTION_CATEGORY'], '1、Emotion Category')
    # 2、Subject ID
    gaussian_mixture(df_feature, 27, df_subject_ID['subjectID'], '2、Subject ID')
    # 3、VideoID
    gaussian_mixture(df_feature, 20, df_video_ID['videoID'], '3、VideoID')
    # 4、valence
    gaussian_mixture(df_feature, 2, df_valence['valence'], '4、valence')
    # 5、arousal
    gaussian_mixture(df_feature, 2, df_arousal['arousal'], '5、arousal')

