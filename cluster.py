#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from k_means import *
from config import *


if __name__ == '__main__':
    # MAHNOB
    MAHNOB_feature_data = pd.read_table(MAHNOB_EEG_feature_file_path)
    MAHNOB_arousal_label_data = pd.read_table(MAHNOB_valence_arousal_label_file_path)
    MAHNOB_emotion_category_data = pd.read_table(MAHNOB_EEG_emotion_category_file_path)
    # DEAP
    DEAP_feature_data = pd.read_table(DEAP_EEG_feature_file_path)
    DEAP_arousal_label_data = pd.read_table(DEAP_valence_arousal_label_file_path)

    count = 0
    JC_valence = [0 for i in range(8)]  # Jaccard系数
    FMI_valence = [0 for i in range(8)]  # FM指数
    RI_valence = [0 for i in range(8)]  # Rand指数
    JC_arousal = [0 for i in range(8)]
    FMI_arousal = [0 for i in range(8)]
    RI_arousal = [0 for i in range(8)]
    JC_emotion = [0 for i in range(8)]
    FMI_emotion = [0 for i in range(8)]
    RI_emotion = [0 for i in range(8)]
    for k in range(2, 10):  # 训练2—9共8个k值
        label_prod = k_means(MAHNOB_feature_data, k)
        JC_valence[count], FMI_valence[count], RI_valence[count], JC_arousal[count], FMI_arousal[count], RI_arousal[count], JC_emotion[count], FMI_emotion[count], RI_emotion[count] = assess(MAHNOB_arousal_label_data, label_prod, 'MAHNOB')
        count = count + 1
    x = [2, 3, 4, 5, 6, 7, 8, 9]
    plt.title('K_means MAHNOB-HCI Result Analysis')
    plt.plot(x, JC_valence, 'ro-', label='JC_valence')
    plt.plot(x, FMI_valence, 'go-', label='FMI_valence')
    plt.plot(x, RI_valence, 'yo-', label='RI_valence')
    plt.plot(x, JC_arousal, 'rx-', label='JC_arousal', markersize=10)
    plt.plot(x, FMI_arousal, 'gx-', label='FMI_arousal', markersize=10)
    plt.plot(x, RI_arousal, 'yx-', label='RI_arousal', markersize=10)
    plt.plot(x, JC_emotion, 'r|-', label='JC_emotion', markersize=10)
    plt.plot(x, FMI_emotion, 'g|-', label='FMI_emotion', markersize=10)
    plt.plot(x, RI_emotion, 'y|-', label='RI_emotion', markersize=10)

    plt.legend()  # 显示图例

    plt.xlabel('k')
    plt.ylabel('assess_parameter')
    plt.show()

    print(JC_valence, FMI_valence, RI_valence,
          JC_arousal, FMI_arousal, RI_arousal)





