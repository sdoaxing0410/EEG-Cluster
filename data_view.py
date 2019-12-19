#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import mne


class DataView:
    def __init__(self, file_path):
        self.file_path = file_path
        df = pd.read_table(self.file_path)
        self.data = df

    # @property
    # def


if __name__ == '__main__':
    # MAHNOB
    MAHNOB_emotion_category_data = DataView(MAHNOB_EEG_emotion_category_file_path)
    MAHNOB_feature_data = DataView(MAHNOB_EEG_feature_file_path)
    MAHNOB_arousal_label_data = DataView(MAHNOB_valence_arousal_label_file_path)
    # DEAP
    DEAP_feature_data = DataView(DEAP_EEG_feature_file_path)
    DEAP_arousal_label_data = DataView(DEAP_valence_arousal_label_file_path)

