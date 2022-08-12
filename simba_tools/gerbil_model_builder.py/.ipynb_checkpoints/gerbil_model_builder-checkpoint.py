import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support,
                             precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class GerbilModelCreator(object):
    def __init__(self,
                 in_path: str=None,
                 train_size: float=None,
                 test_size: float=None):

        self.in_path = in_path
        self.test_size, self.train_size = test_size, train_size
        self.data_df = pd.read_parquet(in_path)
        self.target_ratio = round((self.data_df['target'].sum() / len(self.data_df)), 4)
        self.unique_videos = self.data_df['VIDEO'].unique()

    def find_and_count_labelled_bouts(self):
        bouts_df_lst = []
        for video in self.unique_videos:
            video_df = self.data_df[self.data_df['VIDEO'] == video]
            grouped_df = pd.DataFrame()
            start_stop_df = pd.DataFrame()
            v = (video_df['target'] != video_df['target'].shift()).cumsum()
            u = video_df.groupby(v)['target'].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            grouped_df['groups'] = video_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            start_stop_df[['START_FRAME', 'STOP_FRAME']] = pd.DataFrame(grouped_df['groups'].tolist())
            start_stop_df.insert(loc=0, column='VIDEO', value=video)
            bouts_df_lst.append(start_stop_df)
        self.bouts_df = pd.concat(bouts_df_lst, axis=0).reset_index(drop=True)
        self.bouts_len = len(self.bouts_df)

    def sample_targets(self):
        train_target_samples, test_target_samples = train_test_split(self.bouts_df, test_size=self.test_size)
        train_target_df_lst, test_target_df_lst = [], []
        for video in self.bouts_df['VIDEO'].unique():
            video_df = self.data_df[self.data_df['VIDEO'] == video]
            video_df_train_target_samples = train_target_samples[train_target_samples['VIDEO'] == video]
            video_df_test_target_samples = test_target_samples[test_target_samples['VIDEO'] == video]
            annotations_idx_target_train = list(video_df_train_target_samples.apply(lambda x: list(range(int(x['START_FRAME']), int(x['STOP_FRAME']) + 1)), 1))
            annotations_idx_target_test = list(video_df_test_target_samples.apply(lambda x: list(range(int(x['START_FRAME']), int(x['STOP_FRAME']) + 1)), 1))
            annotations_idx_target_train = [i for s in annotations_idx_target_train for i in s]
            annotations_idx_target_test = [i for s in annotations_idx_target_test for i in s]
            video_target_train_df = video_df[video_df['FRAME'].index.isin(annotations_idx_target_train)]
            video_target_test_df = video_df[video_df['FRAME'].index.isin(annotations_idx_target_test)]
            train_target_df_lst.append(video_target_train_df)
            test_target_df_lst.append(video_target_test_df)

        self.target_train_df = pd.concat(train_target_df_lst, axis=0)
        self.target_test_df = pd.concat(test_target_df_lst, axis=0)

    def sample_non_targets(self):
        non_target_df = self.data_df[self.data_df['target'] == 0]
        non_target_test_n = int(len(self.target_test_df) / self.target_ratio)
        annotations_idx_nontarget_train = np.random.choice(non_target_df.index, len(self.target_train_df), replace=False)
        train_non_target_df = non_target_df.drop(annotations_idx_nontarget_train)
        annotations_idx_nontarget_test = np.random.choice(train_non_target_df.index, non_target_test_n, replace=False)

        self.non_target_train_df = self.data_df[self.data_df['FRAME'].index.isin(annotations_idx_nontarget_train)]
        self.non_target_test_df = self.data_df[self.data_df['FRAME'].index.isin(annotations_idx_nontarget_test)]

    def create_model(self):
        self.train_df = pd.concat([self.non_target_train_df, self.target_train_df], axis=0).reset_index(drop=True).fillna(0)
        self.test_df = pd.concat([self.non_target_test_df, self.target_test_df], axis=0).reset_index(drop=True).fillna(0)
        self.train_df_identifiers = pd.concat([self.train_df.pop(x) for x in ['VIDEO', 'FRAME', 'animal_1_x', 'animal_1_y', 'animal_2_x','animal_2_y']], axis=1)
        self.test_df_identifiers = pd.concat([self.test_df.pop(x) for x in ['VIDEO', 'FRAME', 'animal_1_x', 'animal_1_y', 'animal_2_x', 'animal_2_y']],axis=1)
        self.train_y, self.test_y = self.train_df.pop('target'), self.test_df.pop('target')
        self.rf_clf = RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=-1, criterion='gini', min_samples_leaf=1, bootstrap=True, verbose=1)
        self.rf_clf.fit(self.train_df.values, self.train_y)

    def test_model(self):
        y_proba = self.rf_clf.predict_proba(self.test_df.values)[:, 1]
        self.y_pred = np.where(y_proba > 0.5, 1, 0)
        precision_recall_fscore = precision_recall_fscore_support(self.test_y, self.y_pred, average='binary', pos_label=1)
        p, r, t = precision_recall_curve(self.test_y, y_proba)
        fig, ax = plt.subplots()
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.plot(r, p, color='blue')
        plt.show()
        importances = list(self.rf_clf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.train_df.columns, importances)]
        self.gini_importances_df = pd.DataFrame(feature_importances, columns=['FEATURE', 'FEATURE_IMPORTANCE']).sort_values(by=['FEATURE_IMPORTANCE'], ascending=False)

    # # def descriptive_statistics_predictions(self):
    #     self.test_df = pd.concat([self.test_df_identifiers, self.test_df], axis=1)
    #     prediction_arr = np.vstack([self.test_y, self.y_pred]).T
    #     true_positive_idx = np.argwhere((prediction_arr[:, 0] == 1) & (prediction_arr[:, 1] == 1)).flatten()
    #     false_positive_idx = np.argwhere((prediction_arr[:, 0] == 0) & (prediction_arr[:, 1] == 1)).flatten()
    #     true_negative_idx = np.argwhere((prediction_arr[:, 0] == 0) & (prediction_arr[:, 1] == 0)).flatten()
    #     false_negative_idx = np.argwhere((prediction_arr[:, 0] == 1) & (prediction_arr[:, 1] == 0)).flatten()
    #
    #     false_positives_df = self.test_df[self.test_df.index.isin(false_positive_idx)]
    #     false_negative_df = self.test_df[self.test_df.index.isin(false_negative_idx)]
    #     print(false_negative_df['VIDEO'].unique())

test = GerbilModelCreator(in_path='_legacy/gerbil_data/featurized/features_20220731153157.parquet',
                          train_size=0.8,
                          test_size=0.2)
test.find_and_count_labelled_bouts()
test.sample_targets()
test.sample_non_targets()
test.create_model()
test.test_model()
