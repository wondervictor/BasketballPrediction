# -*- coding: utf-8 -*-

import xgboost as xgb


class Boost(object):

    def __init__(self):
        self.params = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight': 1400.0 / 13458.0,
        'eval_metric': 'auc',
        'gamma':0.1,
        'max_depth':8,
        'lambda ': 550,
        'subsample':0.7,
        'colsample_bytree':0.4,
        'min_child_weight':3,
        'eta': 0.02,
        'seed':231,
        'nthread':2
        }

    def train(self, train_x, train_y):
        d_train = xgb.DMatrix(train_x, label=train_y)
        self.model = xgb.train(self.params, d_train, num_boost_round=1000)

    def predict(self, x):

        x = xgb.DMatrix(x)
        pred_y = self.model.predict(x, ntree_limit=self.model.best_ntree_limit)
        print(pred_y)

    def save_model(self):
        self.model.save_model('model_params/xgboost_model_params.model')

    def load_model(self):
        self.model = xgb.Booster({'nthread':1})
        self.model.load_model('model_params/xgboost_model_params.model')
        pass

