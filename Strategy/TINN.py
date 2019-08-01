from fastai.tabular import *

class TINN():
    def __init__(self, data_df, dep_var, categories):
        procs = [FillMissing, Categorify, Normalize]
        valid_idx = range(len(data_df) - SPLIT_NUM, len(data_df))
        cat_names = categories

        self.data = TabularDataBunch.from_df(path, data_df, dep_var, valid_idx=valid_idx, procs=procsm cat_names=cat_names)
        print(data.train_ds.cont_names)

    def create_learner(self):
        self.learner = tabular_learner(self.data, layers=[200,100], emb_szs={'native-country':10}, metrics=accuracy)




