import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
df_new = pd.read_csv('new.csv')
print(df_new.columns)
X = df_new['image_id']
Y = df_new['label']
df_new['fold'] = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 2021)
fold = 0
for train_index, test_index in skf.split(X, Y):
    fold += 1
    #print(df_new[df_new.index.isin(test_index)])
    #print(df_new.index)
    #exit()
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_test = X[test_index]
    df_new.loc[df_new.image_id.isin(X_test), 'fold'] = fold
    #print(fold)
    #print(df_new[df_new.fold == 10])
    #exit()

df_old = pd.read_csv('old.csv')
#print(df_old.columns)
X = df_old['image_id']
Y = df_old['label']
df_old['fold'] = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 2021)
fold = 0
for train_index, test_index in skf.split(X, Y):
    fold += 1
    X_test = X[test_index]
    #print("TRAIN:", train_index, "TEST:", test_index)
    df_old.loc[df_old.image_id.isin(X_test), 'fold'] = fold
    

frames = [df_new, df_old]
df_all = pd.concat(frames)
df_all.to_csv('new_mix_v2.csv', index = False)
