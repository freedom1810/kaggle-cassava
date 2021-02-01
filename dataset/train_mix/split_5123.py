import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv('/home/hana/sonnh/kaggle-cassava/dataset/train_mix/new_mix.csv')
# print(df_new.columns)
print(df.head())
df_new = df[:21396]
df_old = df[21396:]

def fillter(df, a, b):
    df_a = df[df['fold'] == a]

    df_b = df[df['fold'] == b]

    df_ab = pd.concat([df_a, df_b])
    # print(tyoe(df_14))
    return df_ab

def re_split(df, a, b):
    df = fillter(df, a, b)
    df_new = pd.DataFrame({})
    df_new['image_id'] = df['image_id']
    df_new['label'] = df['label']
    df_new['fold'] = df['fold']
    df = df_new
    #print(df)
    #print(df_new)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=2020)
    X = df['image_id']
    Y = df['label']
    fold = a - 1
    for train_index, test_index in skf.split(X, Y):
        # print(test_index)
        fold += 1
        X_test = X.iloc[test_index]
        df.loc[df.image_id.isin(X_test), 'fold'] = fold

    # for i in range(len(df)):
    #     if df.iloc[i]['fold'] == 2:
    #         df.iloc[i]['fold'] = 3
    df['fold'] = df['fold'].replace(a + 1, b)
    return df

df_new_14 = re_split(df_new, 5, 1)
df_old_14 = re_split(df_old, 5, 1)

df_new_23 = re_split(df_new, 2, 3)
df_old_23 = re_split(df_old, 2, 3)


df_5 = df[df['fold'] == 4]

df_new_split_14 = pd.concat([df_new_14, df_old_14, df_5, df_new_23, df_old_23])
df_new_split_14.to_csv('new_mix_1234.csv', index = False)


