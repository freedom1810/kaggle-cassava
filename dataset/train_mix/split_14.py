import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv('/home/hana/sonnh/kaggle-cassava/dataset/train_mix/new_mix.csv')
# print(df_new.columns)
print(df.head())
df_new = df[:21396]
df_old = df[21396:]

def fillter_14(df):
    df_1 = df[df['fold'] == 1]

    df_4 = df[df['fold'] == 4]

    df_14 = pd.concat([df_1, df_4])
    # print(tyoe(df_14))
    return df_14

def re_split(df):
    df = fillter_14(df)
    # df = df.reindex(list(range(len(df))))
    df_new = pd.DataFrame({})
    df_new['image_id'] = df['image_id']
    df_new['label'] = df['label']
    df_new['fold'] = df['fold']
    df = df_new
    print(df)
    print(df_new)
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    X = df['image_id']
    Y = df['label']
    fold = 0
    for train_index, test_index in skf.split(X, Y):
        fold += 1
        # print(test_index)
        X_test = X.iloc[test_index]
        df.loc[df.image_id.isin(X_test), 'fold'] = fold

    # for i in range(len(df)):
    #     if df.iloc[i]['fold'] == 2:
    #         df.iloc[i]['fold'] = 3
    df['fold'] = df['fold'].replace(2,4)
    return df

df_new = re_split(df_new)
df_old = re_split(df_old)



df_2 = df[df['fold'] == 2]
df_3 = df[df['fold'] == 3]
df_5 = df[df['fold'] == 5]

df_new_split_14 = pd.concat([df_2, df_3, df_5, df_new, df_old])

df_new_split_14.to_csv('new_mix_14.csv', index = False)


