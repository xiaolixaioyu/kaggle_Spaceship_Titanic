# -*- coding: utf-8 -*-
"""
数据预处理 + 特征工程
✅ 只用强特征
✅ 删除 Name/冗余列 等无用特征
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import EXPENSE_COLS, CAT_COLS

def load_and_merge_data(train_path, test_path):
    """加载并合并训练/测试集（统一预处理）"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0, ignore_index=True)
    return df, train['PassengerId'], test['PassengerId']
    #没太明白为什么要返回这两个id列呢
def process_features(df):
    """核心特征工程（无无用特征，全是高价值）"""
    # ===================== 1. 处理 PassengerId（旅行团） =====================
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    #true  转换为1  flase转换为0
    # ===================== 2. 处理 Cabin（拆分甲板/舱号/舷侧） =====================
    df['Cabin'] = df['Cabin'].fillna('U/0/U')

    cabin_split = df['Cabin'].str.split('/', expand=True)
    #形成多列

    df['Deck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce').fillna(0)
    df['Side'] = cabin_split[2]

    # ===================== 3. 处理消费金额（核心强特征） =====================
    df[EXPENSE_COLS] = df[EXPENSE_COLS].fillna(0)
    df['TotalSpent'] = df[EXPENSE_COLS].sum(axis=1)
    #求解消费总金额 消费总金额大于0 新增一列是否消费转换为整数
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)

    # ===================== 4. 处理 CryoSleep（最强单特征） =====================
    #产生一个布尔掩码 标记同时满足两个条件的行
    mask = df['CryoSleep'].isna() & (df['HasSpent'] == 1)
    #用loc定位到所有满足mask条件的行，修改成为未休眠
    df.loc[mask, 'CryoSleep'] = False
    #填充剩余空值并且转换为数字
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)

    # ===================== 5. 填充类别特征 =====================
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    #缺失值填充他们的众数
    # ===================== 6. 处理年龄 =====================
    df['Age'] = df.groupby(['HomePlanet', 'CryoSleep'])['Age'].transform(
        lambda x: x.fillna(x.median())
    ).fillna(df['Age'].median())
    #年龄缺失转换为中位数  不行的话转换为全部的中位数

    # ===================== 7. 处理VIP =====================
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    #缺失直接填充False 转换为整数

    # ===================== 8. 类别特征编码 =====================
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        #讲这些列的数据先转换为字符串在转换为数字 依据LabelEncoder进行编码

    # ===================== 9. 删除无用列（你要求的！） =====================
    drop_cols = [
        'PassengerId', 'Name', 'Cabin', 'Group',  # 无用/冗余列
        'Transported', 'is_train'
    ]
    return df, drop_cols

def split_train_test(df, drop_cols):
    """拆分回训练集/测试集"""
    train_df = df[df['is_train'] == 1].copy()
    test_df = df[df['is_train'] == 0].copy()
    #删除无用列
    X = train_df.drop(columns=drop_cols)
    #将是否被传送这列转为数字
    y = train_df['Transported'].astype(int)
    #测试集也删除无用列
    X_test = test_df.drop(columns=drop_cols)

    return X, y, X_test