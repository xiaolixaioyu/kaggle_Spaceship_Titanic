# ====================== 特征工程模块 ======================
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import EXP_COLS


# -------------------- 工具函数 --------------------
def label_encode_single_col(df, col):
    """单列标签编码（复用函数，减少重复代码）"""
    df[col] = LabelEncoder().fit_transform(df[col])
    return df


# -------------------- 1. ID特征处理 --------------------
def process_id_features(df):
    print('=================3.选择id特征=================')
    # 拆分旅行团ID + 个人编号
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['Group_Person'] = df['PassengerId'].str.split('_').str[1].astype(int)
    # 旅行团人数 + 是否独自旅行
    df['Group_Size'] = df.groupby('Group')['Group'].transform('count')
    df['Is_Alone'] = (df['Group_Size'] == 1).astype(int)
    return df


# -------------------- 2. 船舱Cabin特征处理 --------------------
def process_cabin_features(df):
    print('=======================6. 船舱号 Cabin特征处理===================')
    # 缺失值填充 + 拆分甲板/舱号/舷侧
    df['Cabin'] = df['Cabin'].fillna('U/0/U')
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Cabin_Num'] = cabin_split[1]
    df['Side'] = cabin_split[2]
    # 舱号转数字
    df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce').fillna(0)
    # 类别编码
    df = label_encode_single_col(df, 'Deck')
    df = label_encode_single_col(df, 'Side')
    return df


# -------------------- 3. 出发星球HomePlanet处理 --------------------
def process_homeplanet(df):
    print('=============4.选择出发星球 HomePlanet特征===========')
    # 先填充分组列缺失值（必须先做！）
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['CryoSleep'] = df['CryoSleep'].fillna(False)

    # 分组众数填充
    df['HomePlanet'] = df.groupby(['Deck', 'CryoSleep'])['HomePlanet'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Earth')
    )
    # 全局兜底填充 + 标签编码
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df = label_encode_single_col(df, 'HomePlanet')
    return df


# -------------------- 4. 冷冻睡眠CryoSleep+消费特征处理 --------------------
def process_cryosleep_spending(df):
    print('=========================5 cryosleep 冷冻睡眠======================')
    # 消费列缺失值填充
    df[EXP_COLS] = df[EXP_COLS].fillna(0)
    # 构造总消费、是否消费
    df['Total_Spent'] = df[EXP_COLS].sum(axis=1)
    df['Has_Spent'] = (df['Total_Spent'] > 0).astype(int)

    # 业务逻辑填充睡眠状态
    mask = df['CryoSleep'].isna() & (df['Has_Spent'] == 1)
    df.loc[mask, 'CryoSleep'] = False
    # 众数填充 + 转数字
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    return df
#--------------------5. 目的地Destination特征处理-----------------------
def process_Destination(df):
    # 分组填充缺失值
    df['Destination'] = df.groupby(['HomePlanet', 'Deck'])['Destination'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'TRAPPIST-1e')
    )
    #按照HomePlanet Deck进行分组 选择Destination列进行转换  对这一列缺失的值填充众数如果没有众数则填充
    #TRAPPIST-1e
    # 兜底填充
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    # 编码
    df['Destination'] = LabelEncoder().fit_transform(df['Destination'])
    #自动进行编码将目的地从文字转换为编码
    return df
#--------------------5. Age年龄特征处理-----------------------
def Process_Age(df):
    # 1. 分组填充：按睡眠+星球分组填中位数
    df['Age'] = df.groupby(['CryoSleep', 'HomePlanet'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    #按照CryoSleep HomePlanet进行分组 选择Age这一列进行转换  将缺失值填充为中位数
    # 兜底填充
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 2. 年龄分箱（核心：儿童单独成组）
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 40, 60, 80],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
'''
bins：分箱边界（必须是有序数字）
自动切成 5 个区间：
0 ~ 12 → 儿童
12 ~ 18 → 青少年
18 ~ 40 → 青年
40 ~ 60 → 中年
60 ~ 80 → 老年
'''
'''
labels=[0,1,2,3,4]
labels：每个区间对应的标签
对应关系：
0~12 岁 → 标签 0
12~18 岁 → 标签 1
18~40 岁 → 标签 2
40~60 岁 → 标签 3
60~80 岁 → 标签 4
'''
    return df
# -------------------- 总特征工程入口 --------------------
def build_all_features(df):
    """
    按正确顺序执行所有特征处理
    （顺序严格遵循原代码，保证结果一致）
    """
    df = process_id_features(df)
    df = process_cabin_features(df)
    df = process_homeplanet(df)
    df = process_cryosleep_spending(df)
    return df