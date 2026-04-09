# ====================== 数据加载模块 ======================
import pandas as pd
from config import TRAIN_PATH, TEST_PATH, SUB_PATH
#  数据读取合并与特征工程解耦

def load_raw_data():
    """
    读取原始训练集、测试集、提交样例
    return: train, test, submission
    """
    print('==========1 加载数据=================')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SUB_PATH)

    # 打印数据形状
    print('训练集形状:', train.shape)
    print('测试集形状:', test.shape)
    return train, test, sub


def merge_train_test(train, test):
    """
    合并训练集+测试集，标记训练/测试样本
    return: 合并后的总数据集 df
    """
    print("==================2.合并数据===================")
    # 标记训练/测试集
    train['is_train'] = 1
    test['is_train'] = 0
    # 上下拼接 + 重置索引
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    print('合并后总数据形状:', df.shape)
    return df