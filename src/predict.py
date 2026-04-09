# -*- coding: utf-8 -*-
"""
预测结果 + 生成提交文件
"""
import pandas as pd
from src.config import DATA_DIR

def generate_submission(test_ids, preds):
    """生成Kaggle提交文件"""
    sub = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': preds > 0.5
    })
    sub_path = f"{DATA_DIR}/submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n提交文件已生成：{sub_path}")
    return sub