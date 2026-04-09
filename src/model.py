# -*- coding: utf-8 -*-
"""
模型训练 + 5折交叉验证
【无版本冲突，绝对可运行】
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from src.config import N_FOLDS

def train_cross_validation(X, y, X_test):
    """交叉验证训练，返回测试集预测概率"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    val_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 极简模型参数，全版本兼容
        model = lgb.LGBMClassifier(
            objective='binary',
            learning_rate=0.05,
            n_estimators=200,
            random_state=42,
            verbose=-1
        )

        # 🔥 核心修复：只保留最基础的 fit，无任何冲突参数！
        model.fit(X_tr, y_tr)

        # 验证计算准确率
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        val_scores.append(acc)
        print(f"Fold {fold+1} 准确率: {acc:.4f}")

        # 测试集预测
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    print(f"\n平均验证准确率: {np.mean(val_scores):.4f}")
    return test_preds