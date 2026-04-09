# -*- coding: utf-8 -*-
"""
项目主入口
运行命令：python main.py
"""
from src.config import TRAIN_PATH, TEST_PATH, SUBMIT_PATH
from src.preprocess import load_and_merge_data, process_features, split_train_test
from src.model import train_cross_validation
from src.predict import generate_submission

if __name__ == '__main__':
    print("=" * 50)
    print("🚀  Spaceship Titanic 预测任务启动")
    print("=" * 50)

    # 1. 加载数据
    df, train_ids, test_ids = load_and_merge_data(TRAIN_PATH, TEST_PATH)

    # 2. 特征工程
    df, drop_cols = process_features(df)

    # 3. 拆分数据集
    X, y, X_test = split_train_test(df, drop_cols)

    # 4. 训练模型
    test_preds = train_cross_validation(X, y, X_test)

    # 5. 生成提交
    generate_submission(test_ids, test_preds)

    print("\n✅  全部执行完成！")