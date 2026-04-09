# ====================== 主程序入口 ======================
# 导入自定义模块
from data_loader import load_raw_data, merge_train_test
from features import build_all_features

# ====================== 执行流程 ======================
if __name__ == '__main__':
    # 1. 加载原始数据
    train, test, sub = load_raw_data()

    # 2. 合并训练+测试集
    df = merge_train_test(train, test)

    # 3. 执行全部特征工程
    df = build_all_features(df)

    # ====================== 后续代码 ======================
    # 这里可以继续写：模型训练、交叉验证、预测、提交文件生成
    print("\n✅ 数据处理完成！")