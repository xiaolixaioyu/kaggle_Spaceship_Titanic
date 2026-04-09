# Spaceship Titanic | Kaggle 入门竞赛
Kaggle 经典入门表格竞赛「太空船泰坦尼克号」的完整解决方案，基于 LightGBM 实现，验证集准确率 **0.8123**，公开榜分数约 **0.808~0.815**。

---

## 🚀 项目简介
2912年，星际客轮「泰坦尼克号」在首航途中撞上了隐藏在尘埃云中的时空异常，近一半乘客被传送到了另一个维度。

本项目的任务是：利用飞船受损计算机系统中恢复的乘客个人记录，预测哪些乘客被异常现象传送，帮助救援队伍找回失踪人员。

这是一个标准的**二分类机器学习问题**，非常适合作为机器学习入门实战项目。

---

## 📁 项目结构
```
PythonProject2 [kaggle_Spaceship_Titanic]/
├── Data/                     # 数据集文件夹（需自行放入）
│   ├── train.csv             # 训练集（带标签）
│   ├── test.csv              # 测试集（无标签）
│   └── sample_submission.csv # 提交样例
├── src/                      # 核心源码
│   ├── __init__.py
│   ├── config.py             # 全局配置
│   ├── data_loader.py        # 数据加载
│   ├── preprocess.py         # 数据预处理 + 特征工程
│   ├── model.py              # 模型训练 + 5折交叉验证
│   └── predict.py            # 预测 + 生成提交文件
├── main.py                   # 项目入口（一键运行）
├── README.md                 # 项目说明
└── submission.csv            # 生成的Kaggle提交文件
```

---

## 🛠️ 环境依赖
确保你的 Python 环境安装了以下依赖：
```bash
pip install pandas numpy scikit-learn lightgbm
```

---

## 🚀 快速开始
### 1. 准备数据集
- 从 [Kaggle 比赛页面](sslocal://flow/file_open?url=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fspaceship-titanic%2Fdata&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=) 下载数据集
- 将 `train.csv`、`test.csv`、`sample_submission.csv` 放入 `Data/` 文件夹

### 2. 运行项目
在项目根目录执行以下命令，一键完成数据加载、特征工程、模型训练和预测：
```bash
python main.py
```

### 3. 查看结果
运行完成后，会在项目根目录生成 `submission.csv` 文件，同时控制台输出 5 折交叉验证的准确率：
```
Fold 1 准确率: 0.8114
Fold 2 准确率: 0.8125
Fold 3 准确率: 0.8148
Fold 4 准确率: 0.8199
Fold 5 准确率: 0.8026

平均验证准确率: 0.8123
✅ 提交文件已生成：xxx/submission.csv
```

---

## 📊 模型与结果
- **模型**：LightGBM 梯度提升树
- **验证方式**：5 折分层交叉验证（Stratified K-Fold）
- **验证集准确率**：0.8123
- **Kaggle 公开榜分数**：约 0.808~0.815（排名前 30%）

### 核心特征工程
1.  拆解 `PassengerId` 提取旅行团信息（团体大小、是否独自旅行）
2.  拆解 `Cabin` 提取甲板、舱号、舷侧信息
3.  聚合 5 个消费列生成总消费、是否消费特征
4.  利用「有消费的人一定没在冷冻睡眠」的业务逻辑智能填充缺失值
5.  分组填充年龄、母星、目的地等特征

---

## 📂 文件说明
| 文件 | 作用 |
|------|------|
| `src/config.py` | 统一管理所有路径、模型参数、特征列表 |
| `src/data_loader.py` | 负责加载和合并训练集、测试集 |
| `src/preprocess.py` | 所有数据清洗、缺失值填充、特征工程逻辑 |
| `src/model.py` | LightGBM 模型训练 + 5 折交叉验证 |
| `src/predict.py` | 生成符合 Kaggle 格式的提交文件 |
| `main.py` | 项目主入口，串联所有流程 |

---

## 📤 提交到 Kaggle
1.  登录 [Kaggle 比赛页面](sslocal://flow/file_open?url=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fspaceship-titanic&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
2.  点击顶部导航栏的 **Submit Predictions**
3.  上传项目根目录下的 `submission.csv` 文件
4.  勾选同意比赛规则，点击 **Make Submission**
5.  等待 10~30 秒，即可看到你的公开榜分数和排名

> 该比赛每天最多提交 5 次，提交次数在 UTC 时间每天 0 点重置。

---

## 📈 进阶优化方向
如果想冲击更高的排名（0.82+，前 10%），可以尝试以下优化：
1.  **深度挖掘旅行团特征**：计算同一个旅行团中其他成员的被传送概率（K-Fold 内计算，防止数据泄露）
2.  **更精细的缺失值填充**：使用 KNN 或模型预测缺失值，替代简单的中位数/众数填充
3.  **模型集成**：将 LightGBM、XGBoost、CatBoost 三个模型的预测结果加权平均
4.  **贝叶斯调参**：使用 Optuna 优化 LightGBM 的超参数
5.  **特征交叉**：构造 `CryoSleep × TotalSpent`、`Deck × Side` 等交叉特征

---

## 📄 许可证
MIT License