# ====================== 项目配置文件 ======================
# 数据集路径（只需要在这里修改一次路径）
DATA_DIR = r"D:\数据集\泰坦尼克号飞船"
TRAIN_PATH = r"D:\数据集\泰坦尼克号飞船\train.csv"
TEST_PATH = r"D:\数据集\泰坦尼克号飞船\test.csv"
SUBMIT_PATH = r"D:\数据集\泰坦尼克号飞船\sample_submission.csv"

# 原始消费列（核心特征）
EXPENSE_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# 类别特征（需要编码）
CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']



# 交叉验证折数
N_FOLDS = 5