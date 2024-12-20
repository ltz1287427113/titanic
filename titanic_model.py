# 导入所需的基础库
import csv
import statistics
import random
import os  # 添加os模块
import sys

"""
titanic_model.py用于构建预测模型主要功能：数据预处理（处理缺失值、特征工程），
、准备训练数据，构建和训练预测模型，评估模型性能，对测试集进行预测，生成预测结果文件。
"""

# 在文件开头添加:
class TeeOutput:
    """
    类TeeOutput用于同时将输出写入终端和文件，类似于Unix的tee命令。

    属性:
    - terminal: 保存当前的sys.stdout，即终端的输出。
    - file: 打开指定的文件用于写入输出。
    """

    def __init__(self, filename):
        """
        初始化TeeOutput实例。

        参数:
        - filename: 字符串，指定用于记录输出的文件名。
        """
        self.terminal = sys.stdout  # 保存当前终端的输出
        self.file = open(filename, 'w', encoding='utf-8')  # 打开指定文件，准备写入

    def write(self, message):
        """
        同时将消息写入终端和文件。

        参数:
        - message: 字符串，要写入的消息。
        """
        self.terminal.write(message)  # 将消息写入终端
        self.file.write(message)  # 将消息写入文件

    def flush(self):
        """
        刷新终端和文件的输出缓冲，确保所有输出即时写入。
        """
        self.terminal.flush()  # 刷新终端的输出缓冲
        self.file.flush()  # 刷新文件的输出缓冲


# 创建Result文件夹（如果不存在）
if not os.path.exists('Result'):
    os.makedirs('Result')

# 重定向输出
sys.stdout = TeeOutput(os.path.join('Result', 'model_output.md'))

def load_data(filename):
    """从CSV文件加载数据"""
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

def handle_missing_values(data):
    """处理数据中的缺失值"""
    # 收集所有非空年龄值
    ages = [float(row['Age']) for row in data if row['Age'] != '']
    mean_age = statistics.mean(ages)
    
    # 获取最常见的登船港口
    embarked = [row['Embarked'] for row in data if row['Embarked'] != '']
    most_common_embarked = max(set(embarked), key=embarked.count)
    
    # 收集所有非空票价
    fares = [float(row['Fare']) for row in data if row['Fare'] != '']
    mean_fare = statistics.mean(fares)
    
    # 处理缺失值
    for row in data:
        if row['Age'] == '':
            row['Age'] = str(mean_age)
        if row['Embarked'] == '':
            row['Embarked'] = most_common_embarked
        if row['Fare'] == '':
            row['Fare'] = str(mean_fare)
    return data

def feature_engineering(data):
    """特征工程：转换和创建特征"""
    processed_data = []
    
    for row in data:
        features = {}
        
        # 基本特征
        features['is_female'] = 1 if row['Sex'] == 'female' else 0
        features['Pclass'] = int(row['Pclass'])
        features['Age'] = float(row['Age'])
        features['Fare'] = float(row['Fare'])
        
        # 家庭特征
        features['family_size'] = int(row['SibSp']) + int(row['Parch'])
        features['is_alone'] = 1 if features['family_size'] == 0 else 0
        features['has_family'] = 1 if features['family_size'] > 0 else 0
        
        # 年龄分组 (更细致的分组)
        if features['Age'] < 12:
            features['age_group'] = 0  # 儿童
        elif features['Age'] < 18:
            features['age_group'] = 1  # 青少年
        elif features['Age'] < 35:
            features['age_group'] = 2  # 青年
        elif features['Age'] < 50:
            features['age_group'] = 3  # 中年
        else:
            features['age_group'] = 4  # 老年
        
        # 票价分组 (基于数据分析的分组)
        if features['Fare'] == 0:
            features['fare_group'] = 0  # 免费
        elif features['Fare'] < 10:
            features['fare_group'] = 1  # 低价
        elif features['Fare'] < 30:
            features['fare_group'] = 2  # 中价
        elif features['Fare'] < 100:
            features['fare_group'] = 3  # 高价
        else:
            features['fare_group'] = 4  # 豪华
            
        # 登船港口编码
        port_mapping = {'S': 0, 'C': 1, 'Q': 2}
        features['Embarked'] = port_mapping.get(row['Embarked'], 0)
        
        # 重要组合特征
        features['class_sex'] = features['is_female'] * 10 + features['Pclass']  # 性别和舱位组合
        features['age_class'] = features['age_group'] * 3 + features['Pclass']  # 年龄和舱位组合
        features['family_class'] = min(features['family_size'], 3) * 3 + features['Pclass']  # 家庭和舱位组合
        
        if 'Survived' in row:
            features['Survived'] = int(row['Survived'])
            
        processed_data.append(features)
    
    return processed_data

def prepare_data(processed_data):
    """准备训练数据"""
    feature_names = [
        'is_female', 'Pclass', 'Age', 'Fare', 
        'family_size', 'is_alone', 'has_family',
        'age_group', 'fare_group', 'Embarked',
        'class_sex', 'age_class', 'family_class'
    ]
    
    X = []  # 特征
    y = []  # 标签
    
    for row in processed_data:
        features = [row[feature] for feature in feature_names]
        X.append(features)
        if 'Survived' in row:
            y.append(row['Survived'])
    
    return X, y

def split_data(X, y, test_size=0.2):
    """将数据分割为训练集和验证集"""
    # 计算测试集大小
    test_count = int(len(X) * test_size)
    
    # 生成随机索引
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # 分割数据
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # 准备训练集和测试集
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """训练模型"""
    feature_weights = {}
    feature_means = {}
    
    # 计算每个特征的平均值
    for i in range(len(X_train[0])):
        values = [x[i] for x in X_train]
        feature_means[i] = sum(values) / len(values)
    
    # 计算每个特征的权重
    for i in range(len(X_train[0])):
        pos_count = neg_count = 0
        for j in range(len(X_train)):
            if X_train[j][i] > feature_means[i]:
                if y_train[j] == 1:
                    pos_count += 1
                else:
                    neg_count += 1
        
        # 计算权重 (使用对数比率)
        if pos_count == 0: pos_count = 1
        if neg_count == 0: neg_count = 1
        weight = statistics.mean([
            pos_count / (pos_count + neg_count),
            1 - (neg_count / (pos_count + neg_count))
        ])
        feature_weights[i] = (weight - 0.5) * 2
    
    return {
        'weights': feature_weights,
        'means': feature_means,
        'threshold': 0.4  # 根据实际生存率调整阈值
    }

def predict(model, X):
    """使用模型进行预测"""
    predictions = []
    weights = model['weights']
    means = model['means']
    threshold = model['threshold']
    
    for sample in X:
        score = 0
        for i, value in enumerate(sample):
            if value > means[i]:
                score += weights[i]
            else:
                score -= weights[i]
        
        # 使用sigmoid函数将分数转换为概率
        probability = 1 / (1 + 2.71828 ** (-score))
        prediction = 1 if probability > threshold else 0
        predictions.append(prediction)
    
    return predictions

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tp': tp, 'fp': fp,
            'fn': fn, 'tn': tn
        }
    }

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    
    # 加载数据
    print("加载数据...")
    train_data = load_data('train.csv')
    test_data = load_data('test.csv')
    
    # 处理缺失值
    print("处理缺失值...")
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    # 特征工程
    print("进行特征工程...")
    processed_train = feature_engineering(train_data)
    processed_test = feature_engineering(test_data)
    
    # 准备训练数据
    print("准备训练数据...")
    X, y = prepare_data(processed_train)
    
    # 分割训练集和验证集
    print("分割训练集和验证集...")
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
    
    # 训练模型
    print("\n训练模型...")
    model = train_model(X_train, y_train)
    
    # 评估模型
    print("\n评估模型...")
    val_predictions = predict(model, X_val)
    metrics = evaluate_model(y_val, val_predictions)
    
    print("\n模型性能:")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"精确率: {metrics['precision']:.2%}")
    print(f"召回率: {metrics['recall']:.2%}")
    print(f"F1分数: {metrics['f1_score']:.2%}")
    
    print("\n混淆矩阵:")
    print(f"真正例: {metrics['confusion_matrix']['tp']}")
    print(f"假正例: {metrics['confusion_matrix']['fp']}")
    print(f"假负例: {metrics['confusion_matrix']['fn']}")
    print(f"真负例: {metrics['confusion_matrix']['tn']}")
    
    # 对测试集进行预测
    print("\n对测试集进行预测...")
    test_features, _ = prepare_data(processed_test)
    test_predictions = predict(model, test_features)
    
    # 生成提交文件
    submission = []
    for i, pred in enumerate(test_predictions):
        submission.append({
            'PassengerId': test_data[i]['PassengerId'],
            'Survived': pred
        })
    
    # 保存预测结果到Result文件夹
    print("\n保存预测结果...")
    predictions_path = os.path.join('Result', 'predictions.csv')
    with open(predictions_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['PassengerId', 'Survived'])
        writer.writeheader()
        writer.writerows(submission)
    
    print(f"预测结果已保存到 {predictions_path}")
    
    # 恢复标准输出
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal