# 导入所需的基础库
import csv
import statistics
import random

def load_data(filename):
    """
    从CSV文件加载数据
    返回: 包含所有数据的列表，每个元素是一个字典
    """
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

def handle_missing_values(data):
    """
    处理数据中的缺失值
    """
    # 收集所有非空年龄值
    ages = [float(row['Age']) for row in data if row['Age'] != '']
    
    # 计算年龄的平均值
    mean_age = statistics.mean(ages)
    
    # 获取最常见的登船港口
    embarked = [row['Embarked'] for row in data if row['Embarked'] != '']
    most_common_embarked = max(set(embarked), key=embarked.count)
    
    # 处理缺失值
    for row in data:
        # 处理年龄缺失值
        if row['Age'] == '':
            row['Age'] = str(mean_age)
            
        # 处理登船港口缺失值
        if row['Embarked'] == '':
            row['Embarked'] = most_common_embarked
            
        # 处理船票价格缺失值
        if row['Fare'] == '':
            row['Fare'] = '0'
            
    return data

def feature_engineering(data):
    """特征工程：转换和创建特征"""
    processed_data = []
    
    for row in data:
        # 创建新的特征字典
        features = {}
        
        # 1. 转换性别为数值
        features['is_female'] = 1 if row['Sex'] == 'female' else 0
        
        # 2. 转换船票等级为数值
        features['Pclass'] = int(row['Pclass'])
        
        # 3. 转换年龄为浮点数
        features['Age'] = float(row['Age'])
        
        # 4. 创建年龄段特征 (0: 儿童, 1: 成年, 2: 老年)
        if features['Age'] < 18:
            features['age_group'] = 0
        elif features['Age'] < 60:
            features['age_group'] = 1
        else:
            features['age_group'] = 2
        
        # 5. 转换票价为浮点数
        features['Fare'] = float(row['Fare'])
        
        # 6. 创建家庭规模特征
        features['family_size'] = int(row['SibSp']) + int(row['Parch'])
        
        # 7. 创建是否独自旅行的特征
        features['is_alone'] = 1 if features['family_size'] == 0 else 0
        
        # 8. 转换登船港口为数值
        port_mapping = {'S': 0, 'C': 1, 'Q': 2}
        features['Embarked'] = port_mapping.get(row['Embarked'], 0)
        
        # 如果是训练数据，添加目标变量
        if 'Survived' in row:
            features['Survived'] = int(row['Survived'])
            
        processed_data.append(features)
    
    return processed_data

def prepare_data(processed_data):
    """准备训练数据"""
    # 定义要使用的特征
    feature_names = ['is_female', 'Pclass', 'Age', 'age_group', 
                    'Fare', 'family_size', 'is_alone', 'Embarked']
    
    # 分离特征和标签
    X = []  # 特征
    y = []  # 标签
    
    for row in processed_data:
        # 提取特征
        features = [row[feature] for feature in feature_names]
        X.append(features)
        
        # 提取标签（如果存在）
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
    
    # 准备训练集
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    
    # 准备测试集
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """训练简单的逻辑回归模型"""
    # 计算每个特征的平均值
    feature_means = []
    for i in range(len(X_train[0])):
        values = [x[i] for x in X_train]
        feature_means.append(sum(values) / len(values))
    
    # 计算权重 (使用简单的统计方法)
    weights = []
    for i in range(len(X_train[0])):
        feature_values = [x[i] for x in X_train]
        correlation = 0
        for j in range(len(feature_values)):
            if feature_values[j] > feature_means[i]:
                correlation += y_train[j]
            else:
                correlation -= y_train[j]
        weights.append(correlation / len(y_train))
    
    return {'weights': weights, 'means': feature_means}

def predict(model, X):
    """使用训练好的模型进行预测"""
    predictions = []
    weights = model['weights']
    means = model['means']
    
    for sample in X:
        # 计算加权和
        score = 0
        for i in range(len(sample)):
            if sample[i] > means[i]:
                score += weights[i]
            else:
                score -= weights[i]
        
        # 根据阈值进行预测
        prediction = 1 if score > 0 else 0
        predictions.append(prediction)
    
    return predictions

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    # 计算准确率
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    # 计算其他指标
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
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
    
    # 打印数据集大小
    print("\n数据集大小:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    
    # 打印特征示例
    print("\n特征示例:")
    print("特征名称:", ['is_female', 'Pclass', 'Age', 'age_group', 
                     'Fare', 'family_size', 'is_alone', 'Embarked'])
    print("第一个样本:", X_train[0])
    
    # 训练模型
    print("\n训练模型...")
    model = train_model(X_train, y_train)
    
    # 在验证集上评估模型
    print("\n评估模型...")
    val_predictions = predict(model, X_val)
    metrics = evaluate_model(y_val, val_predictions)
    
    print("\n模型性能:")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"精确率: {metrics['precision']:.2%}")
    print(f"召回率: {metrics['recall']:.2%}")
    
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
    
    # 保存预测结果
    print("\n保存预测结果...")
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['PassengerId', 'Survived'])
        writer.writeheader()
        writer.writerows(submission)
    
    print("预测结果已保存到 predictions.csv")