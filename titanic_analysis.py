import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 查看训练集基本信息
print("训练集信息:")
print("数据集大小:", train_data.shape)
print("\n数据类型信息:")
print(train_data.info())

# 查看缺失值情况
print("\n缺失值统计:")
print(train_data.isnull().sum())

# 查看数值型特征的统计描述
print("\n数值型特征统计描述:")
print(train_data.describe())

# 查看类别型特征的分布
print("\n类别型特征分布:")
for column in ['Sex', 'Embarked', 'Pclass']:
    print(f"\n{column}特征的分布:")
    print(train_data[column].value_counts())

# 分析性别对生存的影响
print("\n性别与生存率分析:")

# 计算不同性别的生存率
gender_survival = train_data.groupby('Sex')['Survived'].mean()
print("\n各性别生存率:")
print(gender_survival)

# 统计不同性别的生存人数
gender_survival_count = pd.crosstab(train_data['Sex'], train_data['Survived'])
print("\n性别生存统计:")
print(gender_survival_count)

# 可视化性别生存率
plt.figure(figsize=(8, 6))
gender_survival.plot(kind='bar')
plt.title('不同性别的生存率')
plt.xlabel('性别')
plt.ylabel('生存率')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 可视化性别生存人数分布
plt.figure(figsize=(8, 6))
gender_survival_count.plot(kind='bar', stacked=True)
plt.title('不同性别的生存人数分布')
plt.xlabel('性别')
plt.ylabel('人数')
plt.legend(['未生存', '生存'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 分析年龄对生存的影响
print("\n年龄与生存率分析:")

# 处理年龄缺失值
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

# 创建年龄段
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels)

# 计算不同年龄段的生存率
age_survival = train_data.groupby('AgeGroup', observed=True)['Survived'].mean()
print("\n各年龄段生存率:")
print(age_survival)

# 统计不同年龄段的生存人数
age_survival_count = pd.crosstab(train_data['AgeGroup'], train_data['Survived'])
print("\n年龄段生存统计:")
print(age_survival_count)

# 可视化年龄段生存率
plt.figure(figsize=(12, 6))
age_survival.plot(kind='bar')
plt.title('不同年龄段的生存率')
plt.xlabel('年龄段')
plt.ylabel('生存率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 可视化年龄分布
plt.figure(figsize=(12, 6))
plt.hist(train_data[train_data['Survived'] == 1]['Age'], alpha=0.5, label='生存', bins=30)
plt.hist(train_data[train_data['Survived'] == 0]['Age'], alpha=0.5, label='未生存', bins=30)
plt.title('生存和未生存乘客的年龄分布')
plt.xlabel('年龄')
plt.ylabel('人数')
plt.legend()
plt.tight_layout()
plt.show()

# 箱线图显示年龄与生存关系
plt.figure(figsize=(8, 6))
train_data.boxplot(column='Age', by='Survived')
plt.title('生存状态与年龄的关系')
plt.suptitle('')  # 这行用于删除自动生成的标题
plt.xlabel('是否生存')
plt.ylabel('年龄')
plt.show()

# 分析船票等级对生存的影响
print("\n船票等级与生存率分析:")

# 计算不同等级的生存率
pclass_survival = train_data.groupby('Pclass')['Survived'].mean()
print("\n各等级生存率:")
print(pclass_survival)

# 统计不同等级的生存人数
pclass_survival_count = pd.crosstab(train_data['Pclass'], train_data['Survived'])
print("\n船票等级生存统计:")
print(pclass_survival_count)

# 可视化船票等级生存率
plt.figure(figsize=(8, 6))
pclass_survival.plot(kind='bar')
plt.title('不同船票等级的生存率')
plt.xlabel('船票等级')
plt.ylabel('生存率')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 可视化船票等级生存人数分布
plt.figure(figsize=(8, 6))
pclass_survival_count.plot(kind='bar', stacked=True)
plt.title('不同船票等级的生存人数分布')
plt.xlabel('船票等级')
plt.ylabel('人数')
plt.legend(['未生存', '生存'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 分析船票等级和性别的组合效应
print("\n船票等级和性别的组合分析:")

# 计算船票等级和性别组合的生存率
pclass_sex_survival = train_data.groupby(['Pclass', 'Sex'])['Survived'].mean()
print("\n各组合的生存率:")
print(pclass_sex_survival)

# 可视化船票等级和性别的组合生存率
plt.figure(figsize=(10, 6))
pclass_sex_survival.unstack().plot(kind='bar', width=0.8)
plt.title('不同船票等级和性别的生存率')
plt.xlabel('船票等级')
plt.ylabel('生存率')
plt.legend(title='性别')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 总结各个因素对生存率的影响
print("\n=== 泰坦尼克号乘客生存影响因素分析总结 ===")

# 1. 性别因素总结
print("\n1. 性别对生存率的影响:")
print(f"女性生存率: {gender_survival['female']:.2%}")
print(f"男性生存率: {gender_survival['male']:.2%}")
print("结论: 性别是影响生存率的重要因素，女性的生存率显著高于男性")

# 2. 年龄因素总结
print("\n2. 年龄对生存率的影响:")
print("各年龄段生存率:")
for age_group, survival_rate in age_survival.items():
    print(f"{age_group}: {survival_rate:.2%}")
# 找出生存率最高和最低的年龄段
max_survival_age = age_survival.idxmax()
min_survival_age = age_survival.idxmin()
print(f"生存率最高的年龄段: {max_survival_age} ({age_survival[max_survival_age]:.2%})")
print(f"生存率最低的年龄段: {min_survival_age} ({age_survival[min_survival_age]:.2%})")
print("结论: 年龄对生存率有一定影响，但不如性别的影响显著")

# 3. 船票等级因素总结
print("\n3. 船票等级对生存率的影响:")
print("各等级生存率:")
for pclass, survival_rate in pclass_survival.items():
    print(f"{pclass}等舱: {survival_rate:.2%}")
print("结论: 船票等级与生存率呈现明显的正相关，等级越高生存率越高")

# 4. 多因素组合分析总结
print("\n4. 多因素组合分析:")
print("不同船票等级和性别的组合生存率:")
for (pclass, sex), rate in pclass_sex_survival.items():
    print(f"{pclass}等舱{sex}: {rate:.2%}")

# 找出生存率最高和最低的组合
max_combo = pclass_sex_survival.idxmax()
min_combo = pclass_sex_survival.idxmin()
print(f"\n生存率最高的组合: {max_combo[0]}等舱{max_combo[1]} ({pclass_sex_survival[max_combo]:.2%})")
print(f"生存率最低的组合: {min_combo[0]}等舱{min_combo[1]} ({pclass_sex_survival[min_combo]:.2%})")

print("\n总体结论:")
print("1. 性别是最主要的影响因素，女性的生存机会远高于男性")
print("2. 船票等级是第二重要的因素，高等级船票的乘客生存率更高")
print("3. 年龄对生存率有一定影响，但影响程度不如性别和船票等级")
print("4. 多个因素组合会产生叠加效果，如高等级船票的女性乘客生存率最高")

print("\n===============================================")
print("【泰坦尼克号生存分析结果解释】")
print("===============================================")

print("\n【性别对生存的影响】")
print("1. 生存率差异:")
print(f"  - 女性生存率: {gender_survival['female']:.2%}")
print(f"  - 男性生存率: {gender_survival['male']:.2%}")
print("\n2. 具体人数:")
print("  - 女性: 生存", gender_survival_count.loc['female', 1], "人, 死亡", gender_survival_count.loc['female', 0], "人")
print("  - 男性: 生存", gender_survival_count.loc['male', 1], "人, 死亡", gender_survival_count.loc['male', 0], "人")
print("\n3. 结论: 性别是最关键的生存因素，女性的生存机会显著高于男性，体现了'女士优先'的救生原则")

print("\n【年龄对生存的影响】")
print("1. 年龄段分析:")
for age_group, rate in age_survival.items():
    print(f"  - {age_group}岁: {rate:.2%}")
print("\n2. 关键发现:")
print(f"  - 生存率最高的年龄段: {max_survival_age} ({age_survival[max_survival_age]:.2%})")
print(f"  - 生存率最低的年龄段: {min_survival_age} ({age_survival[min_survival_age]:.2%})")
print("3. 结论: 年龄对生存率有影响，但不如性别显著。儿童的生存率相对较高")

print("\n【船票等级对生存的影响】")
print("1. 各等级生存率:")
for pclass, rate in pclass_survival.items():
    print(f"  - {pclass}等舱: {rate:.2%}")
print("\n2. 组合分析:")
print("  不同等级和性别的组合生存率:")
for (pclass, sex), rate in pclass_sex_survival.items():
    print(f"  - {pclass}等舱{sex}: {rate:.2%}")
print(f"\n3. 最佳/最差生存组合:")
print(f"  - 最佳: {max_combo[0]}等舱{max_combo[1]} ({pclass_sex_survival[max_combo]:.2%})")
print(f"  - 最差: {min_combo[0]}等舱{min_combo[1]} ({pclass_sex_survival[min_combo]:.2%})")
print("\n4. 结论: 船票等级与生存率呈正相关，且与性别因素存在明显的交互作用")

print("\n【总体结论】")
print("1. 影响因素重要性排序:")
print("   - 第一位：性别（女性生存优势明显）")
print("   - 第二位：船票等级（高等级优势明显）")
print("   - 第三位：年龄（影响相对较小）")
print("\n2. 生存机会最大的组合: 高等级船舱的女性乘客")
print("3. 生存机会最小的组合: 低等级船舱的男性乘客") 