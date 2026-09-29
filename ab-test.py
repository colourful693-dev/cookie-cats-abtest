#导入依赖库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
df = pd.read_csv("C:/Users/HP/code/data/cookie_cats.csv")

df.head
df.shape
print("\n分组数量统计：")
print(df["version"].value_counts())

#检查数据
df.isnull().sum()
df.duplicated().sum()

#问题1，版本变化是否会提高游戏回合数

import pandas as pd
import scipy.stats as stats

# 定义分析变量
x = ['sum_gamerounds']

# 计算描述性统计，偏度，峰度
descriptive_stats = df[x].describe().T
skewness = df[x].skew().rename('skewness')
kurtosis = df[x].kurtosis().rename('kurtosis')

# 合并数据
concatenated_control = pd.concat([
    descriptive_stats,
    skewness,
    kurtosis
], axis=1)

# 计算置信度区间
confidence_level = 0.95
values = df[x].dropna()
mean = values.mean()
std_error = stats.sem(values)

if std_error != 0:
    lower, upper = stats.t.interval(confidence_level, len(values) - 1, loc=mean, scale=std_error)
else:
    lower, upper = mean, mean

concatenated_control['lower_ci'] = lower
concatenated_control['upper_ci'] = upper

print(concatenated_control)
#绘图
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.boxplot(df['sum_gamerounds'], 
            vert=False,
           patch_artist=True
#            sym='r|'
           )
plt.title('Box Plot of Sum Gamerounds')
plt.ylabel('Sum Gamerounds')
plt.show()

# 清洗异常值
max_value = df['sum_gamerounds'].max()
df = df[df['sum_gamerounds'] != max_value]
df.reset_index(drop=True, inplace=True)
df.head()
#
pivot_table = df.pivot_table(
    values='sum_gamerounds',
    index='version',
    aggfunc='mean'
)
print(pivot_table)

# 创建一个箱线图以对数化的游戏进度
df['log_sum_gamerounds'] = np.log1p(df['sum_gamerounds'])
sns.boxplot(data=df, x='version', y='log_sum_gamerounds',
            hue='version',
            palette=['#199FF4','#01EC63'] )

plt.title('按版本划分的对数转换后游戏回合数的箱线图', fontsize=16)
plt.xlabel('版本', fontsize=12)
plt.ylabel('对数转换后游戏回合数', fontsize=12)

# Display the plot
plt.show()

#ab试验，分离对照和测试数据
df_test= df[df['version']=='gate_40']
df_control= df[df['version']=='gate_30']

import pandas as pd
from scipy.stats import norm

alpha = 0.05    # I类错误，显著性水平
beta = 0.2      # II类错误；统计功效 power = 1-beta = 0.8
# 计算Z：Zα/2 + Zβ
Z = (norm.ppf(1-(alpha/2), loc=0, scale=1) + norm.ppf(1-beta, loc=0, scale=1))**2
# 合并两组数据，计算总体方差
var = pd.concat([df_control['sum_gamerounds'], df_test['sum_gamerounds']]).var()
expected_effect_size = 5   # 预期效应量：两组均值差值=5
# 样本量公式
sample_size = (Z*2*var)/(expected_effect_size**2)
print(sample_size.astype('int'))

#选择随机样本
def choose_random_sample(data):
    return data.sample(n=int(sample_size), random_state=42)
# 直接抽样
df_control_smpl = choose_random_sample(df_control)
df_test_smpl = choose_random_sample(df_test)

print(df_control_smpl)
print(df_test_smpl)

#可视化分类样本后两组数据
df_control_smpl['log_sum_gamerounds'] = np.log1p(df_control_smpl['sum_gamerounds'])
df_test_smpl['log_sum_gamerounds'] = np.log1p(df_test_smpl['sum_gamerounds'])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
ax.boxplot(
    x=[df_control_smpl['log_sum_gamerounds'], df_test_smpl['log_sum_gamerounds']],
    vert=False,
    patch_artist=True,
    widths=0.5,
    flierprops={
        "marker": "|",
        "markeredgecolor": "red",
        "markeredgewidth": 1.5, # 线条粗细，调大更容易看见红色
        "markerfacecolor": "red"
    },
    showmeans=True,
    showfliers=True,
    boxprops=dict(facecolor='#199FF4'),
    medianprops=dict(color='#01EC63', linewidth=2),
    meanprops=dict(marker='x', markerfacecolor='red', markeredgecolor='#01EC63', markersize=8)
)
ax.set_yticklabels(['log_sum_gamerounds_control', 'log_sum_gamerounds_test'])
ax.set_xlabel('log(sum_gamerounds)')
plt.title('Log-Transformed Boxplot of sum_gamerounds')
plt.show()

#两样本t检验
from scipy.stats import levene, ttest_ind, t
#分组计算均值
mean_control = df_control_smpl['sum_gamerounds'].mean()
mean_test = df_test_smpl['sum_gamerounds'].mean()

print("mean_control =", mean_control)
print("mean_test =", mean_test)

groupC = df_control['sum_gamerounds']
groupT = df_test['sum_gamerounds']

statistic, p_value_L = levene(groupC, groupT)
#方差齐性检验
if p_value_L > 0.05:
    # If variances are equal
    t_statistic, p_value = ttest_ind(groupC, groupT, alternative="less", equal_var=True)
else:
    # If variances are not equal
    t_statistic, p_value = ttest_ind(groupC, groupT, alternative="less", equal_var=False)

print("t-statistic:", t_statistic)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("拒绝原假设：两个均值之间存在显著差异。")
else:
    print("无法拒绝原假设：两个均值之间不存在显著差异。")
    
#计算均值差
mean_difference =  mean_test-mean_control 
print("Mean difference =", mean_difference)

std_control = df_control_smpl['sum_gamerounds'].std()
std_test = df_test_smpl['sum_gamerounds'].std()

#计算置信区间
n_control = len(groupC)
n_test = len(groupT)

SE = np.sqrt((std_control**2 / n_control) + (std_test**2 / n_test))

ddf = n_control + n_test - 2

t_critical = t.ppf(1 - 0.025, ddf)

margin_of_error = t_critical * SE
confidence_interval = (mean_difference - margin_of_error, mean_difference + margin_of_error)
print("置信区间:", confidence_interval)