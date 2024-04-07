import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


# 梯度递减函数
def gradient_descent(x, y, alpha=0.001, ep=0.00000001, max_iter=100000):
    converged = False
    time = 0
    m = len(x)

    # 初始化参数
    t0 = x[0]
    t1 = 1

    # 构造代价函数, J(theta)
    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    # 进行迭代
    while not converged:
        # 计算训练集中每一行数据的梯度
        grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        # 更新方程参数
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
        t0 = temp0
        t1 = temp1

        # 均方误差 (MSE)
        e = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

        # 判断是否下降到最低点（即此时梯度接近0，t0和t1的值基本不变）
        if abs(J - e) <= ep:
            # print('完成优化，迭代次数:', time)
            # print('均方误差为:', e)
            converged = True
        J = e  # 更新误差值
        time += 1  # 更新迭代次数

        if time == max_iter:
            # print('达到最大迭代次数，结束迭代')
            # print('此时的均方误差为:', e)
            converged = True

    return t0, t1


def house_predict(S, P):
    x_max = max(S)
    x_min = min(S)
    y_max = max(P)
    y_min = min(P)
    x = []
    y = []
    # 数据归一化
    for i in S:
        x.append((i - x_min) / (x_max - x_min))
    for i in P:
        y.append((i - y_min) / (y_max - y_min))
    # data = [S, P]
    # 数据归一化
    # Max-Min标准化
    # 建立MinMaxScaler对象
    # minmax = preprocessing.MinMaxScaler()
    # # 标准化处理
    # data_minmax = minmax.fit_transform(data)
    # x = data_minmax[0]
    # y = data_minmax[1]

    # 得到线性方程参数进行预测，得到线性方程参数
    t0, t1 = gradient_descent(x, y)

    # 绘图
    # 1.数据散点图
    S = np.array(S)
    # print(S)
    P = np.array(P)

    # col = np.where(S == 137, 'red', 'black')  # 颜色设置
    plt.figure()
    plt.scatter(S, P, c='purple')
    # 2.线性方程直线图
    x_line = S
    # 生成y轴对应的坐标
    # y_line = [(0 - (t0 + t1 * i)) for i in x_line]
    y_line = [(t0 + t1 *( i-x_min)/(x_max-x_min))*((y_max-y_min)+y_min) for i in x_line]
    plt.plot(x_line, y_line)
    # plt.show()
    # print(t0, t1)
    return y_line

files = os.listdir('data/work_final/')
for file in files:
    file_path = os.path.join('data', 'work_final', file)
    df = pd.read_csv(file_path)
    x = list(df.loc[:, 'score'])
    y = list(df.loc[:, 'final_score'])
    y_line = house_predict(x, y)
    y_lst = []
    level = []
    for i in y_line:
        y_lst.append(round(i, 2))
        if i >= 90:
            level.append('强')
        elif 90 > i >= 80:
            level.append('较强')
        elif 80 > i >= 60:
            level.append('一般')
        else:
            level.append('弱')
    # df.insert(loc=4, column="predict_score", value=y_line)
    df['predict_score'] = y_lst
    df['level'] = level
    df.to_csv('data/linear_analysis_result/predict_score_{}.csv'.format((df.loc[0, 'courseid'])), index = False)
    plt.savefig('data/linear_analysis_result/{}.jpg'.format(df.loc[0, 'courseid']))  # 保存图片
    data = []
    q = list(df.index)
    # print(q)
    for sy in q:
        d = {}
        d["学生编号"] = sy
        # print(sy)
        d["预测成绩"] = df.loc[sy, 'predict_score']
        data.append(d)
    f = open('data/linear_analysis_result/data_{}.txt'.format(df.loc[0, 'courseid']), 'w')
    f.write(str(data))
    f.close()
