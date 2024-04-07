# 计算学生新课程学习能力  课程相似度*作业平均分
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os


# 课程能力预测计算
def predict_ability(similarity, score):
    return similarity * score


# 课程能力预测
def a(df, df1, df2):
    # 创建二维数组，用于可视化
    lis = [[] for _ in range(df.shape[0])]
    for i in range(df.shape[0]):
        for j in range(df1.shape[0]):
            if df.loc[i, 'courseid'] == df1.loc[j, 'course1']:
                df2.loc[i, 'personid'] = df.loc[i, 'personid']
                df2.loc[i, 'new_courseid'] = df1.loc[i, 'course2']
                df2.loc[i, 'predicted_ability'] = predict_ability(df1.loc[j, 'similarity'], df.loc[i, 'score'])
                lis[i].append(df2.loc[i, 'predicted_ability'])
    # 保存文件
    df2.to_csv('predict_ability_{}.csv'.format(df.loc[0, 'courseid']), index=False)
    # 可视化
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.figure()
    x = list(range(len(lis[0])))
    y = lis[0]
    plt.scatter(x, y, c='green', s=10, label='预测的分数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()  # 显示图片中的标签
    plt.savefig('result_img/{0}_{1}.jpg'.format(0, df.loc[0, 'courseid']))  # 保存图片



files = os.listdir('../data/work_answer/')
# 读取课程相似度文件
df1 = pd.read_csv('course_similarity.csv')
for file in files:
    file_path = os.path.join('../data', 'work_answer', file)
    # 读取作业平均分文件
    df = pd.read_csv(file_path)
    # 创建存储结果的df
    df2 = pd.DataFrame(columns=['personid', 'new_courseid', 'predicted_ability'])
    a(df, df1, df2)

df = pd.read_csv('predict_ability_222652756.csv')
level = []
for i in range(df.shape[0]):
    p_score = round(df.loc[i, 'predicted_ability'], 2)
    if p_score >= 60:
        le = '基础较好'
    elif 40 <= p_score < 60:
        le = '基础一般'
    else:
        le = '基础薄弱'
    level.append(le)
df['level'] = level
df.to_csv('predict_ability_222652756_level.csv', index=False)