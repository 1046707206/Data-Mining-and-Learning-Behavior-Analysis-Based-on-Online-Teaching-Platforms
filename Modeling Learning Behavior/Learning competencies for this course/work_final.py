# 利用课程的作业平均分来预测课程最终成绩
# 先把平均分与课程最终成绩合并，在做线性规划
# 线性规划：绘制散点图、做相关性分析
import os
import pandas as pd
import matplotlib.pyplot as plt

# 先把平均分与课程最终成绩合并
files = os.listdir('data/work_answer/')

df_score = pd.read_excel('data/t_stat_student_score.xls')
df_score = df_score.loc[:, [ 'personid', 'courseid', 'score']]
df_score = df_score.rename(columns={'score':'final_score'})
for file in files:
    file_path = os.path.join('data', 'work_answer', file)
    df = pd.read_csv(file_path)
    df_final = pd.merge(df, df_score, on=['courseid', 'personid'], how='left')
    df_final.to_csv('data/work_final/{}.csv'.format(df.loc[0, 'courseid']), index=True)


