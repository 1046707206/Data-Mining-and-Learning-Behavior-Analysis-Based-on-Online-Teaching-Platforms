import pandas as pd

# 数据预处理   t_stat_work_answer
# 导入数据
df = pd.read_excel('data/t_stat_work_answer.xls')

# 只要id', 'personid', 'courseid', 'score'这4列
df1 = df.loc[:, ['id', 'personid', 'courseid', 'score']]

counted_person = []  # 记录已经计算了的用户
i = 0
j = 0
df_lis = {}

# for i in range(1133):
for i in range(df1.shape[0] - 1):
    count = 1  # 记录特定的person个数
    personid = df1.loc[i, 'personid']
    if personid in counted_person:
        continue  # 已经计算过该用户了，继续遍历
    else:
        if i != 0:
            if df1.loc[i, 'courseid'] != df.loc[i - 1, 'courseid']:  # 该条记录与上一条记录的课程是否相等。若不等，到下一个课程了
                del df_lis[j]['id']
                df_lis[j].to_csv('data/work_answer/aver_work_{}.csv'.format(df1.loc[i-1, 'courseid']), index=False)
                j = j + 1
                counted_person = []
                i = i - 1
                continue
        if j in df_lis:
            df_lis[j] = pd.concat([df_lis[j], df1.loc[[i]]], ignore_index=True)  # 同一课程不同用户，新增一行
        else:
            df_lis[j] = df1.loc[[i]]
        counted_person.append(personid)
        for k in range(i + 1, df1.shape[0]):
            if df1.loc[i, 'courseid'] == df1.loc[k, 'courseid'] and k != df1.shape[0] - 1:  # 是同一门课程放在同一个文件里
                # df_lis[j] = df1.loc[[i]]
                if int(df1.loc[k, 'personid']) != personid:
                    continue
                else:
                    # last_count = []
                    count = count + 1
                    df_lis[j].iloc[-1, -1] = df_lis[j].iloc[-1, -1] + df1.loc[k, 'score']
                    # last_count = count
            elif df1.loc[i, 'courseid'] != df1.loc[k, 'courseid'] or df1.loc[i, 'courseid'] == df1.loc[df1.shape[0]-1, 'courseid']:  # 因为数据是按课程排列的，所以课程不同了 就可以计算平均分了
                df_lis[j].iloc[-1, -1] = round(df_lis[j].iloc[-1, -1] / count, 2)
                break
# for num in range(df_lis[j].shape[0]):
#     df_lis[j].iloc[num, -1] = round(df_lis[j].iloc[num, -1] / last_count, 2)
del df_lis[j]['id']
df_lis[j].to_csv('data/work_answer/aver_work_{}.csv'.format(df1.loc[i, 'courseid']), index=False)

