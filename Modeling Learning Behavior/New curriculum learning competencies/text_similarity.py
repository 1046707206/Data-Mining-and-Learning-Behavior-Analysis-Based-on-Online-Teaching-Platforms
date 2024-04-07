# -*- coding: utf-8 -*-
import jieba
import math
import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def similatity(s1,s2):
    # 利用jieba分词与停用词表，将词分好并保存到向量中
    stopwords = []
    fstop = open('stop_words.txt', 'r', encoding='utf-8-sig')
    for eachWord in fstop:
        eachWord = re.sub("\n", "", eachWord)
        stopwords.append(eachWord)
    fstop.close()
    s1_cut = [i for i in jieba.cut(s1, cut_all=True) if (i not in stopwords) and i != '']
    s2_cut = [i for i in jieba.cut(s2, cut_all=True) if (i not in stopwords) and i != '']
    word_set = set(s1_cut).union(set(s2_cut))

    # 用字典保存两篇文章中出现的所有词并编上号
    word_dict = dict()
    i = 0
    for word in word_set:
        word_dict[word] = i
        i += 1

    # 根据词袋模型统计词在每篇文档中出现的次数，形成向量
    s1_cut_code = [0] * len(word_dict)

    for word in s1_cut:
        s1_cut_code[word_dict[word]] += 1

    s2_cut_code = [0] * len(word_dict)
    for word in s2_cut:
        s2_cut_code[word_dict[word]] += 1

    # 计算余弦相似度
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(s1_cut_code)):
        sum += s1_cut_code[i] * s2_cut_code[i]
        sq1 += pow(s1_cut_code[i], 2)
        sq2 += pow(s2_cut_code[i], 2)

    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 3)
    except ZeroDivisionError:
        result = 0.0
    return result

df = pd.read_excel('information_of_course.xlsx')
df1 = pd.DataFrame(columns=['course1', 'course2', 'similarity'])  # 存储结果
lis = [[0] * df.shape[0] for _ in range(df.shape[0])]

for i in range(df.shape[0]):
    s1 = str(df.loc[i, '专业、相关课程'])
    for j in range(df.shape[0]):
        s2 = str(df.loc[j, '专业、相关课程'])
        lis[i][j] = similatity(s1, s2)
        df1.loc[i+j+i*df.shape[0]-1, 'course1'] = df.loc[i, 'courseid']
        df1.loc[i+j+i*df.shape[0]-1, 'course2'] = df.loc[j, 'courseid']
        df1.loc[i + j + i * df.shape[0] - 1, 'similarity'] = similatity(s1, s2)

# 可视化 作为示范，只对每门课程的第一位用户进行了可视化
x_ticks = [i for i in range(df.shape[0])]
y_ticks = [i for i in range(df.shape[0])]  # 自定义横纵轴
ax = sns.heatmap(lis, xticklabels=x_ticks, yticklabels=y_ticks, annot=True)
ax.set_title('course_similarity')  # 图标题
ax.set_xlabel('x course')  # x轴标题
ax.set_ylabel('y course')
plt.show()

# 保存到文件
df1.to_csv('course_similarity.csv', index=False)



