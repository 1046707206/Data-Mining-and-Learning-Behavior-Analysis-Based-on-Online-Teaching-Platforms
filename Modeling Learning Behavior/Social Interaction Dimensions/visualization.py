import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
df = pd.read_csv('data/social_analysis_of_all_class.csv')
plt.figure(figsize=(20, 8))

x = np.arange(len(df['personid']))
width=0.15
# x1 = x-width/2
# x2 = x+width/2
y1 = df['total_postings']
y2 = df['num_of_postings']
y3 = df['num_of_replies']
y4 = df['reply_to_teacher']
y5 = df['reply_to_student']

# 绘制分组柱状图

plt.bar(x-2*width,y1,width,label='总发贴和回帖数',color='#f9766e')
plt.bar(x-width,y2,width,label='发帖数',color='#7093DB')
plt.bar(x,y3,width,label='回帖数',color='#215E21')
plt.bar(x+width,y4,width,label='回老师帖数',color='#8E236B')
plt.bar(x+2*width,y5,width,label='回学生帖数',color='#00bfc4')


# 添加x,y轴名称、图例和网格线
plt.xlabel('用户',fontsize=11)
plt.ylabel('次数',fontsize=11)

# plt.grid(ls='--',alpha=0.8)

# 修改x刻度标签为对应日期
plt.xticks(x,labels=range(len(df['personid'])))
plt.legend()
# plt.tick_params(axis='x',length=0)

# plt.tight_layout()
# plt.savefig('bar2.png',dpi=600)
plt.show()