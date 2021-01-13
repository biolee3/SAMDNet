import os
import numpy as np
from os.path import *
#seq_home = 'E:\\Temple-color-128\\'
#seq_home = 'E:\\UAV123_10fps_O\\data_seq\\UAV123_10fps\\'
#seq_home = 'E:/OTB2015/'
#seq_home = 'C:\\Users\\biolee\\Desktop\\UPan\\vot2015\\'
seq_home = 'E:\\vot2016\\'
seq_name = os.listdir(seq_home)
# 获取当前目录绝对路径
#dir_path = dirname(abspath(__file__))
#print('当前目录绝对路径:', dir_path)

# 获取上级目录绝对路径
dir_path = dirname(dirname(abspath(__file__)))
#print('上级目录绝对路径:', dir_path)

for i in range(len(seq_name)):
    seq = seq_name[i]
    file = dir_path + '/results/' + seq + '.txt'
    if (os.path.exists(file)):
        continue
    else:
        print(seq)
        fig_path = dir_path + '/results_figs/' + seq
        if(os.path.exists(fig_path)) == False:
            os.makedirs(fig_path)
        os.system('python run_tracker.py -s ' + seq_home + seq + ' -f ' + fig_path)

if(os.path.exists(dir_path + '/results/A_seqs_results.txt')):
    os.remove(dir_path + '/results/A_seqs_results.txt')
file = open(dir_path + '/results/A_seqs_results.txt','a')
for j in range(len(seq_name)):
    seq = seq_name[j]
    f1 = open(dir_path + '/results/' + seq + '_overlap.txt')
    data = f1.readline()
    file.write(seq + ': '+ data + '\n')
    f1.close()
file.close()
#统计结果

#统计两个文件之间的结果差别
'''
A_dir = 'C:\\Users\\biolee\Desktop\py-MDNet-master'
#B 是自己追踪算法的路径

B_dir = dir_path
if(os.path.exists(B_dir + '\\results\A_diffs_results.txt')):
    os.remove(B_dir + '\\results\A_diffs_results.txt')
diff_file  = open(B_dir + '\\results\A_diffs_results.txt','a')
diff_file.write('(Overlap(my) - Overlap(literature)) * 100\n')

Num_Up = 0
Num_Down = 0
for k in range(len(seq_name)):
    seq = seq_name[k]
    f1 = open(A_dir + '\\results\\' + seq + '_overlap.txt')
    f2 = open(B_dir + '\\results\\' + seq + '_overlap.txt')
    data_A = float(f1.read())
    data_B = float(f2.read())
    diffs = (data_B - data_A) * 100
    if diffs > 0:
        Num_Up = Num_Up + 1
    else:
        Num_Down = Num_Down + 1
    diffs = str(diffs)
    diff_file.write(seq + ': ' + diffs + '\n')
    f1.close()
    f2.close()
diff_file.write('Pos : '+ str(Num_Up) + '\n' + 'Neg : ' + str(Num_Down) + '\n')
diff_file.close()
'''
