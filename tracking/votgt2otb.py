import os
import numpy as np

filepath  = 'C:\\Users\\biolee\\Desktop\\UPan\\vot2015\\bag\\groundtruth.txt'
'''
with open(filepath) as f:
    gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')

x = np.zeros(gt.shape[0] * 4).reshape(gt.shape[0],4)
y = np.zeros(gt.shape[0] * 4).reshape(gt.shape[0],4)

x[:, 0] = gt[:, 0]
x[:, 1] = gt[:, 2]
x[:, 2] = gt[:, 4]
x[:, 3] = gt[:, 6]
x_sort = np.sort(x, axis=1)
width = np.floor(x_sort[:, 3])-np.floor(x_sort[:, 0])

y[:, 0] = gt[:, 1]
y[:, 1] = gt[:, 3]
y[:, 2] = gt[:, 5]
y[:, 3] = gt[:, 7]
y_sort = np.sort(y, axis=1)
height = np.floor(y_sort[:, 3])-np.floor(y_sort[:, 0])
#modify coordinate to x,y,w,h
x[:, 0] = x_sort[:, 0]
x[:, 1] = y_sort[:, 0]
x[:, 2] = width
x[:, 3] = height
print('ok')
'''
dir_path = 'E:\\SAMDNet\\results\\'
origion_dir  = 'E:\\vot2016\\'

dir_list = os.listdir(dir_path)
origion_list = os.listdir(origion_dir)
for i in origion_list:
    if i == 'list.txt':
        origion_list.remove(i)
for i in origion_list:
    if not os.path.exists(dir_path+i):
        os.makedirs(dir_path+i)
    if os.path.exists(dir_path + i + '\\' + i + '_001.txt'):
        continue
    else:
        f1 = open(dir_path + i + '.txt', 'r')
        f2 = open(dir_path + i + '\\' + i + '_001.txt', 'a')
        for x in f1:
            if x == '1,1,1,1,\n':
                x = '1\n'
                f2.write(x)
            elif x == '2,2,2,2,\n':
                x = '2\n'
                f2.write(x)
            elif x == '0,0,0,0,\n':
                x = '0\n'
                f2.write(x)
            elif x.find(',\n') == -1:
                x = x + '\n'
                x = x.replace(',\n', '\n')
                f2.write(x)
            else:
                x = x.replace(',\n', '\n')
                f2.write(x)
        f1.close()
        f2.close()
        f1 = open(dir_path + i +'_time.txt', 'r')
        f2 = open(dir_path + i + '\\' + i  + '_time.txt', 'w')
        for i in f1:
            i = i.replace('\n', '')
            i = i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i + '_' + i
            i = i.replace('_', ',').strip() + '\n'
            f2.write(i)
        f1.close()
        f2.close()
print('hello')
