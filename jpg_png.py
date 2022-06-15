# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:39:03 2019

@author: wsb
"""

import cv2
import os

print('----------------------------------------------------')
print('程序的功能为：将该目录下输入的文件内的图片转为指定格式')  # 目前我测试了jpg转化为png和png转化为jpg。
print('转化结果保存在当前目录下的new_picture内')
print('----------------------------------------------------')


picture_type = "jpg"
path = "./data/market1501-mask/bounding_box_test_seg"

newpath = "./new_market1501-mask/bounding_box_test_seg"
if not os.path.exists(newpath):
    os.mkdir(newpath)

path_list = os.listdir(path)
number = 0  # 统计图片数量
for filename in path_list:
    number += 1
    portion = os.path.splitext(filename)
    print('convert  ' + filename + '  to ' + portion[0] + '.' + picture_type)
    img = cv2.imread(path + "/" + filename)
    cv2.imwrite("./" + newpath + "/" + portion[0] + '.' + picture_type, img)
print("共转化了%d张图片" % number)
print('转换完毕，文件存入 ' + newpath + ' 中')
cv2.waitKey(0)
cv2.destroyAllWindows()
