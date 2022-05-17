import os
import random
import shutil

import pandas as pd
from PIL import Image


class dataPreprocess:
    dataRoot = '../正式数据/'  # 正式数据目录
    imageRoot = dataRoot + '附件1/'  # 附件1
    wormLocationTablePath = dataRoot + '附件2/图片虫子位置详情表.csv'  # 位置表路径
    targetRoot = dataRoot + '附件3/无位置信息的图片汇总表.csv'

    outPath = '../datasets/teddy/'  # 输出数据集根目录
    trainImgRoot = 'images/train/'  # 训练集图片目录
    valImgRoot = 'images/val/'  # 验证集图片目录
    trainTxtRoot = 'labels/train/'  # 训练集标签目录
    valTxtRoot = 'labels/val/'  # 验证集标签目录
    targetImtRoot = 'targetImg/'

    wormsDict = {}  # 虫子对应字典

    dataset = []  # 数据集
    dataset_train = []  # 训练数据集
    dataset_val = []  # 验证数据集
    dataset_target = [] #目标数据集

    imgExpandRange = (5, 32)  # 图片拓展范围
    imageSize = (5472, 3648)  # 图片尺寸
    null = 'null'

    # 初始化 提取虫子位置信息表数据并存储
    def __init__(self):
        if not os.path.exists(self.outPath + self.trainImgRoot):  # 创建目录
            os.makedirs(self.outPath + self.trainImgRoot)
        if not os.path.exists(self.outPath + self.valImgRoot):
            os.makedirs(self.outPath + self.valImgRoot)
        if not os.path.exists(self.outPath + self.trainTxtRoot):
            os.makedirs(self.outPath + self.trainTxtRoot)
        if not os.path.exists(self.outPath + self.valTxtRoot):
            os.makedirs(self.outPath + self.valTxtRoot)
        if not os.path.exists(self.outPath + self.targetImtRoot):
            os.makedirs(self.outPath + self.targetImtRoot)
        # if not os.path.exists(self.outPath + self.targetTxtRoot):
        #     os.makedirs(self.outPath + self.targetTxtRoot)

        wTable = pd.read_csv(self.wormLocationTablePath, encoding='gb2312')
        print('读取到虫子位置信息表', wTable)
        wormTypeCnt = 0
        for idx, row in wTable.iterrows():  # 按行读取数据

            if row['虫子编号'] != 0:
                isAdded = False
                for i, v in self.wormsDict.items():
                    if v == row['虫子名称']:  # 已存在虫子
                        isAdded = True

                if not isAdded:
                    self.wormsDict[wormTypeCnt] = row['虫子名称']  # 保存虫子字典
                    wormTypeCnt += 1

                wormId = 0
                for k, v in self.wormsDict.items():
                    if v == row['虫子名称']:
                        wormId = k
                        break

                wormInfo = (wormId, row['中心点x坐标'] / self.imageSize[0], row['中心点y坐标'] / self.imageSize[1],
                            (row['右下角x坐标'] - row['左上角x坐标']) / self.imageSize[0],
                            (row['右下角y坐标'] - row['左上角y坐标']) / self.imageSize[1],
                            row['文件名'])

                added = False
                for i in range(0, len(self.dataset)):
                    if row['文件名'] == self.dataset[i]['filename']:
                        self.dataset[i]['worms'].append(wormInfo)
                        added = True
                if not added:
                    self.dataset.append({
                        'filename': row['文件名'].split('.')[0],
                        'worms': [wormInfo]
                    })

        targetTable = pd.read_csv(self.targetRoot, encoding='gb2312')
        print('无位置信息的图片汇总表', targetTable)
        for idx, row in targetTable.iterrows():
            self.dataset_target.append(row['文件名'])
            # print(self.dataset_target[idx])



    # 切分数据集为训练集和验证集
    def data_split(self, valProportion=0.2):
        print('正在进行数据集划分')
        random.shuffle(self.dataset)  # 打乱数据集
        size = len(self.dataset)
        trainSize = int(size * (1 - valProportion))
        self.dataset_train = self.dataset[:trainSize]
        self.dataset_val = self.dataset[trainSize:]

        print('trainDataCnt = ', len(self.dataset_train), 'testDataCnt = ', len(self.dataset_val))

    # 输出数据集
    def outputData(self):
        self.data2yolo_v1(self.dataset_train, self.trainImgRoot, self.trainTxtRoot, '训练集')
        self.data2yolo_v1(self.dataset_val, self.valImgRoot, self.valTxtRoot, '验证集')
        self.data2yolo_v1(ds=self.dataset_target, imgRoot=self.targetImtRoot, tag='目标集')
    # 提取数据集并按照yolo所需格式写入硬盘
    def data2yolo_v1(self, ds, imgRoot, txtRoot=None, tag='数据集'):
        if txtRoot is not None:
            for data in ds:
                shutil.copyfile(self.imageRoot + data['filename'] + '.jpg',
                                self.outPath + imgRoot + data['filename'] + '.jpg')
                txt = ''
                for v in data['worms']:
                    txt += '{} {} {} {} {}\n'.format(v[0], v[1], v[2], v[3], v[4])
                    # print(v[5])
                    lable_name = v[5].split(".jpg")[0]
                    with open(self.outPath + txtRoot + lable_name + '.txt', 'a') as f:
                        f.write(txt)
                        f.close()
                print('图片:', lable_name, ' 数据:', txt, '已添加入', tag)
        else:
            for data in ds:
                shutil.copyfile(self.imageRoot + data,
                                self.outPath + imgRoot + data)
                print('图片:', data, '已添加入', tag)

    # 区域剪切图片
    # def crop_target(self, sorce_photo_path, aim_path, upLeftX, upLeftY, lowerRighX, lowerRighY):
    def crop_target(self, sorce_photo_path, aim_path):
        img = Image.open(sorce_photo_path)
        # cropd_img = img.crop((upLeftX, upLeftY, lowerRighX, lowerRighY))  # 切割图片
        self.saveImg(filepath=aim_path, imgs=img)

    # 保存图片
    def saveImg(self, filepath, imgs):
        isExists = os.path.exists(filepath)
        if isExists:
            print("该文件已存在")
            x = random.randint(1, 256)
            temp = filepath.split(".jpg")[0]
            new_filepath = temp + '_' + str(x) + ".jpg"
            imgs.save(new_filepath)
        else:
            imgs.save(filepath)

    # 打印虫子字典
    def printWormDict(self):
        names = []
        dignames = []
        for i, v in self.wormsDict.items():
            names.append(v)
            dignames.append(str(i))
            print('虫子ID:', i, ' 对应:', v)
        print('names 汇总列表:', names)
        print('names 数字列表:', dignames)
        print('namesCnt =', len(names))


if __name__ == '__main__':
    dp = dataPreprocess()
    dp.data_split()
    dp.outputData()
    dp.printWormDict()
