# icAN

## 数据集

数据集中大部分的图标的信息都是128*128，部分需要裁减或者差值，然后透明度图：通道为 128*128*4，

## 模型

decoder 的输入为 两个向量
可以调: kernel channels, 可以尝试加dropout(目前没有)，目前输出为-1到1之间的值，可以确认一下是不是所有的四个channel大小都256

train_basic: 最基本的模型

train_resnet：中间加入 res_block

train_icon: 加入 reference pictur

## To do

加上dataloader
模型搭建完成，写一下loss，这个有好几个loss，可以好好看一下，可以适当改进一下？
搞一下embedding的降维可视化
训练，训练，调参，调参！
写report，海报

网络结构：基于原图画一下


颜色： 一开始把颜色选取好

## 环境

```
conda create -n icAN python=3.9
conda activate icAN
pip install -r requirements.txt

```
