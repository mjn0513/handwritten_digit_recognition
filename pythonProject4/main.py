# test1227
from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
from pip._internal import network

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()   # 下载train_image,train_labels,test_image,test_labels
# test1：输出第零张图片的每位像素值以及绘制该图片
# print(train_images.shape, test_images.shape)
# print(train_images[0])
# print(train_labels[0])
# plt.imshow(train_images[0])     # 使用python自带画图软件绘制train_images
# plt.show()

# 压缩向量
train_images = train_images.reshape((60000, 28*28)).astype('float')   # 将二维矩阵压缩为一维向量，将数据剋行转换为float
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)           # 将train_labels重新编码，数组中第n位为1则数组表示为n
test_labels = to_categorical(test_labels)
# test2：输出第0张图片的所示数字，由上述已知第0张图片所示数字为5，则输出应该为[0 0 0 0 0 1 0 0 0 0]
# print(train_labels[0])

# 搭建神经网络
network = models.Sequential()
network.add(layers.Dense(units=15, activation='relu', input_shape=(28*28, ),))
network.add(layers.Dense(units=10, activation='softmax'))
# test3：搭建的神经网络的信息
# print(network.summary())

# 编译步骤
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)

# 使用测试集进行测试
y_pre = network.predict(test_images[:5])
print(y_pre, test_labels[:5])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
