import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# 为了节省时间，我们只使用每个集合的前1000个样本。
# 如果需要，可以随意更改这个数值。
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    完成CNN的前向传播，并计算准确率和交叉熵损失。
    - image是一个二维numpy数组
    - label是一个数字
    '''
    # 将图像从[0, 255]转换为[-0.5, 0.5]，以便更容易处理。这是标准做法。
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # 计算交叉熵损失和准确率。np.log()是自然对数。
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    '''
    对给定的图像和标签进行完整的训练步骤。
    返回交叉熵损失和准确率。
    - image是一个二维numpy数组
    - label是一个数字
    - lr是学习率
    '''
    # 前向传播
    out, loss, acc = forward(im, label)

    # 计算初始梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # 反向传播
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('MNIST CNN 初始化完成！')

# 训练CNN，进行3个epoch
for epoch in range(3):
    print('--- 第 %d 个epoch ---' % (epoch + 1))

    # 打乱训练数据
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # 开始训练！
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[步骤 %d] 过去100个步骤：平均损失 %.3f | 准确率：%d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# 测试CNN
print('\n--- 测试CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('测试损失:', loss / num_tests)
print('测试准确率:', num_correct / num_tests)
