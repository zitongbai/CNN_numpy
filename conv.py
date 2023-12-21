import numpy as np

class Conv3x3:
    # 使用3x3滤波器的卷积层。

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters是一个3维数组，维度为(num_filters, 3, 3)
        # 我们除以9来减小初始值的方差
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        生成所有可能的3x3图像区域，使用valid padding。
        - image是一个2维numpy数组。
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        使用给定的输入执行卷积层的前向传播。
        返回一个3维numpy数组，维度为(h, w, num_filters)。
        - input是一个2维numpy数组
        '''
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行卷积层的反向传播。
        - d_L_d_out是该层输出的损失梯度。
        - learn_rate是一个浮点数。
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新filters
        self.filters -= learn_rate * d_L_d_filters

        # 我们这里不返回任何东西，因为我们将Conv3x3作为CNN中的第一层。
        # 否则，我们需要返回该层输入的损失梯度，就像CNN中的每一层一样。
        return None
