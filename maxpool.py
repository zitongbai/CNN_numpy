import numpy as np

class MaxPool2:
    # 使用2x2的池化大小的最大池化层。

    def iterate_regions(self, image):
        '''
        生成非重叠的2x2图像区域进行池化。
        - image是一个2维的numpy数组
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        对给定的输入进行最大池化层的前向传播。
        返回一个3维的numpy数组，维度为(h / 2, w / 2, num_filters)。
        - input是一个3维的numpy数组，维度为(h, w, num_filters)
        '''
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        '''
        对最大池化层进行反向传播。
        返回该层输入的损失梯度。
        - d_L_d_out是该层输出的损失梯度。
        '''
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # 如果该像素是最大值，则将梯度复制到该像素。
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
