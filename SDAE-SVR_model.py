#######################################
######## SDAE-SVR Model  ##############
#######################################

from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import xlrd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xlwt
import math
import datetime
from xlrd import xldate_as_tuple
import matplotlib.pyplot as plt

'''参数初始化 用xavier初始化权重 防止权重过大或过小'''
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

'''定义去噪自编码的class，包含一个构建函数_init_'''
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        '''激活函数处理'''
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        '''重建操作'''
        self.reconstruction = tf.add(tf.matmul(
            self.hidden, self.weights['w2']), self.weights['b2'])
        '''定义损失函数'''
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        '''全局参数初始化'''
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    '''参数初始化函数'''
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    '''定义训练，用batch进行训练，然后返回cost'''
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    '''编码'''
    def transform(self, X):
        return self.sess.run(self.hidden,
                             feed_dict={self.x: X, self.scale: self.training_scale})

    '''将隐层的编码输出作为下一层的输入'''
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    '''解码操作，重构输出'''
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X, self.scale: self.training_scale})

    '''定义getWeights函数的作用是获取隐含层的权重w1'''
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    '''定义getBiases函数则是获取隐含层的偏置系数b1'''
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


'''随机获取block数据'''
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

'''载入两个训练集以及一个预测样本集'''
def read_data(file1, file2, file3, file4):
    book_train_1 = xlrd.open_workbook(file1)
    book_train_2 = xlrd.open_workbook(file2)
    book_train_3 = xlrd.open_workbook(file3)
    book_target = xlrd.open_workbook(file4)

    sheet_train_1 = book_train_1.sheet_by_index(0)
    sheet_train_2 = book_train_2.sheet_by_index(0)
    sheet_train_3 = book_train_3.sheet_by_index(0)
    sheet_target = book_target.sheet_by_index(0)

    train_data_list = []
    target_data_list = []
    target_validation_list = []

    for row in range(1, sheet_train_1.nrows):
        for col in range(sheet_train_1.ncols - 1):
            value = sheet_train_1.cell(row, col).value
            train_data_list.append(value)
    for row in range(1, sheet_train_2.nrows):
        for col in range(sheet_train_2.ncols - 1):
            value = sheet_train_2.cell(row, col).value
            train_data_list.append(value)
    for row in range(1, sheet_train_3.nrows):
        for col in range(sheet_train_3.ncols - 1):
            value = sheet_train_3.cell(row, col).value
            train_data_list.append(value)
    input_data_train = np.asarray(train_data_list).reshape(
        sheet_train_1.nrows+sheet_train_2.nrows+sheet_train_3.nrows-3, sheet_train_1.ncols-1)

    for row in range(1, sheet_target.nrows):
        value = sheet_target.cell(row, 1).value
        target_data_list.append(value)
    for col in range(sheet_target.ncols):
        for row in range(1, sheet_target.nrows):
            value = sheet_target.cell(row, col).value
            target_validation_list.append(value)
    target_set = np.asarray(target_data_list).reshape(sheet_target.nrows - 1, 1)
    return input_data_train,sheet_target,target_set



'''SDAE训练'''
def train(input_data_train):
    '''定义训练参数'''
    global weight
    training_epochs = 20
    batch_size = 288
    display_step = 1
    stack_size = 3
    hidden_size = [200, 200, 200]
    input_n_size = [2, 200, 200]

    '''建立SDAE'''
    SDAE = []
    for i in range(stack_size):
        autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=input_n_size[i], n_hidden=hidden_size[i],
                                                       transfer_function=tf.nn.softplus,
                                                       optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
                                                       scale=0.01)
        autoencoder._initialize_weights()
        SDAE.append(autoencoder)

    '''保存每个ae的特征'''
    hidden_feature = []

    '''训练SDAE'''
    X_train = np.array([0])
    for j in range(stack_size+1):
        '''输入训练集'''
        if j == 0:
            X_train = np.array(input_data_train)
        elif j == 3:
            X_train_pre = X_train
            X_train = SDAE[j - 1].transform(X_train_pre)
            print(X_train.shape)
            hidden_feature.append(X_train)
            break
        else:
            X_train_pre = X_train
            X_train = SDAE[j-1].transform(X_train_pre)
            print(X_train.shape)
            hidden_feature.append(X_train)
        '''训练并提取特征'''
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)
            for k in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)
                cost = SDAE[j].partial_fit(batch_xs)
                avg_cost += cost / X_train.shape[0] * batch_size
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        weight = SDAE[j].getWeights()
    return weight, hidden_feature




'''SVR预测'''
def pred(hidden_feature, target_set, sheet_target):
    '''定义参数'''
    test_batch_size = 11
    n_data_size = len(target_set)
    y_predict = []
    y_real = []
    MSE = []
    MAE = []
    '''滑动窗口依次读取数据，并进行预测'''
    for i in range(int(n_data_size - test_batch_size) + 1):
        x_set = hidden_feature[2][i:i + test_batch_size]
        y_set = target_set[i:i + test_batch_size]
        x_train = x_set[0:10]
        x_test = np.array(x_set[10].ravel()).reshape(1, -1)
        y_train = y_set[0:10]
        y_test = np.array(y_set[10].ravel()).reshape(1, -1)
        '''标准化特征表达'''
        ss_X = StandardScaler()
        x_train = ss_X.fit_transform(x_train)
        x_test = ss_X.transform(x_test)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        '''选择核函数'''
        SVR_pred = SVR(kernel='linear')
        SVR_pred.fit(x_train, y_train)
        y_predict.append(SVR_pred.predict(x_test))
        y_real.append(y_test[0])

        '''计算模型评价指数'''
        MSE.append(mean_squared_error(y_test, SVR_pred.predict(x_test)))
        MAE.append(mean_absolute_error(y_test, SVR_pred.predict(x_test)))
    '''打印模型评价指数'''
    print('The root mean squared error of SVR is', math.sqrt(sum(MSE) / len(MSE)))
    print('The mean squared error of SVR is', sum(MSE) / len(MSE))
    print('The mean absolute error of SVR is', sum(MAE) / len(MAE))

    '''将预测结果导出到excel'''
    result_book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = result_book.add_sheet('test', cell_overwrite_ok=True)
    for i in range(len(y_predict)):
        sheet.write(test_batch_size - 1 + i, 2, float(y_predict[i]))
    for col in range(sheet_target.ncols - 1):
        for row in range(1, sheet_target.nrows):
            value = sheet_target.cell(row, col).value
            if sheet_target.cell(row, col).ctype == 3:
                date = xldate_as_tuple(sheet_target.cell(row, col).value, 0)
                value = datetime.datetime(*date)
                value = value.strftime("%Y-%m-%d %H:%S:%M")
            sheet.write(row - 1, col, value)
    result_book.save('/Users/Patrick/Desktop/traffic_data/result.xls')

    '''绘制折线对比图'''
    x = range(len(y_predict))
    plt.plot(x, [x[0] for x in y_predict])
    plt.plot(x, list(y_real))
    #plt.legend(handles=[l1, l2], labels=['predict', 'real'], loc='best')
    plt.savefig('/Users/Patrick/Desktop/traffic_data/.png')
    #plt.legend()
    plt.show()

'''选择训练集以及预测集'''
file1 = '/Users/Patrick/Desktop/traffic_data/500010102_aug.xls'
file2 = '/Users/Patrick/Desktop/traffic_data/500010092_aug.xls'
file3 = '/Users/Patrick/Desktop/traffic_data/500010072_aug.xls'
file4 = '/Users/Patrick/Desktop/traffic_data/500010102_oct.xls'

'''开始训练、预测'''
input_data_train, sheet_target, target_set = read_data(file1, file2, file3, file4)
weight, hidden_feature = train(input_data_train)
pred(hidden_feature, target_set, sheet_target)
