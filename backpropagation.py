
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import sklearn.utils as sk
from datetime import datetime

def softmax_function(u):
    exp_u = np.exp(u)#матрица
    sum_K = exp_u.sum(1)
    u = np.transpose(exp_u)/sum_K 
    u = np.transpose(u)
    return u

def LReLU_function_v(value):
    if value < 0:
        return 0.1 * value
    else:
        return value

def LReLU_function(values):
    return np.vectorize(LReLU_function_v)(values)

def derivative_LReLU_v(value):
    if value < 0:
        return 0.1
    else:
        return 1
    
def derivative_LReLU(values):
    return np.vectorize(derivative_LReLU_v)(values)

class network:
    def __init__(self, num_hidden_neurons_v, num_output_neurons_u):
        self.speed = 0
        self.batch = 0
        
        #число нейронов входного слоя
        self.num_input_neurons_x = 0
        
        #число нейронов скрытого слоя
        self.num_hidden_neurons_v = num_hidden_neurons_v       
        self.V = 0
        
        #число нейронов выходного слоя
        self.num_output_neurons_u = num_output_neurons_u
        self.U = 0
        
        self.w2 = np.array([])
        self.w1 = np.array([])
        
        self.der_LReLU = 0
        
    def random_w(self):
        c1 = np.sqrt(2 / self.num_input_neurons_x)
        c2 = np.sqrt(2 / self.num_hidden_neurons_v)
        
        self.w1 = c1 * np.random.randn(self.num_hidden_neurons_v, self.num_input_neurons_x)
        self.w2 = c2 * np.random.randn(self.num_output_neurons_u, self.num_hidden_neurons_v)
    
    def error_function(self, x_train, y_train):
        
        logU = np.log(self.U)
        crossentropy = -np.sum(y_train * logU)
        crossentropy = (1 / x_train.shape[0]) * crossentropy

        u1 = np.argmax(self.U, 1)
        y1 = np.argmax(y_train, 1) #по строчкам
        accuracy = (u1 == y1).mean()*100

        return crossentropy, accuracy
    
    def test_error_function(self, x_test, y_test):
        V = np.matmul(x_test, np.transpose(self.w1))#(128, 300)
        V = LReLU_function(V)
        
        U = np.matmul(V, np.transpose(self.w2))#(10, 128)
        U = softmax_function(U)
        
        logU = np.log(U)
        crossentropy = -np.sum(y_test * logU)
        crossentropy = (1 / x_test.shape[0]) * crossentropy

        u1 = np.argmax(U, 1)
        y1 = np.argmax(y_test, 1) #по строчкам
        accuracy = (u1 == y1).mean()*100

        return crossentropy, accuracy
    
    def straight_run(self, x_train, st, end):
        #(128, 785)
        V = np.matmul(x_train, np.transpose(self.w1))#(128, 300)
        V = LReLU_function(V)
        
        self.der_LReLU[st:end] = derivative_LReLU(V)
        self.V[st:end] = V
        
        U = np.matmul(V, np.transpose(self.w2))#(10, 128)
        U = softmax_function(U)
        
        self.U[st:end] = U
     
    def back_run(self, x_train, y_train, st, end):
        s = - y_train + self.U[st:end]
        delta_w2 = np.matmul(np.transpose(s), self.V[st:end])

        sumw = np.matmul(s, self.w2) 
        dsumw = sumw * self.der_LReLU[st:end]
        delta_w1 = np.matmul(np.transpose(dsumw), x_train)

        self.w1 = self.w1 - (self.speed / self.batch) * delta_w1
        self.w2 = self.w2 - (self.speed / self.batch) * delta_w2
        
    def fit(self, x_train, y_train, batch, speed_train, epoch):
        np.random.seed()
        self.batch = batch
        self.speed = speed_train
        self.num_input_neurons_x = x_train.shape[1]
        self.random_w()
        
        self.der_LReLU = np.zeros((x_train.shape[0], self.num_hidden_neurons_v)) 
        self.V = np.zeros((x_train.shape[0], self.num_hidden_neurons_v))
        self.U = np.zeros((x_train.shape[0], self.num_output_neurons_u))

        for j in range(epoch):
            x_train, y_train = sk.shuffle(x_train, y_train)
            for i in range(0, x_train.shape[0], self.batch):
                self.straight_run(x_train[i:i + self.batch], i, i + self.batch)
                self.back_run(x_train[i:i + self.batch], y_train[i:i + self.batch], i, i + self.batch)
            crossentropy, accuracy = self.error_function(x_train, y_train)
            print('Epoch: ', j, 'Train_accuracy %: ', accuracy, 'Сrossentropy: ', crossentropy)

def run(num_hidden_neurons_v, epoch, batch, speed_train):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    
    ones = np.ones((60000, 1))
    train_images = np.hstack((ones, train_images))
    num_hidden_neurons_x = train_images.shape[0]
    
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    
    ones = np.ones((10000, 1))
    test_images = np.hstack((ones, test_images))
    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    number_classes = len(train_labels[0]) #число классов = 10 (от 0 до 10)
    
    net = network(num_hidden_neurons_v, number_classes)

    time_start = datetime.now()
    net.fit(train_images, train_labels, batch, speed_train, epoch)
    time = datetime.now() - time_start
    
    train_result = net.error_function(train_images, train_labels)
    test_result = net.test_error_function(test_images, test_labels)
    
    return train_result, test_result, time

num_hidden_neurons_v = 300
epoch = 20
batch = 32
speed_train = 0.1

train_result, test_result, time = run(num_hidden_neurons_v, epoch, batch, speed_train)

print('Test_accuracy: ', test_result[1], 'Сrossentropy', test_result[0])
print('Time: ', time)