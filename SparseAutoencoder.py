import numpy as np
import os
import urllib
import urllib.request
import gzip
import struct
import time

class mnist_helper:

    def __init__(self):
        self.train_lbl, self.train_img, self.test_lbl, self.test_img = self.load_mnist_data()

    def get_data(self):
        return self.train_lbl, self.train_img, self.test_lbl, self.test_img

    @staticmethod
    def download_data(url, force_download=False):
        fname = url.split("/")[-1]
        if force_download or not os.path.exists(fname):
            urllib.request.urlretrieve(url, fname)
        return fname

    def load_data(self, label_url, image_url, force_download=False):
        with gzip.open(self.download_data(label_url, force_download)) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(self.download_data(image_url, force_download), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
        return label, image

    def load_mnist_data(self):
        path = 'http://yann.lecun.com/exdb/mnist/'
        train_lbl, train_img = self.load_data(
            path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
        test_lbl, test_img = self.load_data(
            path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

        return self.to_one_hot(train_lbl), train_img, self.to_one_hot(test_lbl), test_img

    @staticmethod
    def to_one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels]

def sigmoid_function(signal, derivative=False):
    if derivative:
        return np.multiply(signal, 1.0 - signal)
    else:
        return 1.0 / (1.0 + np.exp(-signal))

def subtract_err(outputs, targets):
    res = outputs - targets
    return res

class LinearLayer:
    def __init__(self, name, n_in, n_out, activation=sigmoid_function):
        self.name = name
        self.activation = activation
        self.result = []

        self.weights = 2 * np.random.random((n_in, n_out)) - 1
        self.biases = np.zeros(n_out)

    def get_output(self, x):
        result = self.activation(x.dot(self.weights) + self.biases)
        self.result = result
        return result


class SparseLayer(LinearLayer):
    def __init__(self, name, n_in, n_out, activation=sigmoid_function, num_k_sparse=10):
        LinearLayer.__init__(self, name, n_in, n_out, activation)
        self.num_k_sparse = num_k_sparse

    def get_output(self, x):
        result = self.activation(x.dot(self.weights) + self.biases)

        k = self.num_k_sparse
        if k < result.shape[1]:
            for raw in result:
                indices = np.argpartition(raw, -k)[-k:]
                mask = np.ones(raw.shape, dtype=bool)
                mask[indices] = False
                raw[mask] = 0

        self.result = result
        return result
    
def costAE(self, theta, visible_input):
        W1, W2, b1, b2 = self.unpack_theta(theta)
        
        hidden_layer = self.sigmoid(np.dot(W1, visible_input) + b1)
        output_layer = self.sigmoid(np.dot(W2, hidden_layer) + b2)
        m = visible_input.shape[1] 
        
        error = -(visible_input - output_layer)
        sum_sq_error =  0.5 * np.sum(error * error, axis = 0)
        avg_sum_sq_error = np.mean(sum_sq_error)
        reg_cost =  self.lambda_ * (np.sum(W1 * W1) + np.sum(W2 * W2)) / 2.0
        rho_bar = np.mean(hidden_layer, axis=1)
        KL_div = np.sum(self.rho * np.log(self.rho / rho_bar) + 
                        (1 - self.rho) * np.log((1-self.rho) / (1- rho_bar)))        
        cost = avg_sum_sq_error + reg_cost + self.beta * KL_div
        
        # Back propagation
        KL_div_grad = self.beta * (- self.rho / rho_bar + (1 - self.rho) / 
                                    (1 - rho_bar))
        
        del_3 = error * output_layer * (1.0 - output_layer)
        del_2 = np.transpose(W2).dot(del_3) + KL_div_grad[:, np.newaxis]
        del_2 *= hidden_layer * (1 - hidden_layer)
               
        W1_grad = del_2.dot(visible_input.transpose()) / m
        W2_grad = del_3.dot(hidden_layer.transpose()) / m
        b1_grad = del_2
        b2_grad = del_3
        
        W1_grad += self.lambda_ * W1 
        W2_grad += self.lambda_ * W2
        b1_grad = b1_grad.mean(axis = 1)
        b2_grad = b2_grad.mean(axis = 1)
        
        theta_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten(), 
                                     b1_grad.flatten(), b2_grad.flatten()))        
        return [cost, theta_grad]
    
class FCNeuralNet:
    def __init__(self, layers, cost_func=subtract_err):
        self.layers = layers
        self.cost_func = cost_func

    def summary(self, name=None):
        if name != None:
            print(name,":")
        for layer in self.layers:
            print("layer - %s: weights: %s" % (layer.name, layer.weights.shape))

    def train(self, x, y, learning_rate=0.01, epochs=10000,
              batch_size=256, print_epochs=1000,
              monitor_train_accuracy=False):
        # print("training start")
        start_time = time.time()

        for k in range(epochs):
            rand_indices = np.random.randint(x.shape[0], size=batch_size)
            batch_x = x[rand_indices]
            batch_y = y[rand_indices]

            results = self.feed_forward(batch_x)

            error = self.cost_func(results[-1], batch_y)

            if (k+1) % print_epochs == 0:
                loss = np.mean(np.abs(error))
                msg = "epochs: {0}, loss: {1:.4f}".format((k+1), loss)
                if monitor_train_accuracy:
                    accuracy = self.accuracy(x, y)
                    msg += ", accuracy: {0:.2f}%".format(accuracy)
                # print(msg)
            deltas = self.back_propagate(results, error)
            self.update_weights(results, deltas, learning_rate)

        end_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
        print("Training completed".center(65,'-'))
        print("Training time:", elapsed_time)

    def feed_forward(self, x):
        results = [x]
        for i in range(len(self.layers)):
            output_result = self.layers[i].get_output(results[i])
            results.append(output_result)
        return results

    def back_propagate(self, results, error):
        last_layer = self.layers[-1]
        deltas = [error * last_layer.activation(results[-1], derivative=True)]
        for i in range(len(results) - 2, 0, -1):
            layer = self.layers[i]
            delta = deltas[-1].dot(layer.weights.T) * layer.activation(results[i], derivative=True)
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def update_weights(self, results, deltas, learning_rate):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_result = results[i]
            delta = deltas[i]
            layer.weights -= learning_rate * layer_result.T.dot(delta)
            # layer.biases += delta

    def predict(self, x):
        return self.feed_forward(x)[-1]

    def accuracy(self, x_data, y_labels):
        predictions = np.argmax(self.predict(x_data), axis=1)
        labels = np.argmax(y_labels, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy * 100