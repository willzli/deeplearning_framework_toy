# Compatible with python2.7 and python3.6

import numpy as np
import autodiff_engine as ad

x = ad.Variable(name = "x")
w = ad.Variable(name = "w")
labels = ad.Variable(name = "lables")

# Define Computation graph
p = 1.0 / (1.0 + ad.exp_op((-1.0 * ad.matmul_op(x, w))))
loss = -1.0 * ad.reduce_sum_op(labels * ad.log_op(p) + (1.0 - labels) * ad.log_op(1.0 - p), axis = 0)
grad_y_w, = ad.gradients(loss, [w])

num_features = 2
num_points = 200
num_iterations = 1000
learning_rate = 0.01

# The dummy dataset consists of two classes.
# The classes are modelled as a random normal variables with different means.

class_1_num = int(num_points / 2)
class_2_num = int(num_points / 2)
class_1 = np.random.normal(2, 0.1, (class_1_num, num_features))
class_2 = np.random.normal(4, 0.1, (class_2_num, num_features))
x_val = np.concatenate((class_1, class_2), axis = 0)

x_val = np.concatenate((x_val, np.ones((num_points, 1))), axis = 1)
w_val = np.random.normal(size = (num_features + 1, 1))

labels_val = np.concatenate((np.zeros((class_1_num, 1)), np.ones((class_2_num, 1))), axis=0)
executor = ad.Executor([loss, grad_y_w])

for i in range(100000):
    # evaluate the graph
    loss_val, grad_y_w_val =  executor.run(feed_dict={x:x_val, w:w_val, labels:labels_val})
    # update the parameters using SGD
    w_val = w_val - learning_rate * grad_y_w_val
    if i % 1000 == 0:
        print("Epoch: %d, Learning rate: %.2f, Loss: %f" % (i, learning_rate, loss_val))
