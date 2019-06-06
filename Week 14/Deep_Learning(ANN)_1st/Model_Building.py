#!/usr/bin/env python
# coding: utf-8

# ## Build Tensorflow Model for ANN

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import pickle

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import importlib.util


# In[2]:


tf.__version__


# In[3]:


# Load the preprocessing file 
file = open('CSV_files/Preprocessing_file.pickle','rb')
x_train,y_train,x_cv,y_cv = pickle.load(file)


# In[4]:


# Feature Scaling
"""
we are fitting and transforming the training data using the StandardScaler function.
We standardize our scaling so that we use the same fitted method to transform/scale test data. 
"""
sc = StandardScaler() 
x_train = sc.fit_transform(x_train) 
x_cv = sc.transform(x_cv) 
# Data scaled properly. And done with preprocessing 
# If we fit_tranform on train data then no need to fit it again


# In[5]:


x_train.shape, y_train.shape, x_cv.shape, y_cv.shape,type(x_train), type(y_train)


# In[6]:


x_train


# In[7]:


x_cv.shape, y_cv.shape


# In[8]:


x_cv


# In[32]:


# Import test datset
# Loading testing file from pickle file
test_file = open("CSV_files/Testing_file.pickle","rb")
x_test = pickle.load(test_file)
y_test = pickle.load(test_file) 
print("x_test:",x_test.shape,"y_test", y_test.shape)


# ### Tensorflow Model save and restore

# In[31]:


# number of nodes in each layer
y_train = y_train.reshape(6400,1)
y_cv = y_cv.reshape(1600, 1)
y_test = y_test.reshape(-1,1)

# Neurons in hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
display_step = 1
batch_size = 100

model_path = "ANN_model_files/model.ckpt"

"""A placeholder is simply a variable that we will assign data to at a later date. 
It allows us to create our operations and build our computation graph, without needing the data. 
In Tens"""
x = tf.placeholder('float', [None, x_train.shape[1]])
y = tf.placeholder('float')

def neural_network_model(data):
    # Weight = x_columns * neurons in 1st layer
    # Bias = neurons in 1st layer * 1
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([x_train.shape[1], n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    # Weight = neurons in 1st layer * neurons in 2nd layer
    # Bias = neurons in 2nd layer * 1 ..so on
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    
    # result = ((data * weights) + Bias)
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # activation (relu) function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

# train ANN model using tensorflow
def train_neural_network(x):
    # Here we call neural_network_model() function & passing variable x
    prediction = neural_network_model(x)
#     print("prediction.....",prediction)
    """
    ..The functionality of numpy.mean and tensorflow.reduce_mean are the same. They do the same thing.
    ..tf.nn.softmax produces just the result of applying the softmax function to an input tensor.
    ..Softmax is often used in neural networks, to map the non-normalized output of a network to a
    probability distribution over predicted output classes.
    
    ..Cost variable.
    .This measures how wrong we are, and is the variable we desire to minimize by manipulating our weights
    .To optimize our cost, we will use the AdamOptimizer, which is a popular optimizer along with others 
    like Stochastic Gradient Descent and AdaGrad
    """
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = y) )
#     print("cost.....",cost)
    """
    Within AdamOptimizer(), you can optionally specify the learning_rate as a parameter. 
    The default is 0.001, which is fine for most circumstances.
    optimizer.minimize(cost) is creating new values & variables in your graph.
    """
    # Add the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
#     cost = tf.get_variable(cost)
#     prediction = tf.get_variable(prediction)

    """
    hm_epochs variable which will determine how many epochs to have (cycles of feed forward and back prop
    """
    hm_epochs = 10
    ##########################
    saver = tf.train.Saver()
#     saver = tf.train.Saver([prediction, cost])
    
    # launch the graph in a session
    with tf.Session() as sess:
        #These variables must be initialized before you can train a model.
        # it only add ops(AdamOptimizer) that will initialize the variables (i.e. assign their default value) when run.
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(10):
                # Actually intialize the variables
                _, c = sess.run([optimizer, cost], feed_dict={x:x_train, y:y_train})
                epoch_loss += c
            
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
#         c = tf.get_variable(c)
#         saver = tf.train.Saver([c])
        # Save the model 
        """
        # This will save following files in Tensorflow v >= 0.11 (For next version than .11)
        # my_test_model.data-00000-of-00001
        # my_test_model.index
        # my_test_model.meta
        # checkpoint
        """
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_cv, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) *100
        print('Accuracy:',accuracy.eval({x:x_cv, y:y_cv}))
#         print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
        
        ##############################
         # Save model weights to disk
#         saver.save(sess,  'ANN_model_files/my_test_model')
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
        
        
        # Running a new session
    print("\n\nStarting 2nd session...")
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % save_path)

        # Resume training
        for epoch in range(10):
            avg_cost = 0.
            batch_size = 100
    #         total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
    #         for i in range(total_batch):
            for i in range(10):
#                 batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train,y: y_train})
                # Compute average loss
#                 avg_cost += c / total_batch
                avg_cost += c / 10
            
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost))
        print("Second Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_test, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
    #     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print('Accuracy for test:',accuracy.eval({x:x_test, y:y_test}))

train_neural_network(x)
# print(type(correct))
# print(x_data.shape, y_data.shape, type(x_cv), type(x_cv))
# print(int(mnist.train.num_examples/batch_size)) = 550
# print(int(mnist.train.num_examples))
# print(type(neural_network_model(x)))
# epoch_x, epoch_y = mnist.train.next_batch(batch_size)
# print(type(epoch_x))


# In[ ]:




