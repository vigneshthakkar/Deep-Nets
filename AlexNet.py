import tensorflow as tf

class AlexNet:

    def __init__(self,sess,numofclasses=2,colored=True):

        self.sess=sess
        self.numofclasses=numofclasses

        self.layer=3
        if colored==False: self.layer=1

        self.weights={
            'conv1': tf.Variable(tf.random_normal([11,11,3,96])),
            'conv2': tf.Variable(tf.random_normal([5,5,96,256])),
            'conv3': tf.Variable(tf.random_normal([3,3,256,384])),
            'conv4': tf.Variable(tf.random_normal([3,3,384,384])),
            'conv5': tf.Variable(tf.random_normal([3,3,384,256])),
            'fc1': tf.Variable(tf.random_normal([7*7*256,4096])),
            'fc2': tf.Variable(tf.random_normal([4096,4096])),
            'fc3': tf.Variable(tf.random_normal([4096,self.numofclasses]))
        }

        self.biases={
            'conv1': tf.Variable(tf.random_normal([96])),
            'conv2': tf.Variable(tf.random_normal([256])),
            'conv3': tf.Variable(tf.random_normal([384])),
            'conv4': tf.Variable(tf.random_normal([384])),
            'conv5': tf.Variable(tf.random_normal([256])),
            'fc1': tf.Variable(tf.random_normal([4096])),
            'fc2': tf.Variable(tf.random_normal([4096])),
            'fc3': tf.Variable(tf.random_normal([self.numofclasses]))
        }

        self.x=tf.placeholder('float')
        self.y=tf.placeholder('float')

        self.initialize()

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def maxpool(self,x):
        return tf.nn.max_pool(x,strides=[1,2,2,1],ksize=[1,3,3,1],padding='SAME')

    def conv(self,x,weight,bias):
        return tf.nn.relu(tf.add(tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding='SAME')))

    def localrespnorm(self,x):
        return tf.nn.local_response_normalization(x)

    def fc(self,x,weight,bias):
        return tf.nn.relu(tf.add(tf.matmul(x,weight),bias))

    def network(self):

        z=tf.reshape(self.x,[-1,224,224,self.layer])

        conv1=tf.nn.relu(tf.add(tf.nn.conv2d(z,self.weights['conv1'],strides=[1,4,4,1],padding='SAME')))
        maxpool1=self.maxpool(conv1)
        localrespnorm1=self.localrespnorm(maxpool1)

        conv2=self.conv(localrespnorm1,self.weights['conv2'],self.biases['conv2'])
        maxpool2=self.maxpool(conv2)
        localrespnorm2=self.localrespnorm(maxpool2)

        conv3=self.conv(localrespnorm2,self.weights['conv3'],self.biases['conv3'])
        conv4=self.conv(conv3,self.weights['conv4'],self.biases['conv4'])
        conv5=self.conv(conv4,self.weights['conv5'],self.biases['conv5'])
        maxpool3=self.maxpool(conv5)

        remaxpool3=tf.reshape(maxpool3,[-1,7*7*256])

        fc1=self.fc(remaxpool3,self.weights['fc1'],self.biases['fc1'])
        fc2=self.fc(fc1,self.weights['fc2'],self.biases['fc2'])
        fc3=tf.add(tf.matmul(fc2,self.weights['fc3']),self.biases['fc3'])

        return fc3

    def prediction(self):
        return self.network()

    def cost(self):
        predict_y=self.prediction()
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y,self.y))

    def optimize(self,optimizer,learning_rate):
        if optimizer=='Adam': return tf.train.AdamOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='GradientDescent': return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='Optimizer': return tf.train.Optimizer(learning_rate).minimize(self.cost())
        elif optimizer=='Adadelta': return tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='Adagrad': return tf.train.AdagradOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='AdagradDAO': return tf.train.AdagradDAOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='Momentum': return tf.train.MomentumOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='Ftrl': return tf.train.FtrlOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='ProximalGradientDescent': return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='ProximalAdagrad': return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(self.cost())
        elif optimizer=='RMSProp': return tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost())

    def train(self,x,y,epochs=100,batch_size=100,optimizer='Adam',learning_rate=0.1):
        for epoch in range(epochs):
            for batch in range(int(len(x)/batch_size)):
                batchx,batchy=x[batch_size*batch:batch_size*(batch+1)],y[batch_size*batch:batch_size*(batch+1)]
                self.sess.run(self.optimize(optimizer,learning_rate),feed_dict={self.x:batchx,self.y:batchy})

    def predict(self,x):
        return self.sess.run(self.prediction(),feed_dict={self.x:x})