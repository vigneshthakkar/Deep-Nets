import tensorflow as tf

class ZFNet:

    def __init__(self,sess,n,numofclasses=2,colored=True):
        self.numofclasses=numofclasses
        self.sess=sess
        self.layer=3
        if colored==False: self.layer=1

        self.weights={
            'conv1': tf.Variable(tf.random_normal([7,7,self.layer,96])),
            'conv2': tf.Variable(tf.random_normal([5,5,96,256])),
            'conv3': tf.Variable(tf.random_normal([3,3,256,384])),
            'conv4': tf.Variable(tf.random_normal([3,3,384,384])),
            'conv5': tf.Variable(tf.random_normal([3,3,384,256])),
            'fc1': tf.Variable(tf.random_normal([7*7*256,4096])),
            'fc2': tf.Variable(tf.random_normal([4096,4096])),
            'fc3': tf.Variable(tf.random_normal([4096,self.numofclasses])),
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

        x=tf.placeholder('float')
        y=tf.placeholder('float')

        self.initialize()

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def conv(self,x,weight,bias,stride):
        return tf.nn.relu(tf.add(tf.nn.conv2d(x,weight,strides=[1,stride,stride,1],padding='SAME'),bias))

    def maxpool(self,x,stride,kernelsize):
        return tf.nn.max_pool(x,strides=[1,stride,stride,1],ksize=[1,kernelsize,kernelsize,1],padding='SAME')

    def contrastnorm(self,x):
        y=[]
        for item in x:
            y.append(tf.image.per_image_standardization(item))
        return y

    def fc(self,x,weight,bias):
        return tf.nn.relu(tf.add(tf.matmul(x,weight),bias))

    def network(self):

        z=tf.reshape(self.x,[-1,224,224,self.layer])

        conv1=self.conv(z,self.weights['conv1'],self.biases['conv1'],2)
        maxpool1=self.maxpool(conv1,2,3)
        contrastnorm1=self.contrastnorm(maxpool1)

        conv2=self.conv(contrastnorm1,self.weights['conv2'],self.biases['conv2'],2)
        maxpool2=self.maxpool(conv2,2,3)
        contrastnorm2=self.contrastnorm(maxpool2)

        conv3=self.conv(contrastnorm2,self.weights['conv3'],self.biases['conv3'],1)
        maxpool3=self.maxpool(conv3,1,3)
        contrastnorm3=self.contrastnorm(maxpool3)

        conv4=self.conv(contrastnorm3,self.weights['conv4'],self.biases['conv4'],1)
        maxpool4=self.maxpool(conv4,1,3)
        contrastnorm4=self.contrastnorm(maxpool4)

        conv5=self.conv(contrastnorm4,self.weights['conv5'],self.biases['conv5'],1)
        maxpool5=self.maxpool(conv5,2,3)
        contrastnorm5=self.contrastnorm(maxpool5)

        recontrastnorm5=tf.reshape(contrastnorm5,[-1,7*7*256])

        fc1=self.fc(recontrastnorm5,self.weights['fc1'],self.biases['fc1'])
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