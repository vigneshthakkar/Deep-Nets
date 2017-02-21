import tensorflow as tf

class VGGNet:

    def __init__(self,sess,numofclasses=2,colored=True):

        self.sess=sess
        self.numofclasses=numofclasses
        self.layer=3
        if colored==False: self.layer=1

        self.weights={
            'conv1': tf.Variable(tf.random_normal([3,3,self.layer,64])),
            'conv2': tf.Variable(tf.random_normal([3,3,64,64])),
            'conv3': tf.Variable(tf.random_normal([3,3,64,128])),
            'conv4': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv5': tf.Variable(tf.random_normal([3,3,128,256])),
            'conv6': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv7': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv8': tf.Variable(tf.random_normal([3,3,256,512])),
            'conv9': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv10': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv11': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv12': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv13': tf.Variable(tf.random_normal([3,3,512,512])),
            'fc1': tf.Variable(tf.random_normal([7*7*512,4096])),
            'fc2': tf.Variable(tf.random_normal([4096,4096])),
            'fc3': tf.Variable(tf.random_normal([4096,numofclasses])),
        }

        self.biases={
            'conv1': tf.Variable(tf.random_normal([64])),
            'conv2': tf.Variable(tf.random_normal([64])),
            'conv3': tf.Variable(tf.random_normal([128])),
            'conv4': tf.Variable(tf.random_normal([128])),
            'conv5': tf.Variable(tf.random_normal([256])),
            'conv6': tf.Variable(tf.random_normal([256])),
            'conv7': tf.Variable(tf.random_normal([256])),
            'conv8': tf.Variable(tf.random_normal([512])),
            'conv9': tf.Variable(tf.random_normal([512])),
            'conv10': tf.Variable(tf.random_normal([512])),
            'conv11': tf.Variable(tf.random_normal([512])),
            'conv12': tf.Variable(tf.random_normal([512])),
            'conv13': tf.Variable(tf.random_normal([512])),
            'fc1': tf.Variable(tf.random_normal([4096])),
            'fc2': tf.Variable(tf.random_normal([4096])),
            'fc3': tf.Variable(tf.random_normal([numofclasses])),
        }

        self.x=tf.placeholder('float')
        self.y=tf.placeholder('float')

        self.initialize()

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def conv(self,x,weight,bias):
        return tf.nn.relu(tf.add(tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding='SAME'),bias))

    def maxpool(self,x):
        return tf.nn.max_pool(x,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')

    def fc(self,x,weight,bias):
        return tf.nn.relu(tf.add(tf.matmul(x,weight),bias))

    def network(self):

        z=tf.reshape(self.x,[-1,224,224,self.layer])

        conv1=self.conv(z,self.weights['conv1'],self.biases['conv1'])
        conv2=self.conv(conv1,self.weights['conv2'],self.biases['conv2'])

        maxpool1=self.maxpool(conv2)

        conv3=self.conv(maxpool1,self.weights['conv3'],self.biases['conv3'])
        conv4=self.conv(conv3,self.weights['conv4'],self.biases['conv4'])

        maxpool2=self.maxpool(conv4)

        conv5=self.conv(maxpool2,self.weights['conv5'],self.biases['conv5'])
        conv6=self.conv(conv5,self.weights['conv6'],self.biases['conv6'])
        conv7=self.conv(conv6,self.weights['conv7'],self.biases['conv7'])

        maxpool3=self.maxpool(conv7)

        conv8=self.conv(maxpool3,self.weights['conv8'],self.biases['conv8'])
        conv9=self.conv(conv8,self.weights['conv9'],self.biases['conv9'])
        conv10=self.conv(conv9,self.weights['conv10'],self.biases['conv10'])

        maxpool4=self.maxpool(conv10)

        conv11=self.conv(maxpool4,self.weights['conv11'],self.biases['conv11'])
        conv12=self.conv(conv11,self.weights['conv12'],self.biases['conv12'])
        conv13=self.conv(conv12,self.weights['conv13'],self.biases['conv13'])

        maxpool5=self.maxpool(conv13)

        remaxpool5=tf.reshape(maxpool5,[-1,7*7*512])

        fc1=self.fc(remaxpool5,self.weights['fc1'],self.biases['fc2'])
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