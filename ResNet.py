import tensorflow as tf

class ResNet:

    def __init__(self,sess,numofclasses=2,colored=True):

        self.numofclasses=numofclasses
        self.sess=sess
        self.layer=3
        if colored==False: self.layer=1

        self.weights={
            'conv1': tf.Variable(tf.random_normal([7,7,self.layer,64])),
            'conv2': tf.Variable(tf.random_normal([3,3,64,64])),
            'conv3': tf.Variable(tf.random_normal([3,3,64,64])),
            'conv4': tf.Variable(tf.random_normal([3,3,128,64])),
            'conv5': tf.Variable(tf.random_normal([3,3,64,64])),
            'conv6': tf.Variable(tf.random_normal([3,3,129,64])),
            'conv7': tf.Variable(tf.random_normal([3,3,64,64])),
            'conv8': tf.Variable(tf.random_normal([3,3,256,128])),
            'conv9': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv10': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv11': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv12': tf.Variable(tf.random_normal([3,3,256,128])),
            'conv13': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv14': tf.Variable(tf.random_normal([3,3,384,128])),
            'conv15': tf.Variable(tf.random_normal([3,3,128,128])),
            'conv16': tf.Variable(tf.random_normal([3,3,512,256])),
            'conv17': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv18': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv19': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv20': tf.Variable(tf.random_normal([3,3,512,256])),
            'conv21': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv22': tf.Variable(tf.random_normal([3,3,768,256])),
            'conv23': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv24': tf.Variable(tf.random_normal([3,3,1024,256])),
            'conv25': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv26': tf.Variable(tf.random_normal([3,3,1280,256])),
            'conv27': tf.Variable(tf.random_normal([3,3,256,256])),
            'conv28': tf.Variable(tf.random_normal([3,3,1536,512])),
            'conv29': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv30': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv31': tf.Variable(tf.random_normal([3,3,512,512])),
            'conv32': tf.Variable(tf.random_normal([3,3,1024,512])),
            'conv33': tf.Variable(tf.random_normal([3,3,512,512])),
            'fc': tf.Variable(tf.random_normal([7*7*1536,self.numofclasses]))
        }

        self.beta={
            'conv1': tf.Variable(tf.random_normal([64])),
            'conv2': tf.Variable(tf.random_normal([64])),
            'conv3': tf.Variable(tf.random_normal([64])),
            'conv4': tf.Variable(tf.random_normal([64])),
            'conv5': tf.Variable(tf.random_normal([64])),
            'conv6': tf.Variable(tf.random_normal([64])),
            'conv7': tf.Variable(tf.random_normal([64])),
            'conv8': tf.Variable(tf.random_normal([128])),
            'conv9': tf.Variable(tf.random_normal([128])),
            'conv10': tf.Variable(tf.random_normal([128])),
            'conv11': tf.Variable(tf.random_normal([128])),
            'conv12': tf.Variable(tf.random_normal([128])),
            'conv13': tf.Variable(tf.random_normal([128])),
            'conv14': tf.Variable(tf.random_normal([128])),
            'conv15': tf.Variable(tf.random_normal([128])),
            'conv16': tf.Variable(tf.random_normal([256])),
            'conv17': tf.Variable(tf.random_normal([256])),
            'conv18': tf.Variable(tf.random_normal([256])),
            'conv19': tf.Variable(tf.random_normal([256])),
            'conv20': tf.Variable(tf.random_normal([256])),
            'conv21': tf.Variable(tf.random_normal([256])),
            'conv22': tf.Variable(tf.random_normal([256])),
            'conv23': tf.Variable(tf.random_normal([256])),
            'conv24': tf.Variable(tf.random_normal([256])),
            'conv25': tf.Variable(tf.random_normal([256])),
            'conv26': tf.Variable(tf.random_normal([256])),
            'conv27': tf.Variable(tf.random_normal([256])),
            'conv28': tf.Variable(tf.random_normal([512])),
            'conv29': tf.Variable(tf.random_normal([512])),
            'conv30': tf.Variable(tf.random_normal([512])),
            'conv31': tf.Variable(tf.random_normal([512])),
            'conv32': tf.Variable(tf.random_normal([512])),
            'conv33': tf.Variable(tf.random_normal([512])),
        }

        self.gamma={
            'conv1': tf.Variable(tf.random_normal([64])),
            'conv2': tf.Variable(tf.random_normal([64])),
            'conv3': tf.Variable(tf.random_normal([64])),
            'conv4': tf.Variable(tf.random_normal([64])),
            'conv5': tf.Variable(tf.random_normal([64])),
            'conv6': tf.Variable(tf.random_normal([64])),
            'conv7': tf.Variable(tf.random_normal([64])),
            'conv8': tf.Variable(tf.random_normal([128])),
            'conv9': tf.Variable(tf.random_normal([128])),
            'conv10': tf.Variable(tf.random_normal([128])),
            'conv11': tf.Variable(tf.random_normal([128])),
            'conv12': tf.Variable(tf.random_normal([128])),
            'conv13': tf.Variable(tf.random_normal([128])),
            'conv14': tf.Variable(tf.random_normal([128])),
            'conv15': tf.Variable(tf.random_normal([128])),
            'conv16': tf.Variable(tf.random_normal([256])),
            'conv17': tf.Variable(tf.random_normal([256])),
            'conv18': tf.Variable(tf.random_normal([256])),
            'conv19': tf.Variable(tf.random_normal([256])),
            'conv20': tf.Variable(tf.random_normal([256])),
            'conv21': tf.Variable(tf.random_normal([256])),
            'conv22': tf.Variable(tf.random_normal([256])),
            'conv23': tf.Variable(tf.random_normal([256])),
            'conv24': tf.Variable(tf.random_normal([256])),
            'conv25': tf.Variable(tf.random_normal([256])),
            'conv26': tf.Variable(tf.random_normal([256])),
            'conv27': tf.Variable(tf.random_normal([256])),
            'conv28': tf.Variable(tf.random_normal([512])),
            'conv29': tf.Variable(tf.random_normal([512])),
            'conv30': tf.Variable(tf.random_normal([512])),
            'conv31': tf.Variable(tf.random_normal([512])),
            'conv32': tf.Variable(tf.random_normal([512])),
            'conv33': tf.Variable(tf.random_normal([512])),
            'fc': tf.Variable(tf.random_normal([self.numofclasses]))
        }

        x=tf.placeholder('float')
        y=tf.placeholder('float')

        self.initialize()

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def conv(self,x,weight,stride,gamma,beta):
        y=tf.nn.conv2d(x,weight,strides=[1,stride,stride,1],padding='SAME')
        mean,variance=tf.nn.moments(y,[0,1,2])
        z=tf.nn.batch_normalization(y,mean,variance,beta,gamma)
        return z

    def maxpool(self,x,stride,kernelsize):
        return tf.nn.max_pool(x,strides=[1,stride,stride,1],ksize=[1,kernelsize,kernelsize,1],padding='SAME')

    def depthconcat(self,x,y):
        return tf.concat([x,y],3)

    def avgpool(self,x,stride,kernelsize):
        return tf.nn.avg_pool(x,ksize=[1,kernelsize,kernelsize,1],strides=[1,stride,stride,1],padding='SAME')

    def fc(self,x,weight,bias):
        return tf.add(tf.matmul(x,weight),bias)

    def network(self):

        z=tf.reshape(self.x,[-1,224,224,self.layer])

        conv1=self.conv(z,self.weights['conv1'],2,self.gamma['conv1'],self.beta['conv1'])
        maxpool1=self.maxpool(conv1,2,3)

        conv2=self.conv(maxpool1,self.weights['conv2'],1,self.gamma['conv2'],self.beta['conv2'])
        conv3=self.conv(conv2,self.weights['conv3'],1,self.gamma['conv3'],self.beta['conv3'])
        depthconcat1=self.depthconcat(conv3,maxpool1)

        conv4=self.conv(depthconcat1,self.weights['conv4'],1,self.gamma['conv4'],self.beta['conv4'])
        conv5=self.conv(conv4,self.weights['conv5'],1,self.gamma['conv5'],self.beta['conv5'])
        depthconcat2=self.depthconcat(conv5,depthconcat1)

        conv6=self.conv(depthconcat2,self.weights['conv6'],1,self.gamma['conv6'],self.beta['conv6'])
        conv7=self.conv(conv6,self.weights['conv7'],1,self.gamma['conv7'],self.beta['conv7'])
        depthconcat3=self.depthconcat(conv7,depthconcat2)

        conv8=self.conv(depthconcat3,self.weights['conv8'],2,self.gamma['conv8'],self.beta['conv8'])
        conv9=self.conv(conv8,self.weights['conv9'],1,self.gamma['conv9'],self.beta['conv9'])

        conv10=self.conv(conv9,self.weights['conv10'],1,self.gamma['conv10'],self.beta['conv10'])
        conv11=self.conv(conv10,self.weights['conv11'],1,self.gamma['conv11'],self.beta['conv11'])
        depthconcat4=self.depthconcat(conv11,conv9)

        conv12=self.conv(depthconcat4,self.weights['conv12'],1,self.gamma['conv12'],self.beta['conv12'])
        conv13=self.conv(conv12,self.weights['conv13'],1,self.gamma['conv13'],self.beta['conv13'])
        depthconcat5=self.depthconcat(conv13,depthconcat4)

        conv14=self.conv(depthconcat4,self.weights['conv14'],1,self.gamma['conv14'].self.beta['conv14'])
        conv15=self.conv(conv14,self.weights['conv15'],1,self.gamma['conv15'],self.beta['conv15'])
        depthconcat6=self.depthconcat(conv15,depthconcat5)

        conv16=self.conv(depthconcat6,self.weights['conv16'],2,self.gamma['conv16'],self.beta['conv16'])
        conv17=self.conv(conv16,self.weights['conv17'],1,self.gamma['conv17'],self.beta['conv17'])

        conv18=self.conv(conv17,self.weights['conv18'],1,self.gamma['conv18'],self.beta['conv18'])
        conv19=self.conv(conv18,self.weights['conv19'],1,self.gamma['conv19'],self.beta['conv19'])
        depthconcat7=self.depthconcat(conv19,conv17)

        conv20=self.conv(depthconcat7,self.weights['conv20'],1,self.gamma['conv20'],self.beta['conv20'])
        conv21=self.conv(conv20,self.weights['conv21'],1,self.gamma['conv21'],self.beta['conv21'])
        depthconcat8=self.depthconcat(conv21,depthconcat7)

        conv22=self.conv(depthconcat8,self.weights['conv22'],1,self.gamma['conv22'],self.beta['conv22'])
        conv23=self.conv(conv22,self.weights['conv23'],1,self.gamma['conv23'],self.beta['conv23'])
        depthconcat9=self.depthconcat(conv23,depthconcat8)

        conv24=self.conv(depthconcat9,self.weights['conv24'],1,self.gamma['conv24'],self.beta['conv24'])
        conv25=self.conv(conv24,self.weights['conv25'],1,self.gamma['conv25'],self.beta['conv25'])
        depthconcat10=self.depthconcat(conv25,depthconcat9)

        conv26=self.conv(depthconcat10,self.weights['conv26'],1,self.gamma['conv26'],self.beta['conv26'])
        conv27=self.conv(conv26,self.weights['conv27'],1,self.gamma['conv27'],self.beta['conv27'])
        depthconcat11=self.depthconcat(conv27,depthconcat10)

        conv28=self.conv(depthconcat11,self.weights['conv28'],2,self.gamma['conv18'],self.beta['conv28'])
        conv29=self.conv(conv28,self.weights['conv29'],1,self.gamma['conv29'],self.beta['conv29'])

        conv30=self.conv(conv29,self.weights['conv30'],1,self.gamma['conv30'],self.beta['conv30'])
        conv31=self.conv(conv30,self.weights['conv31'],1,self.gamma['conv31'],self.beta['conv31'])
        depthconcat12=self.depthconcat(conv31,conv29)

        conv32=self.conv(depthconcat12,self.weights['conv32'],1,self.gamma['conv32'],self.beta['conv32'])
        conv33=self.conv(conv32,self.weights['conv33'],1,self.gamma['conv33'],self.beta['conv33'])
        depthconcat13=self.depthconcat(conv33,depthconcat12)

        avgpool1=self.avgpool(depthconcat13,1,3)
        reavgpool1=tf.reshape(avgpool1,[-1,7*7*1536])

        fc1=self.fc(reavgpool1,self.weights['fc'],self.beta['fc'])

        return fc1

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