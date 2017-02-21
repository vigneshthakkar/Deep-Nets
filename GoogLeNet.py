import tensorflow as tf

class GoogLeNet:

    def __init__(self,sess,numofclass=2,colored=True):

        self.numofclass = numofclass

        self.sess = sess

        self.layer = 3
        if colored == False: self.layer = 1

        self.weights={'conv1': tf.Variable(tf.random_normal([7,7,self.layer,64])),

                 'conv2': tf.Variable(tf.random_normal([1,1,64,64])),
                 'conv3': tf.Variable(tf.random_normal([3,3,64,192])),

                 'conv4': tf.Variable(tf.random_normal([1,1,192,64])),
                 'conv5': tf.Variable(tf.random_normal([3,3,64,128])),
                 'conv6': tf.Variable(tf.random_normal([5,5,64,32])),
                 'conv7': tf.Variable(tf.random_normal([1,1,192,32])),

                 'conv8': tf.Variable(tf.random_normal([1,1,256,128])),
                 'conv9': tf.Variable(tf.random_normal([3,3,128,192])),
                 'conv10': tf.Variable(tf.random_normal([5,5,128,96])),
                 'conv11': tf.Variable(tf.random_normal([1,1,256,64])),

                 'conv12': tf.Variable(tf.random_normal([1,1,480,192])),
                 'conv13': tf.Variable(tf.random_normal([3,3,192,208])),
                 'conv14': tf.Variable(tf.random_normal([5,5,192,48])),
                 'conv15': tf.Variable(tf.random_normal([1,1,480,64])),

                 'conv16': tf.Variable(tf.random_normal([1,1,512,160])),
                 'conv17': tf.Variable(tf.random_normal([3,3,160,224])),
                 'conv18': tf.Variable(tf.random_normal([5,5,160,64])),
                 'conv19': tf.Variable(tf.random_normal([1,1,512,64])),

                 'conv20': tf.Variable(tf.random_normal([1,1,512,128])),
                 'conv21': tf.Variable(tf.random_normal([3,3,128,256])),
                 'conv22': tf.Variable(tf.random_normal([5,5,128,64])),
                 'conv23': tf.Variable(tf.random_normal([1,1,512,64])),

                 'conv24': tf.Variable(tf.random_normal([1,1,512,112])),
                 'conv25': tf.Variable(tf.random_normal([3,3,112,288])),
                 'conv26': tf.Variable(tf.random_normal([5,5,112,64])),
                 'conv27': tf.Variable(tf.random_normal([1,1,512,64])),

                 'conv28': tf.Variable(tf.random_normal([1,1,528,256])),
                 'conv29': tf.Variable(tf.random_normal([3,3,256,320])),
                 'conv30': tf.Variable(tf.random_normal([5,5,256,128])),
                 'conv31': tf.Variable(tf.random_normal([1,1,528,128])),

                 'conv32': tf.Variable(tf.random_normal([1,1,832,256])),
                 'conv33': tf.Variable(tf.random_normal([3,3,256,320])),
                 'conv34': tf.Variable(tf.random_normal([5,5,256,128])),
                 'conv35': tf.Variable(tf.random_normal([1,1,832,128])),

                 'conv36': tf.Variable(tf.random_normal([1,1,832,384])),
                 'conv37': tf.Variable(tf.random_normal([3,3,384,384])),
                 'conv38': tf.Variable(tf.random_normal([5,5,384,128])),
                 'conv39': tf.Variable(tf.random_normal([1,1,832,128])),

                 'fc1': tf.Variable(tf.random_normal([1024,1000])),
                 'fc2': tf.Variable(tf.random_normal([1000,self.numofclass])),
                 }

        self.biases={'conv1': tf.Variable(tf.random_normal([64])),

                'conv2': tf.Variable(tf.random_normal([64])),
                'conv3': tf.Variable(tf.random_normal([192])),

                'conv4': tf.Variable(tf.random_normal([64])),
                'conv5': tf.Variable(tf.random_normal([128])),
                'conv6': tf.Variable(tf.random_normal([32])),
                'conv7': tf.Variable(tf.random_normal([32])),

                'conv8': tf.Variable(tf.random_normal([128])),
                'conv9': tf.Variable(tf.random_normal([192])),
                'conv10': tf.Variable(tf.random_normal([96])),
                'conv11': tf.Variable(tf.random_normal([64])),

                'conv12': tf.Variable(tf.random_normal([192])),
                'conv13': tf.Variable(tf.random_normal([208])),
                'conv14': tf.Variable(tf.random_normal([48])),
                'conv15': tf.Variable(tf.random_normal([64])),

                'conv16': tf.Variable(tf.random_normal([160])),
                'conv17': tf.Variable(tf.random_normal([224])),
                'conv18': tf.Variable(tf.random_normal([64])),
                'conv19': tf.Variable(tf.random_normal([64])),

                'conv20': tf.Variable(tf.random_normal([128])),
                'conv21': tf.Variable(tf.random_normal([256])),
                'conv22': tf.Variable(tf.random_normal([64])),
                'conv23': tf.Variable(tf.random_normal([64])),

                'conv24': tf.Variable(tf.random_normal([112])),
                'conv25': tf.Variable(tf.random_normal([288])),
                'conv26': tf.Variable(tf.random_normal([64])),
                'conv27': tf.Variable(tf.random_normal([64])),

                'conv28': tf.Variable(tf.random_normal([256])),
                'conv29': tf.Variable(tf.random_normal([320])),
                'conv30': tf.Variable(tf.random_normal([128])),
                'conv31': tf.Variable(tf.random_normal([128])),

                'conv32': tf.Variable(tf.random_normal([256])),
                'conv33': tf.Variable(tf.random_normal([320])),
                'conv34': tf.Variable(tf.random_normal([128])),
                'conv35': tf.Variable(tf.random_normal([128])),

                'conv36': tf.Variable(tf.random_normal([384])),
                'conv37': tf.Variable(tf.random_normal([384])),
                'conv38': tf.Variable(tf.random_normal([128])),
                'conv39': tf.Variable(tf.random_normal([128])),

                'fc1': tf.Variable(tf.random_normal([1000])),
                'fc2': tf.Variable(tf.random_normal([self.numofclass])),
                }
        self.x=tf.placeholder('float')
        self.y=tf.placeholder('float')

        self.initialize()

    def network(self):

        z=tf.reshape(self.x,[-1,224,224,self.layer])

        conv1=tf.nn.relu(tf.add(tf.nn.conv2d(z,self.weights['conv1'],strides=[1,2,2,1],padding='SAME'),self.biases['conv1']))
        maxpool1=tf.nn.max_pool(conv1,strides=[1,2,2,1],ksize=[1,3,3,1],padding='SAME')
        localrespnorm1=tf.nn.local_response_normalization(maxpool1)

        conv2=tf.nn.relu(tf.add(tf.nn.conv2d(localrespnorm1,self.weights['conv2'],strides=[1,1,1,1],padding='SAME'),self.biases['conv2']))
        conv3=tf.nn.relu(tf.add(tf.nn.conv2d(conv2,self.weights['conv3'],strides=[1,1,1,1],padding='SAME'),self.biases['conv3']))
        localrespnorm2=tf.nn.local_response_normalization(conv3)
        maxpool2=tf.nn.max_pool(localrespnorm2,strides=[1,2,2,1],ksize=[1,3,3,1],padding='SAME')

        conv4=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool2,self.weights['conv4'],strides=[1,1,1,1],padding='SAME'),self.biases['conv4']))
        maxpool3=tf.nn.max_pool(maxpool2,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv5=tf.nn.relu(tf.add(tf.nn.conv2d(conv4,self.weights['conv5'],strides=[1,1,1,1],padding='SAME'),self.biases['conv5']))
        conv6=tf.nn.relu(tf.add(tf.nn.conv2d(conv4,self.weights['conv6'],strides=[1,1,1,1],padding='SAME'),self.biases['conv6']))
        conv7=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool3,self.weights['conv7'],strides=[1,1,1,1],padding='SAME'),self.biases['conv7']))
        depthconcat1=tf.concat([conv4,conv5,conv6,conv7],3)

        conv8=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat1,self.weights['conv8'],strides=[1,1,1,1],padding='SAME'),self.biases['conv8']))
        maxpool4=tf.nn.max_pool(depthconcat1,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv9=tf.nn.relu(tf.add(tf.nn.conv2d(conv8,self.weights['conv9'],strides=[1,1,1,1],padding='SAME'),self.biases['conv9']))
        conv10=tf.nn.relu(tf.add(tf.nn.conv2d(conv8,self.weights['conv10'],strides=[1,1,1,1],padding='SAME'),self.biases['conv10']))
        conv11=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool4,self.weights['conv11'],strides=[1,1,1,1],padding='SAME'),self.biases['conv11']))
        depthconcat2=tf.concat([conv8,conv9,conv10,conv11],3)

        maxpool5=tf.nn.max_pool(depthconcat2,strides=[1,2,2,1],ksize=[1,3,3,1],padding='SAME')

        conv12=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool5,self.weights['conv12'],strides=[1,1,1,1],padding='SAME'),self.biases['conv12']))
        maxpool6=tf.nn.max_pool(maxpool5,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv13=tf.nn.relu(tf.add(tf.nn.conv2d(conv12,self.weights['conv13'],strides=[1,1,1,1],padding='SAME'),self.biases['conv13']))
        conv14=tf.nn.relu(tf.add(tf.nn.conv2d(conv12,self.weights['conv14'],strides=[1,1,1,1],padding='SAME'),self.biases['conv14']))
        conv15=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool6,self.weights['conv15'],strides=[1,1,1,1],padding='SAME'),self.biases['conv15']))
        depthconcat3=tf.concat([conv12,conv13,conv14,conv15],3)

        conv16=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat3,self.weights['conv16'],strides=[1,1,1,1],padding='SAME'),self.biases['conv16']))
        maxpool7=tf.nn.max_pool(depthconcat3,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv17=tf.nn.relu(tf.add(tf.nn.conv2d(conv16,self.weights['conv17'],strides=[1,1,1,1],padding='SAME'),self.biases['conv17']))
        conv18=tf.nn.relu(tf.add(tf.nn.conv2d(conv16,self.weights['conv18'],strides=[1,1,1,1],padding='SAME'),self.biases['conv18']))
        conv19=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool7,self.weights['conv19'],strides=[1,1,1,1],padding='SAME'),self.biases['conv19']))
        depthconcat4=tf.concat([conv16,conv17,conv18,conv19],3)

        conv20=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat4,self.weights['conv20'],strides=[1,1,1,1],padding='SAME'),self.biases['conv20']))
        maxpool8=tf.nn.max_pool(depthconcat4,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv21=tf.nn.relu(tf.add(tf.nn.conv2d(conv20,self.weights['conv21'],strides=[1,1,1,1],padding='SAME'),self.biases['conv21']))
        conv22=tf.nn.relu(tf.add(tf.nn.conv2d(conv20,self.weights['conv22'],strides=[1,1,1,1],padding='SAME'),self.biases['conv22']))
        conv23=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool8,self.weights['conv23'],strides=[1,1,1,1],padding='SAME'),self.biases['conv23']))
        depthconcat5=tf.concat([conv20,conv21,conv22,conv23],3)

        conv24=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat5,self.weights['conv24'],strides=[1,1,1,1],padding='SAME'),self.biases['conv24']))
        maxpool9=tf.nn.max_pool(depthconcat5,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv25=tf.nn.relu(tf.add(tf.nn.conv2d(conv24,self.weights['conv25'],strides=[1,1,1,1],padding='SAME'),self.biases['conv25']))
        conv26=tf.nn.relu(tf.add(tf.nn.conv2d(conv24,self.weights['conv26'],strides=[1,1,1,1],padding='SAME'),self.biases['conv26']))
        conv27=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool9,self.weights['conv27'],strides=[1,1,1,1],padding='SAME'),self.biases['conv27']))
        depthconcat6=tf.concat([conv24,conv25,conv26,conv27],3)
        
        conv28=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat5,self.weights['conv28'],strides=[1,1,1,1],padding='SAME'),self.biases['conv28']))
        maxpool10=tf.nn.max_pool(depthconcat6,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv29=tf.nn.relu(tf.add(tf.nn.conv2d(conv28,self.weights['conv29'],strides=[1,1,1,1],padding='SAME'),self.biases['conv29']))
        conv30=tf.nn.relu(tf.add(tf.nn.conv2d(conv28,self.weights['conv30'],strides=[1,1,1,1],padding='SAME'),self.biases['conv30']))
        conv31=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool10,self.weights['conv31'],strides=[1,1,1,1],padding='SAME'),self.biases['conv31']))
        depthconcat7=tf.concat([conv28,conv29,conv30,conv31],3)
        
        maxpool11=tf.nn.max_pool(depthconcat7,strides=[1,2,2,1],ksize=[1,3,3,1],padding='SAME')
        
        conv32=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool11,self.weights['conv32'],strides=[1,1,1,1],padding='SAME'),self.biases['conv32']))
        maxpool12=tf.nn.max_pool(maxpool11,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv33=tf.nn.relu(tf.add(tf.nn.conv2d(conv32,self.weights['conv33'],strides=[1,1,1,1],padding='SAME'),self.biases['conv33']))
        conv34=tf.nn.relu(tf.add(tf.nn.conv2d(conv32,self.weights['conv34'],strides=[1,1,1,1],padding='SAME'),self.biases['conv34']))
        conv35=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool12,self.weights['conv35'],strides=[1,1,1,1],padding='SAME'),self.biases['conv35']))
        depthconcat8=tf.concat([conv32,conv33,conv34,conv35],3)
        
        conv36=tf.nn.relu(tf.add(tf.nn.conv2d(depthconcat8,self.weights['conv36'],strides=[1,1,1,1],padding='SAME'),self.biases['conv36']))
        maxpool13=tf.nn.max_pool(depthconcat8,strides=[1,1,1,1],ksize=[1,3,3,1],padding='SAME')
        conv37=tf.nn.relu(tf.add(tf.nn.conv2d(conv36,self.weights['conv37'],strides=[1,1,1,1],padding='SAME'),self.biases['conv37']))
        conv38=tf.nn.relu(tf.add(tf.nn.conv2d(conv36,self.weights['conv38'],strides=[1,1,1,1],padding='SAME'),self.biases['conv38']))
        conv39=tf.nn.relu(tf.add(tf.nn.conv2d(maxpool13,self.weights['conv39'],strides=[1,1,1,1],padding='SAME'),self.biases['conv39']))
        depthconcat9=tf.concat([conv36,conv37,conv38,conv39],3)
        
        avgpool1=tf.nn.avg_pool(depthconcat9,strides=[1,1,1,1],ksize=[1,7,7,1],padding='VALID')
        reavgpool1=tf.reshape(avgpool1,[-1,1024])
        
        fc1=tf.nn.relu(tf.add(tf.matmul(reavgpool1,self.weights['fc1']),self.biases['fc1']))
        fc2=tf.add(tf.matmul(fc1,self.weights['fc2']),self.biases['fc2'])
        
        return fc2

    def prediction(self):
        return self.network()

    def cost(self):
        predict_y=self.prediction()
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y,self.y))

    def train(self,optimizer,learning_rate=0.1):
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

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def fit(self,x,y,epochs=100,batch_size=100,optimizer='Optimizer',learning_rate=0.1):
        for epoch in range(epochs):
                for batch in range(int(len(x)/batch_size)):
                    epochx,epochy=x[batch_size*batch:batch_size*(batch+1)],y[batch_size*batch:batch_size*(batch+1)]
                    self.sess.run(self.train(optimizer,learning_rate),feed_dict={self.x:epochx,self.y:epochy})

    def predict(self,x):
        return self.sess.run(self.prediction(),feed_dict={self.x:x})