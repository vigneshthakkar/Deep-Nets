import tensorflow as tf

class ZFNet:

    def __init__(self,sess,n,numofclasses=2,colored=True):
        
        super().__init__(sess,numofclasses,colored)

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
