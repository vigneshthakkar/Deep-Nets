import tensorflow as tf

class Net:

    def __init__(self,sess,numofclasses=2,colored=True):

        self.numofclasses=numofclasses
        self.sess=sess
        self.layer=3
        if colored==False: self.layer=1

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

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