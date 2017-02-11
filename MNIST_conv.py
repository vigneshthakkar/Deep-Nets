import tensorflow as tf
import mnist

print('Loading training images...')
images=mnist.train_images()
trainimages=[]
for image in images:
	a=[]
	for row in image:
		for item in row:
			a.append(item)
	trainimages.append(a)
labels=mnist.train_labels()
trainlabels=[]
for label in labels:
	a=[0,0,0,0,0,0,0,0,0,0]
	a[label]=1
	trainlabels.append(a)
print('Training images loaded...')

print('Loading test images...')
images=mnist.test_images()
testimages=[]
for image in images:
	a=[]
	for row in image:
		for item in row:
			a.append(item)
	testimages.append(a)
labels=mnist.train_labels()
testlabels=[]
for label in labels:
	a=[0,0,0,0,0,0,0,0,0,0]
	a[label]=1
	testlabels.append(a)
print('Test images loaded...')

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def conv(x,w,b):
	return tf.nn.relu(tf.add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'),b))

def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

weights={'conv1': tf.Variable(tf.random_normal([5,5,1,32])),
		 'conv2': tf.Variable(tf.random_normal([5,5,32,64])),
		 'fc': tf.Variable(tf.random_normal([7*7*64,1024])),
		 'out': tf.Variable(tf.random_normal([1024,10]))}

biases={'conv1': tf.Variable(tf.random_normal([32])),
		'conv2': tf.Variable(tf.random_normal([64])),
		'fc': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([10]))}

def neural_network(x):
	x=tf.reshape(x,[-1,28,28,1])
	conv1=max_pool(conv(x,weights['conv1'],biases['conv1']))
	conv2=max_pool(conv(conv1,weights['conv2'],biases['conv2']))
	conv2=tf.reshape(conv2,[-1,7*7*64])
	fc=tf.nn.relu(tf.add(tf.matmul(conv2,weights['fc']),biases['fc']))
	out=tf.add(tf.matmul(fc,weights['out']),biases['out'])
	return out

predict_y=neural_network(x)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y,y))
optimize=tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	epochs=100
	print('Training the Neural Network. This might take some time. Please wait...')
	for epoch in range(epochs):
		sess.run([optimize],feed_dict={x:trainimages,y:trainlabels})
		print('Epoch',epoch+1,'of',epochs,'completed...')
	print('Neura Network Trained...')
	print('Testing the Neural network...')
	predictlabels=sess.run([predict_y],feed_dict={x:testimages})
	correct=0
	for i in range(10000):
		greatest,index=predictlabels[i][0],0
		for j in range(1,10):
			if predictlabels[i][j]>greatest: greatest,index=predictlabels[i][j],j
		if testlabels[index]==1: correct+=1
	print('Accuracy :',correct/100,'%')