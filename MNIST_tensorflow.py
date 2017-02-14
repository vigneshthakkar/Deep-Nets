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
print('Train images loaded...')

print('Loading test images...')
images=mnist.test_images()
testimages=[]
for image in images:
	a=[]
	for row in image:
		for item in row:
			a.append(item)
	testimages.append(a)
labels=mnist.test_labels()
testlabels=[]
for label in labels:
	a=[0,0,0,0,0,0,0,0,0,0]
	a[label]=1
	testlabels.append(a)
print('Test images loaded...')

ch='y'
while ch=='y':
	print('The neural network is 2 layer deep...')
	nodes=[]
	for i in range(2): nodes.append(int(input('Enter the number of nodes in layer '+str(i+1)+' : ')))
	
	x=tf.placeholder('float', [None,784])
	y=tf.placeholder('float')
	
	weights={'l1': tf.Variable(tf.random_normal([784,nodes[0]])),
		 'l2': tf.Variable(tf.random_normal([nodes[0],nodes[1]])),
		 'out': tf.Variable(tf.random_normal([nodes[1],10]))}
	
	biases={'l1': tf.Variable(tf.random_normal([nodes[0]])),
		 'l2': tf.Variable(tf.random_normal([nodes[1]])),
		 'out': tf.Variable(tf.random_normal([10]))}
	
	def neural_network(x):
		l1=tf.nn.relu(tf.add(tf.matmul(x,weights['l1']),biases['l1']))
		l2=tf.nn.relu(tf.add(tf.matmul(l1,weights['l2']),biases['l2']))
		out=tf.add(tf.matmul(l2,weights['out']),biases['out'])
		return out
	
	predict_y=neural_network(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y,y))
	optimize=tf.train.AdamOptimizer().minimize(cost)
	
	c='y'
	while c=='y':
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			char='y'
			while char=='y':
				epoch=int(input('Enter the number of epochs : '))
				print('Training the Neural Network...')
				print('This might take some time. Please wait...')
				for i in range(epoch):
					_,loss=sess.run([optimize,cost], feed_dict={x:trainimages, y:trainlabels})
					print('Epoch',i+1,'out of',epoch,'completed...loss=',loss)
				print('Neural Network Trained...')
				print('Testing the neural Network...')
				predictlabels=sess.run(predict_y, feed_dict={x:testimages})
				correct=0
				for i in range(len(testimages)):
					greatest,index=predictlabels[i][0],0
					for j in range(1,10):
						if predictlabels[i][j]>greatest: greatest,index=predictlabels[i][j],j
					if testlabels[i][index]==1: correct+=1
				accuracy=correct/100
				print('The accuracy is', correct)
				char=input('Do you want to increase the number of epochs? (y/n) : ')
		c=input('Try with some other number of epoch? (y/n) : ')
	ch=input('Do you want to retrain the Neural Network? (y/n) : ')
