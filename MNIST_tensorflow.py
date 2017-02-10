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
images=mnist.train_images()
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

ch='y'
while ch=='y':
	n=int(input('Enter the depth of the neural Network : '))
	nodes=[]
	for i in range(n):
		print('Enter the number of nodes in layer', i+1,': ',end='')
		nodes.append(int(input()))
	
	
	x=tf.placeholder('float', [None,784])
	y=tf.placeholder('float')
	
	layer_param=[]
	for i in range(n):
		if i==0: layer_param.append({'weights': tf.Variable(tf.random_normal([784,nodes[i]])), 'biases': tf.Variable(tf.random_normal([nodes[i]]))})
		else: layer_param.append({'weights': tf.Variable(tf.random_normal([nodes[i-1],nodes[i]])), 'biases': tf.Variable(tf.random_normal([nodes[i]]))})
	layer_param.append({'weights': tf.Variable(tf.random_normal([nodes[n-1],10])), 'biases': tf.Variable(tf.random_normal([10]))})
	
	def neural_network(x):
		layeroutput=tf.placeholder('float')
		for i in range(n+1):
			if i==0:
				layeroutput=tf.add(tf.matmul(x,layer_param[i]['weights']),layer_param[i]['biases'])
				layeroutput=tf.nn.relu(layeroutput)
			elif i==n: layeroutput=tf.add(tf.matmul(layeroutput,layer_param[i]['weights']),layer_param[i]['biases'])
			else: 
				layeroutput=tf.add(tf.matmul(layeroutput,layer_param[i]['weights']),layer_param[i]['biases'])
				layeroutput=tf.nn.relu(layeroutput)
		return layeroutput
	
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
					sess.run([optimize], feed_dict={x:trainimages, y:trainlabels})
					print('Epoch',i+1,'out of',epoch,'completed...')
				print('Neural Network Trained...')
				print('Testing the neural Network...')
				predictlabels=sess.run(predict_y, feed_dict={x:testimages})
				correct=0
				for i in range(len(testimages)):
					greatest,index=predictlabels[i][0],0
					for j in range(1,10):
						if predictlabels[i][j]>greatest: greatest,index=predictlabels[i][j],j
					if testlabels[i][j]==1: correct+=1
				accuracy=correct/100
				print('The accuracy is', accuracy,'%')
				char=input('Do you want to increase the number of epochs? (y/n) : ')
		c=input('Try with some other number of epoch? (y/n) : ')
	ch=input('Try with some other configuration of the Neural Network? (y/n) : ')
