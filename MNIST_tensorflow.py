import tensorflow as tf
import mnist

print('loading train images')
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
print('train images loaded')

print('loading test images')
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
print('test images loaded')
	
n_nodes_hl1=1000
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10
batch_size=100

x=tf.placeholder('float', [None, 784])
y=tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer={'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}
	
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1=tf.nn.relu(l1)
	l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2=tf.nn.relu(l2)
	l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3=tf.nn.relu(l3)
	output=tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])
	return output

def train_neural_network(x):
	prediction=neural_network_model(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	
	hm_epoch=10
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('variables initialized')
		for epoch in range(hm_epoch):
			epoch_loss=0
			for batch in range(int(len(trainimages)/batch_size)):
				epoch_x,epoch_y=trainimages[batch_size*batch:batch_size*(batch+1)],trainlabels[batch_size*batch:batch_size*(batch+1)]
				op,c=sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
				epoch_loss+=c
			print('epoch',epoch+1,', loss',epoch_loss)
		
		predictlabels=sess.run(prediction, feed_dict={x:testimages})
		correct=0
		for i in range(len(testlabels)):
			greatest,index=predictlabels[i][0],0
			for j in range(1,10):
				if predictlabels[i][j]>greatest: greatest,index=predictlabels[i][j],j
			if testlabels[i][index]==1: correct+=1
		print(correct)

train_neural_network(x)