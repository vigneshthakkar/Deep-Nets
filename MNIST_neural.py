import mnist
from sklearn.neural_network import MLPClassifier

print('Loading training images...')
images=mnist.train_images()
trainimages=[]
for image in images:
	a=[]
	for row in image:
		for item in row:
			a.append(item)
	trainimages.append(a)
trainlabels=mnist.train_labels()
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
testlabels=mnist.test_labels()
print('Test images loaded...')

neural=MLPClassifier(hidden_layer_sizes=(784,784,), shuffle=False, activation='logistic')

print('Training the Neural Network...')
neural.fit(trainimages,trainlabels)
print('Neural Network trained...')

print('Testing the Neural Network...')
predictlabels=neural.predict(testimages)
count=0
for x,y in zip(predictlabels, testlabels):
	if x==y: count+=1
print('Accuracy :', count/10000*100, '%')