from sklearn.neural_network import MLPClassifier
import mysql.connector

cls={'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
clsrev={1:'Iris-setosa', 2:'Iris-versicolor', 3:'Iris-virginica'}

def fetchdata():
	db=mysql.connector.connect(user='root', password='Vmrkvv$2212', host='127.0.0.1', database='vignesh')
	c=db.cursor()
	c.execute('select * from iris')
	return c.fetchall()

def train(x,y):
	neural=MLPClassifier(hidden_layer_sizes=(4,2), max_iter=5000)
	neural.fit(x,y)
	return neural

def test(neural,x,y):
	i=1
	correctcount,wrongcount=0,0
	for xtest,ytest in zip(x,y):
		ypredict=neural.predict([xtest])
		if ypredict[0]==ytest: correctcount+=1
		else: wrongcount+=1
		i+=1
		neural.partial_fit([xtest],[ytest])
	print('Correct count :',correctcount)
	print('Wrong count :',wrongcount)
	return neural
	
def prediction(neural,x):
	y=neural.predict(x)
	return y

data=fetchdata()
x,y=[],[]
traindata=data[:40]+data[50:90]+data[100:140]
testdata=data[40:50]+data[90:100]+data[140:150]
for row in traindata:
	a=[]
	for i in range(5):
		if i==4: y.append(cls[row[i]])
		else: a.append(row[i])
	x.append(a)
neural=train(x,y)
x,y=[],[]
for row in testdata:
	a=[]
	for i in range(5):
		if i==4: y.append(cls[row[i]])
		else: a.append(row[i])
	x.append(a)
neural=test(neural,x,y)
ch='y'
while ch=='y':
	print('Enter the features of the flower to predict the class...')
	z=[]
	z.append(float(input('Enter the sepal length : ')))
	z.append(float(input('Enter the sepal width : ')))
	z.append(float(input('Enter the petal length : ')))
	z.append(float(input('Enter the patel width : ')))
	x=[z]
	y=prediction(neural,x)
	print('The flower might belong to',clsrev[y[0]])
	ch=input('Predict the class of another flower? (y/n) : ')