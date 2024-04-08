import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load data from saved data set
data_dict = pickle.load(open('./data.pickle', 'rb'))

# split data from data set
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# split samples into 2 parts: samples for training and samples for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=labels)

# training random forest classifier using python library
model = RandomForestClassifier()
model.fit(x_train, y_train)

# print out accuracy of the model based on test samples
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# save trained model in model.p for the application
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
