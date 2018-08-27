
import numpy as np
import os
from sklearn import metrics
import numpy as np
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


num_classes = 2
img_rows, img_cols = 32, 32
os.chdir("/Users/maryameshraghievari/desktop/imgs1")

file = "imgs1_labels.txt"
labels = np.genfromtxt(file)
#print(lbl[0:100])

path = "/Users/maryameshraghievari/desktop/imgs1"
valid_images = [".png"]
names = os.listdir(path)
np.random.seed(0)

data = []

for name in names:
    ext = os.path.splitext(name)[1]
    if ext.lower() not in valid_images:
        continue
    img = Image.open(name)
    '''print(img)'''
    img = img.resize((img_rows, img_cols))
    img = np.array(img)[np.newaxis, :, :, :3]
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    #print(new_shape)
    img = img.reshape(new_shape)
    data.append(img)

data = np.concatenate(data)
n = 90
#n = 900
x_train = data[:n].astype(np.float32)
y_train = labels[:n]
x_test = data[n:].astype(np.float32)
y_test = labels[n:]

'''
    x_train = data[:900].astype(np.float32)
    y_train = labels[:900]
    x_test = data[900:].astype(np.float32)
    y_test = labels[900:]
'''

#x_train = x_t.reshape()
n_f = (len(data[0]) * (len(data)**2))
n_sam = len(data)
#print(n_f)


clf = RandomForestClassifier(bootstrap=True, class_weight=None,
max_depth=2, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
oob_score=False, random_state=0, verbose=0, warm_start=False)

#clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)

print(clf.feature_importances_)
print(clf.predict(x_test))
classification = result.reshape((len(x_test), len(x_test[0])))
      
count = 0
for i in range(0, len(x_test)):
    if(result[i] == y_test[i]):
        count+=1
acc = count/len(x_test)

print(acc)



