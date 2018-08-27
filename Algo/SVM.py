################### SVM #############################

from sklearn import datasets, svm, metrics

num_classes = 2
epochs = 50
batch_size = 4
n_filters = 32
n_pool = 2
n_conv = 2
img_rows, img_cols = 32, 32

os.chdir("/Users/maryameshraghievari/desktop/imgs1")
labels = []

file = "imgs1_labels.txt"
lbl = np.genfromtxt(file)
for i in range(0, len(lbl)):
    if lbl[i] == 0 :
        labels.append( -1)
    elif lbl[i] == 1 :
        labels.append( 1)
    else:
        continue

#print(labels)

path = "/Users/maryameshraghievari/desktop/imgs1"
valid_images = [".png"]
names = os.listdir(path)


data = []
for name in names:
    ext = os.path.splitext(name)[1]
    if ext.lower() not in valid_images:
        continue
    img = Image.open(name)
    img = img.resize((img_rows, img_cols))
    img = np.array(img)[np.newaxis, :, :, :3]
    data.append(img)

data = np.concatenate(data)

n = 90
#n = 900
x_train = data[:n].astype(np.float32)
y_train = labels[:n]
x_test = data[n:].astype(np.float32)
y_test = labels[n:]


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

n_samples = len(x_train)
data = x_train.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
% (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
plt.subplot(2, 4, index + 5)
plt.axis('off')
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Prediction: %i' % prediction)

plt.show()
