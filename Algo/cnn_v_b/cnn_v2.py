{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\csgray\c0;\csgray\c100000;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #{\field{\*\fldinst{HYPERLINK "https://www.youtube.com/watch?v=u8BW_fl6WRc"}}{\fldrslt https://www.youtube.com/watch?v=u8BW_fl6WRc}}\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f1\fs22 \cf2 \cb3 \CocoaLigature0 import os\
from PIL import Image\
from PIL import ImageFilter\
import numpy as np\
#import keras\
import pandas as pd\
\
    \
print('hiiiiiiiiiiiiiiiiiiii')\
num_classes = 10\
epochs = 50\
batch_size = 4\
    \
img_rows, img_cols = 32, 32\
\
os.chdir("/afs/cad/u/m/e/me258/cnn/imgs1")\
\
##########read lbls################   \
file = "imgs1_labels.txt"\
labels = np.genfromtxt(file)\
#print(lbl[0:100])  \
\
###########read imgs################\
path = "/afs/cad/u/m/e/me258/cnn/imgs1"\
valid_images = [".png"]\
names = os.listdir(path)\
###################################\
data = []\
for name in names:\
    ext = os.path.splitext(name)[1]\
    if ext.lower() not in valid_images:\
                continue\
    img = Image.open(name)#shape is 128x128x4\
    img = img.resize((img_rows, img_cols))#resize image into fixed siz32x32x4\
    img = np.array(img)[np.newaxis, :, :, :3]#add new axis and new size is 1x32x32x3\
    data.append(img)\
\
data = np.concatenate(data)\
\
x_train = data[:90].astype(np.float32)\
y_train = labels[:90]\
x_test = data[90:].astype(np.float32)\
y_test = labels[90:]\
\
x_train /= 255\
x_test /= 255\
\
#print (x_train)\
\
#y_train = keras.utils.to_categorical(y_train, num_classes)\
#y_test = keras.utils.to_categorical(y_test, num_classes)\
\
}