import os
import glob
import numpy as np
from PIL import Image
os.chdir("/afs/cad/u/m/e/me258/cnn/imgs1")

##########read lbls################
file = "imgs1_labels.txt"
lbl = np.genfromtxt(file)
print(lbl[0:100])

###########read imgs################
path = "/afs/cad/u/m/e/me258/cnn/imgs1"
valid_images = [".png"]
'''
    for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
    continue
    im = Image.open(os.path.join(path,f))
    '''

'''
    for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    print(im)
    '''

img_list = os.listdir(path)
#print(ls)

for name in img_list:
    img = Image.open(name)#shape is 128x128x4
    img = img.resize((img_rows, img_cols))#resize image into fixed siz32x32x4
    img = np.array(img)[np.newaxis, :, :, :3]#add new axis and new size is 1x32x32x3
    data.append(img)


