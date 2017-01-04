#!/usr/bin/python

import os
import glob

filedir=os.path.join('labels/','*.txt')
filelist=glob.glob(filedir)
ftest=open('data_list_test.txt')
alltest=ftest.read()
ftrain=open('data_list_train.txt')
alltrain=ftrain.read()
fval=open('data_list_val.txt')
allval=fval.read()

# check that all the references in the train/test files are actually present
a=alltest.replace('\n',' ')
files=a.split()
a=alltrain.replace('\n',' ')
files.extend(a.split())
a=allval.replace('\n',' ')
files.extend(a.split())
for f in files:
    ret1=os.path.isfile('labels/'+f+'.txt')
    ret2=os.path.isfile('images/'+f+'.jpg')
    if (not (ret1 and ret2)):
        print('reference %s is not in labels or images dirs'%f)

for f in filelist:
    # check if this file is in the train or test txt files
    sf=f[f.find('/')+1:f.find('.txt')]
    res1=alltest.find(sf)
    res2=alltrain.find(sf)
    res3=allval.find(sf)
    if (res1<0 and res2<0 and res3<0):
        print('file %s not found in either list'%sf)
    with open(f) as fd:
        for (i, line) in enumerate(fd):
            tmp = [float(t.strip()) for t in line.split()]
            # class, xcenter%, ycenter%, xwidth%, yheight%
            if (min(tmp[1:])<0.001):
                print('file %s has value too small'%f)
            if (max(tmp[1:])>0.999):
                print('file %s has value too large'%f)
            # check for too large boxes overlapping past the edges
            if (tmp[1]+(tmp[3]/2.0)>1 or tmp[1]-(tmp[3]/2.0)<0):
                print('file %s has box over the edge on line %d'%(f,i))
            if (tmp[2]+(tmp[4]/2.0)>1 or tmp[2]-(tmp[4]/2.0)<0):
                print('file %s has box over the edge on line %d'%(f,i))

