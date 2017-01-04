#!/usr/bin/python

import os
import glob
import random

fnames=['data_list_full_dog.txt','data_list_full_golf.txt','data_list_full_goose.txt','data_list_full_lawn.txt','data_list_full_person.txt']

for i in range(0,len(fnames),1):
    fp=open(fnames[i])
    allfiles=fp.read()
    fp.close()
    a=allfiles.replace('\n',' ')
    files=a.split()
    # split into 75%, 15%, 10% and put into data_list_[train,test,val].txt
    ind1=int(round(len(files)*0.75))
    ind2=int(round(len(files)*0.90))

    # randomize/shuffle
    random.shuffle(files)

    # write out the segments into the train/test/val lists in append write mode
    fp=open('data_list_train.txt','aw')
    tmpfiles=files[0:ind1]
    tmpfiles=str(tmpfiles).replace('\', \'','\n')[2:-2]+"\n"
    fp.write(tmpfiles)
    fp.close()

    fp=open('data_list_test.txt','aw')
    tmpfiles=files[ind1:ind2]
    tmpfiles=str(tmpfiles).replace('\', \'','\n')[2:-2]+"\n"
    fp.write(tmpfiles)
    fp.close()

    fp=open('data_list_val.txt','aw')
    tmpfiles=files[ind2:]
    tmpfiles=str(tmpfiles).replace('\', \'','\n')[2:-2]+"\n"
    fp.write(tmpfiles)
    fp.close()

