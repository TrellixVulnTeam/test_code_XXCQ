#!/usr/bin/python

import argparse
import sys
import os
import numpy
from shutil import copy

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script copies relevant files to target RGB-D folder   
    ''')
    parser.add_argument('groundtruth_file', help='groundtruth text file')
    parser.add_argument('association_file', help='groundtruth text file')
    parser.add_argument('target_folder', help='target RGB-D folder you want to copy images to')
    args = parser.parse_args()
    groundtruth_file = args.groundtruth_file
    association_file = args.association_file
    target_folder = args.target_folder

    # Read groundtruth file
    file = open(groundtruth_file)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    datalist = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    i = 0
    glist = []
    for l in datalist:
        glist.append([l[0], i])
        i = i + 1
    groundtruth_list = dict(glist)

    # Read association file
    file = open(association_file)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    datalist = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    glist = []
    for l in datalist:
        glist.append([l[0], l[2]])
    association_list = dict(glist)

    for d,i in sorted(groundtruth_list.items()):
        depth = './depth/' + d + '.png'
        idx = str(i).zfill(6)
        fname = target_folder + 'frame-' + idx
        print('Copying ' + fname)
        dst = fname + '.depth.png'
        copy(depth, dst)

        rgb = './rgb/' + association_list[d] + '.png'
        dst = fname + '.color.png'
        copy(rgb, dst)

        pose = './rgb/' + str(i) + '.pose.txt'
        dst = fname + '.pose.txt'
        copy(pose, dst)


        

        
