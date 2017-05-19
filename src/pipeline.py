#!/usr/bin/env python
# coding=utf-8

from align import align_dataset_mtcnn
import os, sys
import argparse
import time
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from tmp import batch_represent

"""
Allows you to generate embeddings from a directory of images in the format:

Instructions:

Image data directory should look like the following figure:
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png

"""
class arguments(object):
    pass

def gen_path_name(path):
    """
    If path was exist already, rename it and return a new name
    """

    timeStr = str(time.time())
    i = 1
    while os.path.exists(path):
        path = '%s_%s%s' % (path,timeStr, str(i))
        i += 1
    print('Generated path name: ' + path)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory with unaligned images.', default='input')
    parser.add_argument('--output', type=str, help='Directory with aligned face thumbnails.', default='output')
    parser.add_argument('--model', type=str, help='Load a trained model before training starts.', default = '20170512-110547')
    parser.add_argument('--clean', help='Remove temporary folders', action='store_true')
    args = parser.parse_args()
    
    aligner_args = arguments()
    aligner_args.input_dir = args.input
    path = gen_path_name(args.output)
    aligned_path = path + '_aligned'
    aligner_args.output_dir = aligned_path
    aligner_args.image_size = 182
    aligner_args.margin = 44
    aligner_args.random_order = True
    aligner_args.gpu_memory_fraction = 0.3
    faces_count = align_dataset_mtcnn.main(aligner_args)

    if faces_count > 0:
        inference_args = arguments()
        inference_args.data_dir = aligned_path
        inference_args.output_dir = path + '_embeddings'
        inference_args.trained_model_dir = args.model
        inference_args.batch_size = 50
        batch_represent.main(inference_args)

    if args.clean:
        if os.path.exists(aligned_path):
            shutil.rmtree(aligned_path)

if __name__ == '__main__':
    main()
