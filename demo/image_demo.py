# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import json
import os
import cv2

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    ## run inference on all images in the input folder
    images = []
    result =[]
    # for filename in os.listdir("~/mmdetection/demo/input_images/"):
    folder = "demo/input_images/"
    # i = 0
    for i in range(len(os.listdir(folder)) + 30):
        for filename in os.listdir(folder):
            check_string = "-" + f"{i:03}"
            if check_string in filename:
                print(filename)
                print(os.path.join(folder,filename))
                img = os.path.join(folder,filename)
                if img is not None:
                    images.append(img)

                ## run inference on the model 
                single_result = inference_detector(model, img)
                result.append(single_result)

    ## this for loop is to create bbox files for each inference individually
    for i in range (len(result)): 
        bbox_filename = "bboxes_folder/" + str(i) + "_bboxes_for_mask.txt"
        clean_filename = "clean_bboxes_folder/" + str(i) + "_clean_bboxes_for_mask.txt"
        file = open(bbox_filename, "w+")

        ## iterate over all images and only write the content of the 0th category
        ## which is the person class 
        
        for j in range(len(result[i][0])):
            if result[i][0][j][4] > 0.4:
                content = str(result[i][0][j])
                file.write("".join(content) + "\n") 
        file.close()
        
        ## clean up the bbox files such it can be interpreted downstream
        with open(bbox_filename, 'r') as infile, \
             open(clean_filename, 'w') as outfile:
                data = infile.read()
                data = data.replace("[", "")
                data = data.replace("]", "")
                outfile.write(data)

    ## call this method on the entire array of images, rather than a single image
    show_result_pyplot(
        model,
        # args.img,
        images,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
