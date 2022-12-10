This codebase is based on mmdetection, which includes general implementations for a large number of applications, one of which is autonomous vehicles. 

In order to run this codebase, you will have to follow the installation instructions described in README.md, including the creation of a conda environment.

Also important to note: I had to run many different configurations of GCP before this current configuration was able to run. My GCP configuration is 

n1-standard-1
1 x NVIDIA Tesla K80
Vertex AI Deeplearning VM -- CUDA version 11.0

Then using this configuration, I opened a JupyterLab notebook, and created my conda environment. You may have to modify the GCP configuration to get things working. You may also have to downgrade the CUDA version to <9.0 to get things working. 



To run inference on the input_images folder using yolov3 pretrained on Coco. (if your env is set up correctly, works every time), input this command from terminal from the /mmdetection directory 

python demo/image_demo.py demo/input_images/ yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file output_images/


To mask the output folder -- producing the final outputs "masked_outputs" (if your env is set up correctly, works every time)
python mask_output_image.py 



I have made some modifications to finetune the model on Citypersons:

To run inference on the input_images folder using Faster RCNN pre-trained on COCO, finetuned on Citypersons , input this command from terminal from the /mmdetection directory :

python demo/image_demo.py demo/input_images/ configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --device cpu --out-file output_images/



The main modifications for this pipeline to infer on a set of images, paint the bounding boxes for pedestrian class only, and then mask everythign else:

mask_output_image.py
image_demo.py


