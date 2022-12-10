import numpy as np
import cv2
import os


## get the total number of output images from the detection model
folder = "output_images/"
no_of_files = len(os.listdir(folder))
# print("no of files: ", no_of_files)

## get filename for each unmasked file, ensuring alignment between bounding boxes and unmasked images 
for i in range(no_of_files):
    filename = str(i) + "_unmasked.jpg"  
    clean_bbox_file = "clean_bboxes_folder/" + str(i) + "_clean_bboxes_for_mask.txt"
    File_data = np.loadtxt(clean_bbox_file, dtype=float)
    image = cv2.imread(folder+filename)    
    ## create mask 
    mask = np.zeros(image.shape[:2], dtype="uint8")

    ## use bounding box dimensions to create rectangles that will remain unmasked
    for j in range(len(File_data)):
        cv2.rectangle(mask, (int(File_data[j][0]), int(File_data[j][1])), (int(File_data[j][2]), int(File_data[j][3])), 255, -1)

    ## apply the mask
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    ## create output file, now masked
    output_filename = "masked_outputs/" + str(i) + "_masked_frame.jpg"
    cv2.imwrite(output_filename, masked)
    cv2.waitKey(0)