# import the necessary packages
import numpy as np
import cv2
import os


# Displaying the contents of the text file
# file = open("bboxes_for_mask.txt", "r")
# # content = file.read()

# Lines = file.readlines()
  
# count = 0
# # Strips the newline character
# for line in Lines:
#     count += 1
#     print("Line{}: {}".format(count, line.strip()))

# file.close()

# i = 0
folder = "output_images/"
no_of_files = len(os.listdir(folder))
print("no of files: ", no_of_files)

# for filename in os.listdir(folder):
for i in range(no_of_files):
    # print(filename)
    filename = str(i) + "_unmasked.jpg"
        
    clean_bbox_file = "clean_bboxes_folder/" + str(i) + "_clean_bboxes_for_mask.txt"
    # File_data = np.loadtxt("clean_bboxes_for_mask.txt", dtype=float)
    File_data = np.loadtxt(clean_bbox_file, dtype=float)

    image = cv2.imread(folder+filename)
    # image = cv2.imread("result1.jpg")
                                                             
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for j in range(len(File_data)):
        cv2.rectangle(mask, (int(File_data[j][0]), int(File_data[j][1])), (int(File_data[j][2]), int(File_data[j][3])), 255, -1)

    masked = cv2.bitwise_and(image, image, mask=mask)
    
    output_filename = "masked_outputs/" + str(i) + "_masked_frame.jpg"
    print(output_filename)
    cv2.imwrite(output_filename, masked)
    # cv2.imwrite("result_mask2.jpg", masked)
    # cv2.imshow("Mask Applied to Image", masked)
    # i += 1
    cv2.waitKey(0)