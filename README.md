# Vision-Image-Retrieval
- Description: This repository contents all my individual labs and project of the Vision Image Retrival subject (Term 3 in the third year of Computer Vision major).

## Individual Labs
- Lab 01: Drawing color histogram chart for each frame of a video (it is a video in RGB color)
- Lab 02: Using ORB and SIFT detector to detect key-points in images in CD dataset.
- Lab 03: Using K-means for clustering images based on their color histogram and also using K-means to build visual worlds in BOW model.
 
## Individual Project
- Requirement: Build a retrieval system that allows users retrive the related top-k images from the input image.
- Dataset: CD and TMBuD Buildings. However; each of them (original version) shortage of neccessary files. I have build a file (preprocess.cpp) to re-build the structure of them to ensure both of dataset could be used. 
- This program is run on CMD. To run it, follow this structure: <program_name> <query_text> <dataset_label> <method> <top-k_images> <are_images_displayed>
  + Because the goal that i want to build the system is can be retrive from a list of input images (instead of each image) thta i need to create a query text to contain all image that i want to retrive (you can edit it).
  + dataset_label: is a mark that notice for program know which dataset it need to be used (CD or TMBuD).
  + method: SIFT, ORB, Histogram or Correlogram
  + are_images_displayed: is a flag (boolean) that allows program show the results on or not.
### Result 
- The accuracy of this program is evaluated by mAP
- Number of query images is 10.
- The table below is the result for CD dataset. You can see the result of the others dataset in my report.

#### CD Comparison Table

| K  | SIFT Accuracy | SIFT Time/Image | ORB Accuracy | ORB Time/Image | Histogram Accuracy | Histogram Time/Image | Correlogram Accuracy | Correlogram Time/Image |
|----|---------------|-----------------|--------------|----------------|--------------------|----------------------|----------------------|------------------------|
| 3  | 0.791667      | 1.2 secs/image  | 0.633333     | 0.7 secs/image | 0                  | 0 secs/image         | 0                    | 4.3 secs/image         |
| 5  | 0.791667      | 1.2 secs/image  | 0.607222     | 0.9 secs/image | 0                  | 0.1 secs/image       | 0                    | 4.6 secs/image         |
| 11 | 0.79246       | 1.9 secs/image  | 0.60373      | 0.8 secs/image | 0.0358766          | 0 secs/image         | 0.0785714            | 4.6 secs/image         |
| 21 | 0.700148      | 1.6 secs/image  | 0.54342      | 0.7 secs/image | 0.0358766          | 0.1 secs/image       | 0.0921269            | 5.5 secs/image         |

- The result is showed as the image below: 
![image](https://github.com/user-attachments/assets/13bd1b7c-b7c5-4889-b601-6a04b3ec452e)

