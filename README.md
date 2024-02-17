# Video-Based Face Recognition and Video Creation

This project involves the extraction of individual faces from a given video using various facial recognition techniques and creating separate videos for each person detected. The process includes face detection, alignment checks, confidence filtering, similarity removal, and clustering.

## Overview

1. **Face Detection:**
   - Utilizes the `mediapipe` library for initial face detection in each frame of the input video.
   - Employs `dlib` and its frontal face detector to verify proper alignment of eyes and mouth.

2. **Face Cropping:**
   - If alignment is proper and the model's confidence is greater than 90%, crops the face with additional padding on all sides.
   - Stores the cropped faces in a designated folder.
   - above things has been done in file names proper_aligned_images_cropping.py

3. **Confidence Filtering and Similarity Removal:**
   - Uses `DeepFace.represent` to calculate model confidence, bounding box, and facial area values for each stored face image.
   - Retains images with a confidence level greater than 60%.
   - this can be found in the file named vector_embedding_finding.py
   - Utilizes cosine metrics for comparing face vectors to remove similar images with a similarity threshold of 70%.
   - this can be found in the file embedding_comparison.py

4. **Reference Image Selection and Clustering:**
   - Creates clusters of unique faces using k-means clustering.
   - Selects one image from each cluster as a reference image and places them in a folder named "reference images."
   - this can be found in file named image_vector_clustering.py

5. **Face Recognition and Video Creation:**
   - Generates sample encodings for each reference image.
   - Compares face encodings using `face_recognition.compare_faces` to identify individuals in each frame.
   - Creates separate videos for each person found in the input video.
   - this can be found in the file multiple_person_detection.py
## Dependencies

- mediapipe
- dlib
- DeepFace
- face_recognition
- Other necessary dependencies (see `requirements.txt`)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/face-recognition-video-creation.git
cd face-recognition-video-creation



Install dependencies:
pip install -r requirements.txt


Usage
Place the input video file in the project directory.

The process to run the project.

1st run proper_aligned_images_cropping.py
2nd run vector_embedding_finding.py
3rd run embedding_comparison.py
4th run image_vector_clustering.py
5th run multiple_person_detection.py

The output videos for each identified person will be generated in the project directory.

Acknowledgments
mediapipe: https://github.com/google/mediapipe
dlib: http://dlib.net/
DeepFace: https://github.com/serengil/deepface
face_recognition: https://github.com/ageitgey/face_recognition


This project is licensed under the MIT License - see the LICENSE file for details.
