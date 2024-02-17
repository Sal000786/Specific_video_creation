import cv2
import mediapipe as mp
import time
import math
import os
import dlib



video_path="F:\\Salman codes\\Open_CV_Course\\Frame_recording\\me_and_zunnu.mp4"
cap=cv2.VideoCapture(video_path)
pTime=0.0

#loading mediapipe face_detection for detection of faces 

mpFaceDetection=mp.solutions.face_detection
mp_draw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.75)

# Load the pre-trained face detector and shape predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("F:\\Salman codes\\Open_CV_Course\\Frame_recording\\shape_predictor_68_face_landmarks.dat")


# Create a folder to save cropped faces
output_folder = "F:\\Salman codes\\trail_all_output_faces"
os.makedirs(output_folder, exist_ok=True)

stored_image_sizes={}
details_of_fram=[]

counter=1
while True:
    ret,img=cap.read()
    if not ret:
        print("empty frame received")
        break
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC=detection.location_data.relative_bounding_box
            x_start, y_start = int(bboxC.xmin * img.shape[1]), int(bboxC.ymin * img.shape[0])
            x_end, y_end = int((bboxC.xmin + bboxC.width) * img.shape[1]), int((bboxC.ymin + bboxC.height) * img.shape[0])
            co_ordinates=x_start,y_start,x_end,y_end

        if detection.score[0] > 0.9:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = detector(gray)

            for face in faces:
                # Predict facial landmarks
                landmarks = predictor(gray, face)

                # Extract specific landmark points for lips and eyes
                left_eye = landmarks.part(37).x, landmarks.part(37).y
                right_eye = landmarks.part(46).x, landmarks.part(46).y
                mouth_left = landmarks.part(48).x, landmarks.part(48).y
                mouth_right = landmarks.part(54).x, landmarks.part(54).y

                # Calculate distances or ratios to determine alignment and openness
                
                lip_distance = abs(mouth_right[0] - mouth_left[0])
                # print("lip_distance",lip_distance)

                # Example: Check if the eyes are open based on the aspect ratio
                eye_width = abs(right_eye[0] - left_eye[0])
                eye_height = abs((right_eye[1] + left_eye[1]) / 2 - landmarks.part(24).y)
                if eye_height and eye_width >0:
                    eye_aspect_ratio = eye_width / eye_height
                else:
                    print("eyes height and width are zero")

                # Set thresholds and conditions based on your specific requirements
                threshold_lips=80
                threshold_eyes=1.8
                if lip_distance < threshold_lips and eye_aspect_ratio > threshold_eyes:
                

                    margin = 80
                    x_start -= margin +10
                    y_start -= margin+10
                    x_end += margin
                    y_end += margin

                    # Ensure the coordinates are within the frame boundaries
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = min(img.shape[1], x_end)
                    y_end = min(img.shape[0], y_end)

                    cropped_face = img[y_start:y_end, x_start:x_end]

                    if cropped_face.shape > (250, 150):

                        face_filename = f"face_{counter}.jpg"
                        face_filepath = os.path.join(output_folder, face_filename)
                        cv2.imwrite(face_filepath, cropped_face)
                        counter +=1
                        print("put an image\n")
                    else:
                        pass
                else:
                    print("lip alignment is not good.")
        else:
            print("detection score is less than 90 percent so ignore")
    cv2.imshow("video",img)
    key=cv2.waitKey(1)
    if key==27 & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



