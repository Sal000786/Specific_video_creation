import cv2
import face_recognition
import os

# Dictionary to store frames for each person
person_frames = {}
directory_path = "F:\\Salman codes\\trail_all_output_faces"
image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]

# Full paths to the images
sample_image_paths = [os.path.join(directory_path, f) for f in image_files]
print(sample_image_paths)

# Load and encode faces for each sample image
sample_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in sample_image_paths]

# Initialize video capture
video_path = "F:\\Salman codes\\Open_CV_Course\\Frame_recording\\me_and_zunnu.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        print('salman')
        break

    # Face detection using face_recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        for j, sample_encoding in enumerate(sample_encodings):
            match = face_recognition.compare_faces([sample_encoding], face_encoding)

            if match[0]:
                person_name = f"Sample_Person_{j+1}"

                if person_name not in person_frames:
                    person_frames[person_name] = []

                person_frames[person_name].append(frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save each person's frames as a separate video
for person_name, frames in person_frames.items():
    output_path = f"{person_name}_me_and_zunnu_feb24.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
