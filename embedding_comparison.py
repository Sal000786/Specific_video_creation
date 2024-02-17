# import os
# from deepface import DeepFace

# image_directory = "F:\\Salman codes\\all_output_faces"
# image_files = []
# new_embedding = []


# for filename in os.listdir(image_directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(image_directory, filename)
#         image_files.append(image_path)

# # Iterate through the collected image files
# for image_path in image_files:
#     try:
#         # Use DeepFace to generate face vector
#         face_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False,detector_backend='skip', normalization='base')

#     except Exception as e:
#         print(f"Error processing {image_path}: {str(e)}")

#     for result in face_vector:
#         # Extract information from the dictionary
#         embedding_vector = result["embedding"]
#         # print(embedding_vector)
#         print(type(embedding_vector))

        # if round(face_confidence, 2) < 6.0:
        #     try:
        #         os.remove(image_path)
        #         print(f"The file '{image_path}' has been successfully removed.")
        #     except FileNotFoundError:
        #         print(f"The file '{image_path}' does not exist.")
        #     except Exception as e:
        #         print(f"An error occurred: {e}")
        # if round(face_confidence, 3) > 6.0:
        #     confidence_score.append(face_confidence)
        #     new_embedding.append(embedding_vector)




import os
from deepface import DeepFace
from scipy.spatial.distance import cosine

image_directory = "F:\\Salman codes\\trail_all_output_faces"
image_files = []

# Load all image vectors
image_vectors = {}

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)
        try:
            # Use DeepFace to generate face vector
            face_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False, detector_backend='skip', normalization='base')[0]['embedding']
            image_vectors[filename] = face_vector
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

# Check cosine similarity and delete similar images
for filename1, vector1 in image_vectors.items():
    for filename2, vector2 in image_vectors.items():
        if filename1 != filename2:  # Avoid comparing the same image
            similarity = 1 - cosine(vector1, vector2)
            if similarity > 0.70:  # Adjust the similarity threshold as needed
                print(f"Similarity between {filename1} and {filename2}: {similarity}")
                # Delete one of the images (you may want to keep the original and delete the duplicate)
                os.remove(os.path.join(image_directory, filename2))
                # filename1=filename2
                print(f"{filename2} has been deleted.")
