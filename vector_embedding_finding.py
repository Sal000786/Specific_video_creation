# import os
# from deepface import DeepFace

# image_directory = "F:\\Salman codes\\all_output_faces"
# image_files = []
# confidence_score=[]
# new_embedding=[]

# for filename in os.listdir(image_directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(image_directory, filename)
#         image_files.append(image_path)

# # Iterate through the collected image files
# for image_path in image_files:
#     try:
#         # Use DeepFace to generate face vector
#         face_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False,normalization='base')
 
#     except Exception as e:
#         print(f"Error processing {image_path}: {str(e)}")

#     for result in face_vector:
#         # Extract information from the dictionary
#         embedding_vector = result["embedding"]
#         facial_area = result["facial_area"]
#         face_confidence = result["face_confidence"]

#         print(len(embedding_vector))
#         print(facial_area)
#         print(face_confidence)

#         if round(face_confidence,2) < 0.6:
#             try:
#                 os.remove(image_path)
#                 print(f"The file '{image_path}' has been successfully removed.")
#             except FileNotFoundError:
#                 print(f"The file '{image_path}' does not exist.")
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#         if round(face_confidence,3)>0.6:
#             confidence_score.append(face_confidence)
#             new_embedding.append(embedding_vector)
    



    

# # print(type(embedding_vector))
# # print(type(facial_area))
# print(confidence_score)
# print(len(confidence_score))



import os
from deepface import DeepFace

image_directory = "F:\\Salman codes\\trail_all_output_faces"
image_files = []
confidence_score = []
new_embedding = []

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)
        image_files.append(image_path)

# Iterate through the collected image files
for image_path in image_files:
    try:
        # Use DeepFace to generate face vector
        face_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False, normalization='base')

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

    for result in face_vector:
        # Extract information from the dictionary
        embedding_vector = result["embedding"]
        facial_area = result["facial_area"]
        face_confidence = result["face_confidence"]

        print(len(embedding_vector))
        print(facial_area)
        print(face_confidence)

        if round(face_confidence, 2) < 6.0:
            try:
                os.remove(image_path)
                print(f"The file '{image_path}' has been successfully removed.")
            except FileNotFoundError:
                print(f"The file '{image_path}' does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")
        if round(face_confidence, 3) > 6.0:
            confidence_score.append(face_confidence)
            new_embedding.append(embedding_vector)

print(confidence_score)
print(len(confidence_score))


# print(facial_area["x"],facial_area["y"],facial_area["w"],facial_area["h"])
# print(embedding_vector[0]) # this would result in the value at position 1 in the 2622 vector