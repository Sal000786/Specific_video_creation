# from deepface import DeepFace
# from sklearn.cluster import KMeans
# import os
# import numpy as np

# # Step 1: Generate Image Vectors
# image_directory = "F:\\Salman codes\\trail_all_output_faces"
# image_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]
# print("Length of image files:", len(image_files))

# # List to store image vectors
# vectors = []

# # Iterate through the collected image files
# for image_path in image_files:
#     # Use DeepFace to generate face vector
#     image_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False, normalization='base')[0]['embedding']
#     vectors.append(image_vector)

# # Convert the list of vectors to a NumPy array
# image_vectors = np.array(vectors)

# # Print the shape of the NumPy array
# print("Shape of image vectors array:", image_vectors.shape)


# num_clusters = 2  # Adjust based on your preferences
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# clusters = kmeans.fit_predict(image_vectors)

# # Step 3: Quality Assessment
# for cluster_id in range(num_clusters):
#     cluster_images = [image_files[i] for i in range(len(clusters)) if clusters[i] == cluster_id]

#     print(f"Cluster {cluster_id} - Number of Images: {len(cluster_images)}")
#     print(cluster_images)
#     print("\n")



from deepface import DeepFace
from sklearn.cluster import KMeans
import os
import numpy as np
import shutil

# Step 1: Generate Image Vectors
image_directory = "F:\\Salman codes\\trail_all_output_faces"
output_directory = "F:\\Salman codes\\clustered_faces_output"  # Change this to your desired output directory
image_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]
print("Length of image files:", len(image_files))

# List to store image vectors
vectors = []

# Iterate through the collected image files
for image_path in image_files:
    # Use DeepFace to generate face vector
    image_vector = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False, normalization='base')[0]['embedding']
    vectors.append(image_vector)

# Convert the list of vectors to a NumPy array
image_vectors = np.array(vectors)

# Print the shape of the NumPy array
print("Shape of image vectors array:", image_vectors.shape)



num_clusters = 2  # Adjust based on your preferences
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(image_vectors)

# Step 3: Quality Assessment
for cluster_id in range(num_clusters):
    # Create a folder for the cluster
    cluster_folder = os.path.join(output_directory, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)

    # Get the image paths for the current cluster
    cluster_images = [image_files[i] for i in range(len(clusters)) if clusters[i] == cluster_id]

    print(f"Cluster {cluster_id} - Number of Images: {len(cluster_images)}")
    print(cluster_images)

    # Move the images to the cluster folder
    for image_path in cluster_images:
        image_filename = os.path.basename(image_path)
        destination_path = os.path.join(cluster_folder, image_filename)
        shutil.move(image_path, destination_path)

    print(f"Images moved to {cluster_folder}\n")



# output_directory = "F:\\Salman codes\\clustered_faces_output"
# final_output_folder = "F:\\Salman codes\\trail_all_output_faces"

# Create the final output folder if it doesn't exist
os.makedirs(image_directory, exist_ok=True)

for cluster_id in range(num_clusters):
    # Get the cluster folder
    cluster_folder = os.path.join(output_directory, f"cluster_{cluster_id}")

    # Get the list of images in the cluster folder
    cluster_images = [f for f in os.listdir(cluster_folder) if os.path.isfile(os.path.join(cluster_folder, f))]

    # Take only the first image from the cluster
    if cluster_images:
        source_image_path = os.path.join(cluster_folder, cluster_images[0])

        # Create a unique name for the image in the final output folder
        destination_image_path = os.path.join(image_directory, f"cluster_{cluster_id}_image.jpg")

        # Move the image to the final output folder
        shutil.copy(source_image_path, destination_image_path)

print("One image from each cluster moved to 'trail_all_faces' folder.")
