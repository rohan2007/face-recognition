import os
import shutil
import face_recognition
import numpy as np

# Define input and output folder paths
input_folder = "input"
output_folder = "output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a dictionary to keep track of detected faces
face_dict = {}
counter = 1  # Initialize a counter for naming

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for .jpg, .jpeg, or .png
        # Load the current image
        image_path = os.path.join(input_folder, filename)
        image = face_recognition.load_image_file(image_path)

        # Find face encodings in the current image
        face_encodings = face_recognition.face_encodings(image)

        # Skip images with no or multiple faces
        if len(face_encodings) != 1:
            print(f"Skipping {filename} due to no/multiple faces detected.")
            continue

        # Calculate the face encoding
        face_encoding = face_encodings[0]

        # Convert the face encoding (NumPy array) to a tuple
        face_encoding_tuple = tuple(face_encoding)

        # Check if the same face already exists in the face_dict
        for existing_encoding, person_name in face_dict.items():
            match = face_recognition.compare_faces([np.array(existing_encoding)], np.array(face_encoding_tuple))
            if match[0]:
                print(f"Found a duplicate of {person_name}: {filename}")
                break
        else:
            # If no match is found, add the new face encoding to the dictionary
            person_name = f"person{counter}"
            face_dict[face_encoding_tuple] = person_name

            # Copy the image to the output folder with the new name
            output_filename = f"{person_name}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            shutil.copy(image_path, output_path)
            print(f"Copying {filename} to {output_path}")

            # Increment the counter
            counter += 1

print("Finished processing images.")
