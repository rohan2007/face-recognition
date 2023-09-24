import os
import shutil
import face_recognition
import numpy as np
import cv2

# Define input and output folder paths
input_folder = "input"
output_folder = "output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a list to keep track of detected face encodings
face_encodings_list = []

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for .jpg, .jpeg, or .png
        # Load the current image
        image_path = os.path.join(input_folder, filename)
        image = face_recognition.load_image_file(image_path)

        # Find face encodings and face locations in the current image
        face_encodings = face_recognition.face_encodings(image)
        face_locations = face_recognition.face_locations(image)

        # Skip images with no or multiple faces
        if len(face_encodings) != 1 or len(face_locations) != 1:
            print(f"Skipping {filename} due to no/multiple faces detected.")
            continue

        # Calculate the face encoding
        face_encoding = face_encodings[0]

        # Check if the same face already exists in the face_encodings_list
        match = face_recognition.compare_faces(face_encodings_list, face_encoding)

        if any(match):
            # If a matching face is found, skip the image
            print(f"Found a duplicate face for {filename}")
        else:
            # If no match is found, add the new face encoding to the list
            face_encodings_list.append(face_encoding)

            # Draw a rectangle around the detected face
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add text indicating the person number
            person_number_text = f"Person {len(face_encodings_list)}"
            cv2.putText(image, person_number_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the modified image with the rectangle and text
            output_filename = f"person{len(face_encodings_list)}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Copying {filename} to {output_path}")

print("Finished processing images.")
