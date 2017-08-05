import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                        # OpenCV library for computer vision
from PIL import Image
import time


# Load in color image for face detection
image = cv2.imread('test_image.jpg')

# Convert the image to RGB colorspace
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot our image
fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('image copy')
ax1.imshow(image)


# Load the model
model = load_model('my_model.h5')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('~/haarcascade_frontalface_default.xml')

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray, 1.25, 6)

# Print the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Make a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image)

# For each face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)

    # Add the face
    gray_face = gray[y: y + h, x: x + w]

    # Resize the face
    resized_face = cv2.resize(gray_face, (96, 96))

    # Normalize the face
    normalized_face = (np.vstack(resized_face)/255).astype(np.float32)

    # Run through model and flaten array
    predicted_points= model.predict(normalized_face.reshape(-1, 96, 96, 1)).ravel()

    # Convert coordinates to face
    predicted_points = predicted_points * w/2 + w/2

    # Put in correct place of the picture (position from face plus x and y)
    predicted_points[0::2] = predicted_points[0::2] + x
    predicted_points[1::2] = predicted_points[1::2] + y

    # For each point, overlap it with the original image
    for (x1, y1) in zip(predicted_points[0::2], predicted_points[1::2]):
        cv2.circle(image_with_detections, (x1,y1), 3, (0,188,0), thickness=-1, lineType=8)

# Display the image with the detections
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Image with Face Detections and Face Keypoints')
ax1.imshow(image_with_detections)