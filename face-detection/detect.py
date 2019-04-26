# importing OpenCV
import cv2 as cv

# Reading image
img = './face-detect-4.jpg'
# makes image matrix
original_image = cv.imread(img)
# Converting image to grayscale for Voila-Jones algorithm
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
# Selecting classifier
path = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier(path)
# Storing detected faces co-ordinates
detected_faces = face_cascade.detectMultiScale(grayscale_image)
# Displaying rectangles around detected faces
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
# Writting image in new file
img_result = img[:15]+'-result'+img[15:]
cv.imwrite(img_result, original_image)
