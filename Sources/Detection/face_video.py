# import libraries
import cv2
import face_recognition
import os

# Get a reference to webcam 
video_capture = cv2.VideoCapture("test1.mp4")
path="./frame1"

# Initialize variables
face_locations = []
i=0
dim = (128, 128)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0 :
        for (x,y,w,h) in face_locations:
            test=frame[y:y+h, x:x+w]
            print(test)
            cv2.imwrite(os.path.join(path , 'test'+str(i)+'.png'), frame)
            i = i+1
            print(i)
    # if len(face_locations) > 0 :
    #     i=i+1
    #     print(i)
    #     cv2.imwrite(os.path.join(path , 'test' + str(i) + '.jpg'), frame)


    # Display the results
    # for top, right, bottom, left in face_locations:
    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # # Display the resulting image
    # cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()