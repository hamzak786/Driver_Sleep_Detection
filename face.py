import cv2
import dlib
import time
from playsound import playsound
import pygame
from imutils import face_utils


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()


eye_closed_counter = 0
ear_threshold = 0.25 
closed_eyes_duration = .3 


def eye_aspect_ratio(eye):

    vertical_dist_1 = abs(eye[1][1] - eye[5][1])
    vertical_dist_2 = abs(eye[2][1] - eye[4][1])

   
    horizontal_dist = abs(eye[0][0] - eye[3][0])


    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return ear


while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    faces = detector(gray)

    for face in faces:
   
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)


        left_eye = shape[36:42]
        right_eye = shape[42:48]

 
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)


        avg_ear = (left_ear + right_ear) / 2.0


        if avg_ear < ear_threshold:
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0

       
        if eye_closed_counter >= closed_eyes_duration * 30:  # Assuming 30 FPS
        
            pygame.mixer.init()
            pygame.mixer.music.load("wakeup.mp3")
            pygame.mixer.music.play()
            pygame.time.wait(2000)
            pygame.mixer.music.stop()
            pygame.quit()

       
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
