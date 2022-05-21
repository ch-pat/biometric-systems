import cv2
from datetime import datetime


def face_and_eye_detection():
    """Returns images from video feed and recognizes blinks"""
    # Face and eye cascade classifiers from xml files
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    first_read = True
    eye_detected = False
    blink_detected = False
    # Video Capturing by using webcam
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    start = datetime.now()

    video_images = []
    while ret:
        # this will keep the web-cam running and capturing the image for every loop
        ret, image = cap.read()

        return_image = image.copy()
        video_images += [return_image]
        # Convert the rgb image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying bilateral filters to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)
        # to detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (1, 190, 200), 2)
                # face detector
                roi_face = gray[y:y + h, x:x + w]
                # image
                roi_face_clr = image[y:y + h, x:x + w]
                # to detect eyes
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_face_clr, (ex, ey), (ex + ew, ey + eh), (255, 153, 255), 2)
                    if len(eyes) >= 2:
                        if first_read:
                            cv2.putText(image, "Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 0, 0), 2)
                            eye_detected = True
                        else:
                            cv2.putText(image, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 255, 255), 2)
                    else:
                        if first_read:
                            cv2.putText(image, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 0, 255), 2)
                        else:
                            cv2.putText(image, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (0, 0, 0), 2)
                            print("Blink Detected.....!!!!")
                            blink_detected = True

        else:
            cv2.putText(image, "No Face Detected.", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 255, 255), 2)
        cv2.imshow('Blink', image)
        cv2.waitKey(1)

        end = datetime.now() - start

        if end.seconds > 5 and eye_detected:
            first_read = False
        if end.seconds > 9 and blink_detected:
            break

    # release the web-cam
    cap.release()
    # close the window
    cv2.destroyAllWindows()

    if not blink_detected:
        return "No face or blink detected"

    return video_images, blink_detected

