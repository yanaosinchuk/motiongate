import cv2 as cv   
import numpy as np  
from matplotlib import pyplot as plt 

def motionDetection():
    """
    Detects motion in a video using frame differencing and contours.

    The function captures frames from a video file, processes consecutive frames
    to detect motion, and draws bounding boxes around detected movements. The process
    includes converting the difference between two frames to grayscale, applying Gaussian blur,
    thresholding the image to create a binary image, and dilating the result to enhance
    the detected motion areas. Detected motion is highlighted with rectangles on the original frame.

    If the user presses the 'ESC' key, the video stops and the program exits.

    The video file is expected to be at './img/vtest.avi'.
    """
    # Capture video from the specified file
    cap = cv.VideoCapture("./img/vtest.avi")

    # Read two consecutive frames to compare for motion detection
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Loop through video frames as long as the video is open
    while cap.isOpened():
        # Compute the absolute difference between two frames
        diff = cv.absdiff(frame1, frame2)

        # Convert the difference to grayscale
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)

        # Apply thresholding to get a binary image from the blurred image
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

        # Dilate the thresholded image to fill in holes and enhance the contours
        dilated = cv.dilate(thresh, None, iterations=3)

        # Find contours of the objects in motion
        contours, _ = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Loop through the detected contours
        for contour in contours:
            # Get the bounding rectangle of each contour
            (x, y, w, h) = cv.boundingRect(contour)

            # Ignore small contours that are likely noise
            if cv.contourArea(contour) < 900:
                continue

            # Draw a rectangle around the detected motion
            cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display text indicating movement status on the frame
            cv.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 3)

        # Display the video frame with the bounding boxes
        cv.imshow("Video", frame1)

        # Move to the next frame for comparison
        frame1 = frame2
        ret, frame2 = cap.read()

        # Break the loop if the 'ESC' key is pressed
        if cv.waitKey(50) == 27:
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    motionDetection()
