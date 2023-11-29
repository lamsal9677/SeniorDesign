import cv2

# Set up the camera (0 or 1 for the camera index)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    cv2.imshow('Press Space to Capture', frame)
    
    key = cv2.waitKey(1)
    
    if key == 32:  # Check for space key press (ASCII for space)
        cv2.imwrite("sl2.jpg", frame)  # Save the captured image as "my_image.jpg"
        break  # Break the loop after capturing the image

cap.release()
cv2.destroyAllWindows()