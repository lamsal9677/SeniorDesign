# Face Recognition and Drowsiness Detection

This repository contains a Python script for real-time face recognition and drowsiness detection using OpenCV, dlib, face_recognition, and socket communication. The script captures video from a webcam, recognizes faces, and checks for drowsiness based on the Eye Aspect Ratio (EAR). Detected faces are compared against a set of reference images to identify known individuals.

## Dependencies

Make sure you have the following Python libraries installed:

- OpenCV (`pip install opencv-python`)
- dlib (`pip install dlib`)
- face_recognition (`pip install face-recognition`)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)

## Usage

1. **Clone this repository:**

2. **Install dependencies:**


3. **Download the `shape_predictor_68_face_landmarks.dat` file from [here](https://github.com/davisking/dlib-models) and place it in the project directory.**

4. **Create a directory named "images" and add images of individuals you want to recognize. You can use the capture script provided**

5. **Update the `server_address` variable in the script with the appropriate IP address and port number.**

6. **Run the script:**

    ```bash
    python drowsy.py
    ```

7. **Press 'q' to exit the application.**

## Explanation

- The script initializes a socket client and connects to a server with the specified IP address and port number.

- Facial recognition is performed using the `face_recognition` library, comparing faces against reference images in the "images" directory.

- Drowsiness detection is implemented based on the Eye Aspect Ratio (EAR). If a recognized face exhibits signs of drowsiness, a message is sent to the server.

- The script continuously captures frames from the webcam, processes them, and displays the results. Press 'q' to exit the application.

**Note:** Ensure that the server is set up to receive and handle messages from the client.

Feel free to customize the script based on your specific use case and integrate additional features as needed.
