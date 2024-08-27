# Multiface_Detection
This project focuses on enhancing face detection in crowded urban environments using an improved MTCNN algorithm. By employing data augmentation, redesigning the detection network with deep separable convolution, and optimizing the Non-Maximum Suppression (NMS) algorithm, the improved method offers faster processing and higher accuracy. 
Here's a README file based on the provided descriptions:

# Face Recognition Application

##Overview

This application is designed to capture and recognize faces using various inputs such as webcams and IP cameras. It includes features for capturing face images, processing images from folders, and performing real-time face recognition. The system also supports multi-face detection and face encoding for training purposes.

## Features

- Capture Faces from Webcam: Initiates the process of capturing face images using the webcam.
- Process Images from Folder: Processes and detects faces from images stored in a specified folder.
- Start Webcam Face Recognition: Begins real-time face recognition using the webcam feed.
- Start IP Camera Face Recognition: Starts face recognition using an IP camera feed.
- Close Application**: Exits the application.

## GUI Overview

The graphical user interface (GUI) is designed to be user-friendly with clear buttons that guide users through the face recognition process. Key visual elements such as logos and illustrations enhance the interface's intuitiveness and professionalism.

### Capturing and Labeling Faces

1. Capture Faces from Webcam: Click this button to start capturing images from the webcam.
2. Prompt for Name: After initiating the capture, a dialog box will appear asking for the name of the person.
3. Save Images with Labels: The provided name will be used as a label for the captured images, which are saved in a training folder named after the person.
4. Additional Actions: Process images from a folder, start face recognition using the webcam or an IP camera, or close the application.

### Capturing and Saving Face Images for Training

- **Image Capturing Process**:
  - Captured faces are highlighted with green rectangles.
  - A label shows the number of images captured so far (e.g., "Image 12 captured").
  - The capture process can be stopped anytime by pressing the "q" key.
  - Images are saved in the training data folder in grayscale format.

### Organization and Storage of Training Data

- Training Data Folder:
  - Location: `C:\Project K\Training_Data`
  - Content: Contains subfolders named after individuals (e.g., Dev, Narsingh, Sathish, Uday).
  - Purpose: Each subfolder holds training images for a specific person.

- Individual's Training Data:
  - Location: `C:\Project K\Training_Data\Narsingh`
  - Content: Contains grayscale images of Narsingh’s face, labeled sequentially (e.g., Narsingh_1, Narsingh_2, etc.).
  - Purpose: These images are used to train the face recognition model for recognizing Narsingh's face.

### Generating Face Encodings

1. Navigate to Project Directory:
   - Change the directory to `C:\Project K`.

2. Execute the Script:
   - Run the `encode_faces.py` script with the following command:
     ```bash
     python encode_faces.py -i "C:\Project K\Training_Data" -e "C:\Project K\encodings.pickle"
     ```

3. Processing Images:
   - The script processes each image, quantifying faces and generating corresponding encodings.
   - Real-time feedback will indicate the processing status (e.g., processing image 1/150).

### Multiface Detection Using Webcam

- The application captures images of multiple individuals’ faces from the webcam.
- Detected faces are highlighted with green rectangles.
- Individuals are labeled based on recognition (e.g., "Sathish" for trained faces and "Unknown" for untrained faces).

