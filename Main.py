import cv2
import os
import dlib
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import simpledialog, filedialog
from imutils.video import VideoStream
import imutils
import pickle
import time
import face_recognition
from PIL import Image, ImageTk
import threading

# Load the face encodings from the specified file
data = pickle.loads(open('C:\\Multiface_Detection\\encodings.pickle', "rb").read())

# Use Dlib's face detector
detector = dlib.get_frontal_face_detector()

class Test():
    def __init__(self):
        # Initialize the main window
        self.root = Tk()
        self.root.title('Webcam Face Recognition')
        self.root.geometry('800x500')
        self.stop_event = threading.Event()

        # Load the background image
        try:
            self.background_image = Image.open('C:/background.png')  # Adjust path as needed
            self.background_image = ImageTk.PhotoImage(self.background_image)
        except Exception as e:
            print(f"Error loading background image: {e}")
            self.background_image = None

        # Create a Label to display the background image
        if self.background_image:
            self.background_label = Label(self.root, image=self.background_image)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Add buttons to the GUI with specific styles
        self.button_font = ('Helvetica', '12', 'italic')
        self.button_bg = '#8A2BE2'
        self.button_fg = 'white'

        # Create buttons with commands
        self.capture_button = Button(self.root, text="Capture Faces from Webcam", width=30, font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.capture_faces_webcam)
        self.process_button = Button(self.root, text="Process Images from Folder", width=30, font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.process_images_folder)
        self.recognize_button = Button(self.root, text="Start Webcam Face Recognition", width=30, font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.start_live_cam_thread)
        self.ip_camera_button = Button(self.root, text="Start IP Camera Face Recognition", width=30, font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.start_live_cam_ip_thread)
        self.close_button = Button(self.root, text="Close Application", width=30, font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.quit)

        # Place buttons initially
        self.update_button_positions()

        # Bind the configure event to update button positions when the window is resized
        self.root.bind('<Configure>', self.update_button_positions)

        self.root.mainloop()

    def update_button_positions(self, event=None):
        """ Update button positions dynamically based on window size. """
        button_width = 250
        button_height = 40
        right_margin = int(5 * self.root.winfo_fpixels('1c'))  # Convert cm to pixels
        button_x = self.root.winfo_width() - right_margin - button_width  # Adjust for button width
        vertical_spacing = 70  # Space between buttons

        # Set the position of each button on the GUI
        self.capture_button.place(x=button_x, y=130)
        self.process_button.place(x=button_x, y=130 + vertical_spacing)
        self.recognize_button.place(x=button_x, y=130 + 2 * vertical_spacing)
        self.ip_camera_button.place(x=button_x, y=130 + 3 * vertical_spacing)
        self.close_button.place(x=button_x, y=130 + 4 * vertical_spacing)

    def start_live_cam_thread(self):
        """ Start a thread to capture live video from the webcam. """
        self.stop_event.clear()
        self.live_cam_thread = threading.Thread(target=self.live_cam)
        self.live_cam_thread.start()

    def start_live_cam_ip_thread(self):
        """ Start a thread to capture live video from an IP camera. """
        self.stop_event.clear()
        self.live_cam_ip_thread = threading.Thread(target=self.live_cam_ip)
        self.live_cam_ip_thread.start()

    def live_cam(self):
        """ Capture and recognize faces from the webcam. """
        vs = VideoStream(src=0).start()  # Start the webcam video stream
        writer = None
        time.sleep(2.0)  # Allow the camera sensor to warm up
        while not self.stop_event.is_set():
            frame = vs.read()  # Capture the current frame
            frame = imutils.resize(frame, width=640, height=480)  # Resize frame to 640x480
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            # Detect face locations and face encodings
            boxes = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # Match each face to known faces
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)

            # Draw rectangles and names around detected faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow("Press 'q' to Quit Window", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # If 'q' key is pressed, stop the video stream
                self.stop_event.set()
                break

        vs.stop()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    def live_cam_ip(self):
        """ Capture and recognize faces from an IP camera. """
        ip_camera_url = simpledialog.askstring("IP Camera URL", "Enter the IP Camera URL:")
        if not ip_camera_url:
            return

        vs = VideoStream(src=ip_camera_url).start()  # Start the IP camera video stream
        writer = None
        time.sleep(2.0)  # Allow the camera sensor to warm up
        while not self.stop_event.is_set():
            frame = vs.read()  # Capture the current frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)  # Resize frame to 750 width
            r = frame.shape[1] / float(rgb.shape[1])

            # Detect face locations and face encodings
            boxes = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # Match each face to known faces
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)

            # Draw rectangles and names around detected faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow("Press 'q' to Quit Window", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # If 'q' key is pressed, stop the video stream
                self.stop_event.set()
                break

        vs.stop()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    def capture_faces_webcam(self):
        """ Capture and save faces from the webcam to a folder. """
        name = simpledialog.askstring("Input", "Enter your name:")
        if not name:
            return

        folder_path = f'C:/Multiface_Detection/{name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        vs = VideoStream(src=0).start()  # Start the webcam video stream
        total = 0
        while True:
            frame = vs.read()  # Capture the current frame
            frame = imutils.resize(frame, width=400)
            cv2.imshow("Face Capture - Press 'k' to Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("k"):  # If 'k' key is pressed, capture the frame
                p = os.path.sep.join([folder_path, "{}.png".format(str(total).zfill(5))])
                cv2.imwrite(p, frame)
                total += 1
            elif key == ord("q"):  # If 'q' key is pressed, exit
                break

        vs.stop()
        cv2.destroyAllWindows()

    def process_images_folder(self):
        """ Process images from a specified folder for face recognition. """
        folder_path = filedialog.askdirectory(title='Select Folder')
        if not folder_path:
            return

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                # Display the image with the recognized name
                for (top, right, bottom, left) in boxes:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow("Recognized Image", image)
                cv2.waitKey(0)

    def quit(self):
        """ Stop any running threads and close the application. """
        self.stop_event.set()
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    Test()
