# TeleICU Patient Monitoring System
 Welcome to the TeleICU Patient Monitoring System repository! Developed by our Team Tensor Stars, this project leverages cutting-edge computer vision and machine learning technologies to enhance patient safety and streamline monitoring processes in ICU environments.

📄 Introduction This project aims to develop a comprehensive system for motion detection and object detection in ICU videos. The primary objectives are:

Detect the presence of various objects. Identify motion patterns, particularly focusing on scenarios where the patient is either alone or accompanied by family members. The system integrates YOLOv8s for object detection and an LSTM model for motion detection, providing a robust solution for ICU monitoring.

🎯 Features 🖼 Object Detection Initialising Requirements High-performance computer with a powerful GPU. Essential software: Python, OpenCV, YOLOv8, and other libraries for image and video processing. Robust storage solutions for managing large datasets. Data Preparation Identified and preprocessed a diverse set of YouTube videos featuring doctors, ICU patients, staff, medical equipment, family members, and ECG monitors. Edited videos to remove inappropriate content and converted them into individual frames. Annotation and Training Annotated images using Roboflow, creating bounding boxes around objects of interest. Split dataset into training (70%), validation (20%), and testing (10%). Created a data.yaml file for organized data access. Trained YOLOv8s model, achieving 80% accuracy in detecting and classifying objects in the ICU environment. 🏃 Motion Detection Bounding Boxes and Motion Detection Detected motion by comparing differences between alternate frames using bounding boxes. Implemented proximity checks to identify critical scenarios where the patient’s condition might require immediate attention. Model Accuracy Integrated an LSTM network to handle sequential data and capture temporal dependencies, improving motion detection accuracy. Output The model outputs the top 10 frames where motion is most likely to be detected, focusing on critical moments. 🖥️ API Integration Combining Models Combined object detection and motion detection models into a single API for simultaneous processing. API Functionalities Index: Displays a welcome screen. Upload: Allows direct video upload and outputs the results. Process: Accessible using POSTMAN software with a JSON request, returning detected frames and the output video. Server Access API accessible via the local host server at http://127.0.0.1:9000. 🏁 Conclusion This project outlines the steps taken to develop and integrate a motion detection and object detection system for ICU videos. The combination of YOLOv8s and LSTM models has provided a robust solution for the project's objectives. The API integration further enhances usability, making it easier to deploy and utilize the system in real-world scenarios.

Future work will focus on improving the models’ accuracy, expanding the dataset, and exploring additional functionalities to enhance the system’s capabilities.

🚀 Get Started

Install Dependencies:

Copy code pip install -r requirements.txt Run the API:

Copy code python api.py Access the API: Open your browser and go to http://127.0.0.1:9000
