# TeleICU Patient Monitoring System
Welcome to the **TeleICU Patient Monitoring System** repository! Developed by our **Team Tensor Stars**, this project leverages cutting-edge computer vision and machine learning technologies to enhance patient safety and streamline monitoring processes in ICU environments.

## 📄 Introduction
This project aims to develop a comprehensive system for motion detection and object detection in ICU videos. The primary objectives are:
- Detect the presence of various objects.
- Identify motion patterns, particularly focusing on scenarios where the patient is either alone or accompanied by family members.

The system integrates YOLOv8s for object detection and an LSTM model for motion detection, providing a robust solution for ICU monitoring.

## 🗂 Table of Contents
- [Introduction](#-introduction)
- [Requirements](#-requirements)
- [Demo Video](#-demo-video)
- [Preprocessing](#-preprocessing)
- [Features](#-features)
- [Object Detection](#-object-detection)
- [Motion Detection](#-motion-detection)
- [API Integration](#-api-integration)
- [Output](#-output)
- [Project Report](#-project-report)
- [Conclusion](#-conclusion)
- [Get Started](#-get-started)

## 📦 Requirements
1. [Ultralytics](https://docs.ultralytics.com/)
2. [OpenCV](https://docs.opencv.org/4.x/)
3. [Numpy](https://numpy.org/doc/)
4. [Tensorflow](https://www.tensorflow.org/api_docs)
5. [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
6. [postman](https://learning.postman.com/docs/introduction/overview/)

## 🎥 Demo Video
![Demo Video](https://github.com/AritriPodde2210/TeleICU-Patient-Monitoring-System/assets/123970201/4fd1526b-b4e5-46de-901f-51a98cf9818f)

## 🛠 Preprocessing
![Preprocessing](https://github.com/AritriPodde2210/TeleICU-Patient-Monitoring-System/assets/123970201/2a31e784-deb7-411c-a15a-beb4e1d2e474)

## 🎯 Features

### 🖼 Object Detection
**Initialising Requirements**
- High-performance computer with a powerful GPU.
- Essential software: Python, OpenCV, YOLOv8, and other libraries for image and video processing.
- Robust storage solutions for managing large datasets.

**Data Preparation**
- Identified and preprocessed a diverse set of YouTube videos featuring doctors, ICU patients, staff, medical equipment, family members, and ECG monitors.
- Edited videos to remove inappropriate content and converted them into individual frames.

**Annotation and Training**
- Annotated images using Roboflow, creating bounding boxes around objects of interest.
- Split dataset into training (70%), validation (20%), and testing (10%).
- Created a data.yaml file for organized data access.
- Trained YOLOv8s model, achieving 80% accuracy in detecting and classifying objects in the ICU environment.

### 🏃 Motion Detection
**Bounding Boxes and Motion Detection**
- Detected motion by comparing differences between alternate frames using bounding boxes.
- Implemented proximity checks to identify critical scenarios where the patient’s condition might require immediate attention.

**Model Accuracy**
- Integrated an LSTM network to handle sequential data and capture temporal dependencies, improving motion detection accuracy.

**Output**
- The model outputs the top 10 frames where motion is most likely to be detected, focusing on critical moments.

## 🖥️ API Integration
**Combining Models**
- Combined object detection and motion detection models into a single API for simultaneous processing.

**API Functionalities**
- **Index**: Displays a welcome screen.
- **Upload**: Allows direct video upload and outputs the results.
- **Process**: Accessible using POSTMAN software with a JSON request, returning detected frames and the output video.

**Server Access**
- API accessible via the local host server at `http://127.0.0.1:9000`.
 ## 📬 Access API Using Postman Software
To access the API using Postman, follow these steps:
### Using json

Install Postman:

If you haven't already, download and install Postman from [here](https://www.postman.com/downloads/).
Create a New Request:

Open Postman and click on New to create a new HTTP request.
Set the Request Type and URL:

Select POST as the request type.
Enter the API endpoint URL: http://127.0.0.1:9000/process.
Set Headers:

Click on the Headers tab and add the following key-value pair:
Content-Type: application/json
Set the Body:

Click on the Body tab and select raw and JSON (application/json).
Enter the JSON payload with the video data you want to process. For example:

Copy code
{
  "video_path": "path/to/your/video.mp4"
}
Send the Request:

Click on Send to submit your request to the API.
You should receive a response with the processed frames and video output.

For more details on using Postman, refer to the [Postman Documentation](https://learning.postman.com/docs/introduction/overview/).



## 📊 Output
![Output](https://github.com/AritriPodde2210/TeleICU-Patient-Monitoring-System/assets/123970201/819a41cb-b4d0-45fe-9d5a-ea6ebfd87f3e)

## 📝 Project Report
[Project_Report.pdf](https://github.com/user-attachments/files/16093663/Project_Report.pdf)

## 🌟 Team Members and Contribution

Meet the individuals behind **Team Tensor Stars** who contributed to this project:

- **Md Alsaifi** - Team Lead,Video Collection and Preprocessing
- **Aman Kumar Srivastava** - Object detection,Motion detection,API integration
- **Aritri Podder** - Documentation , Report Writing and Research





## 🏁 Conclusion
This project outlines the steps taken to develop and integrate a motion detection and object detection system for ICU videos. The combination of YOLOv8s and LSTM models has provided a robust solution for the project's objectives. The API integration further enhances usability, making it easier to deploy and utilize the system in real-world scenarios.

Future work will focus on improving the models’ accuracy, expanding the dataset, and exploring additional functionalities to enhance the system’s capabilities.

## 🚀 Get Started

**Install Dependencies and Run the API:**

```sh
pip install -r requirements.txt
python api.py

## Access the API:
Open your browser and go to http://127.0.0.1:9000.


Thank you for checking out our project! If you have any questions or feedback, feel free to reach out to us.

