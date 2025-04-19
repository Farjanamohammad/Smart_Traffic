from roboflow import Roboflow

# Initialize Roboflow API
rf = Roboflow(api_key="IKolIv1zXStWOZ8MoRfP")  # Replace with your actual Roboflow API key

# Select the project and version
project = rf.workspace("learning-evidence").project("helmet-detection_yolov8")
version = project.version(3)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

print("✅ Dataset downloaded successfully!")
