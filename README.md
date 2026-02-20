# 🚀 Real-Time Conveyor Belt Defect Detection 

## 📌 Project Overview

This project presents a real-time AI-based conveyor belt health monitoring system designed to detect surface defects such as:

- Crack  
- Hole / Tear  

The system uses a custom-trained YOLOv8 object detection model integrated with OpenCV for real-time inference through webcam, image, or video input.

This project demonstrates the complete lifecycle of a computer vision pipeline — from dataset creation and model training to deployment and real-time testing.

---

## 🧠 Problem Statement

Conveyor belts operate continuously in industrial environments and are prone to wear and tear due to mechanical stress and environmental exposure.

Manual inspection:
- Is slow  
- Is error-prone  
- May cause unexpected breakdowns  
- Increases safety risks  

This system aims to provide an automated, real-time defect detection solution.

---

## ⚙️ System Architecture

1. Data Collection & Annotation  
2. YOLOv8 Model Training (Google Colab)  
3. Integration with OpenCV  
4. Real-Time Inference  
5. Defect Visualization with Bounding Boxes  

---

## 🛠️ Tech Stack

- Python 3
- Ultralytics YOLOv8
- OpenCV
- Roboflow (Annotation)
- Google Colab (Training)

---

## 📂 Project Structure

```
conveyor-belt-defect-detection/
│
├── models/
│   └── best.pt
│
├── src/
│   └── detector.py
│
├── test_images/
│   └── test.jpg
│
├── outputs/
│   └── result_image.jpg
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 2️⃣ Run Image Detection

```
python src/detector.py --image test_images/test.jpg
```

---

### 3️⃣ Run Video Detection

```
python src/detector.py --video path_to_video.mp4
```

---

### 4️⃣ Run Webcam Detection

```
python src/detector.py
```

---

## 🎯 Model Details

- Model: YOLOv8 custom trained  
- Classes: Crack, Hole-Tear  
- Inference Mode: Real-time  
- Adjustable Confidence Threshold supported  

---

## ⚠️ Dataset Limitations

This model was trained using a very small dataset (~10 annotated images).

Due to this limitation:

- Model confidence scores are relatively low  
- Generalization to new lighting conditions may be inconsistent  
- The system should be considered a proof-of-concept prototype  

The primary objective of this project was to demonstrate:

- End-to-end AI model training  
- Real-time deployment  
- System integration  

rather than production-level accuracy.

---

## 📊 Current Performance

- Successfully detects visible crack and hole defects  
- Real-time detection supported  
- Adjustable confidence threshold allows tuning detection sensitivity  

---

## 🔮 Future Improvements

- Expand dataset to 100+ images per class  
- Apply advanced data augmentation  
- Increase training epochs  
- Implement defect logging to CSV  
- Add alert system for high-confidence detections  
- Deploy as web-based monitoring dashboard  
- Optimize model for edge deployment  

---

## 📚 Key Learnings

- Importance of dataset size and diversity  
- Handling file path issues and debugging  
- Training and deploying YOLO models  
- Integrating AI models with real-time video streams  
- Practical understanding of computer vision workflows  

---

## 🏁 Conclusion

This project demonstrates the feasibility of using AI-driven computer vision for industrial surface defect detection.

While currently a prototype due to dataset limitations, the architecture is scalable and can be extended into a full production-ready conveyor monitoring solution with additional data and optimization.
