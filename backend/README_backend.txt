🌿 Plant Disease Detection Backend (FastAPI + PyTorch)

--------------------------------------------------
📌 Project Overview
--------------------------------------------------
This backend API is developed to detect plant diseases using a trained deep learning model (ResNet18).  
It takes an image of a plant leaf and returns:

- Disease prediction
- Confidence score
- Suggested solution

--------------------------------------------------
⚙️ Requirements
--------------------------------------------------

Make sure you have installed:

- Python 3.9 or above
- pip (Python package manager)

--------------------------------------------------
📦 Install Required Libraries
--------------------------------------------------

Run the following command:

pip install fastapi uvicorn torch torchvision pillow python-multipart

(Optional but recommended)
pip install numpy

--------------------------------------------------
📁 Project Structure
--------------------------------------------------

leaf/
│
├── api.py                     # FastAPI backend
├── train_model.py             # Training script
├── plant_disease_model.pth    # Trained model
├── PlantVillage/              # Dataset (train + val)
│
└── README.txt

--------------------------------------------------
🚀 How to Run the API
--------------------------------------------------

Step 1: Open terminal in project folder

Step 2: Run server

uvicorn api:app --reload

Step 3: Open browser

http://127.0.0.1:8000/docs

👉 This will open Swagger UI (interactive API testing)

--------------------------------------------------
📤 API Endpoint
--------------------------------------------------

POST /predict

Input:
- Form-data
- Key: file
- Value: Upload image (JPG/PNG)

Output (JSON):

{
  "message": "Your tomato plant has early blight.",
  "confidence": 97.5,
  "solution": "Use fungicide and remove infected leaves."
}

--------------------------------------------------
🧠 Model Details
--------------------------------------------------

- Architecture: ResNet18
- Framework: PyTorch
- Dataset: PlantVillage Dataset
- Classes: Multiple plant diseases + healthy

--------------------------------------------------
⚠️ Important Notes
--------------------------------------------------

- Works best with:
  ✔ Clear leaf image
  ✔ Plain background
  ✔ Single leaf

- If plant is NOT in dataset:
  → Model may give incorrect prediction

--------------------------------------------------
🐛 Common Errors & Fix
--------------------------------------------------

1. Error: python-multipart missing
Fix:
pip install python-multipart

2. Model not found error
Fix:
Make sure:
plant_disease_model.pth is in same folder

3. Slow performance
Reason:
Running on CPU

--------------------------------------------------
🎯 Future Improvements
--------------------------------------------------

- Add more plant classes
- Improve real-world accuracy
- Deploy API online (Render / AWS)
- Add database for predictions

--------------------------------------------------
👨‍💻 Developed By Muhammad Mozzam
--------------------------------------------------

Plant Disease Detection System
AI + Deep Learning Project