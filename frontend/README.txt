🌿 Plant Disease Detection Frontend (React + Vite)

--------------------------------------------------
📌 Project Overview
--------------------------------------------------
This frontend is developed using React (Vite) to interact with the backend API.

It allows users to:
- Upload a plant leaf image
- Get disease prediction
- View confidence score
- See suggested treatment

The UI is designed with a modern dark theme and interactive elements.

--------------------------------------------------
⚙️ Requirements
--------------------------------------------------

Make sure you have installed:

- Node.js (v16 or above)
- npm (Node Package Manager)

Check versions:
node -v
npm -v

--------------------------------------------------
📦 Install Dependencies
--------------------------------------------------

Open terminal in frontend folder and run:

npm install

If axios is not installed, run:

npm install axios

--------------------------------------------------
📁 Project Structure
--------------------------------------------------

plant-frontend/
│
├── src/
│   ├── App.jsx         # Main UI logic
│   ├── App.css         # Styling (dark + RGB UI)
│   ├── main.jsx        # Entry point
│   └── assets/         # Images (optional)
│
├── index.html
├── package.json
└── README.txt

--------------------------------------------------
🚀 How to Run Frontend
--------------------------------------------------

Step 1: Open terminal in frontend folder

Step 2: Start development server

npm run dev

Step 3: Open browser

http://localhost:5173

--------------------------------------------------
🔗 Backend Connection
--------------------------------------------------

Make sure backend is running at:

http://127.0.0.1:8000

API endpoint used:

POST /predict

If backend is not running → frontend will not work

--------------------------------------------------
📤 How to Use
--------------------------------------------------

1. Click "Choose File"
2. Select a plant leaf image
3. Click "Upload & Predict"
4. View results:
   - Disease message
   - Confidence score
   - Suggested solution

--------------------------------------------------
📸 Image Guidelines
--------------------------------------------------

For best results:
✔ Use clear, high-quality images  
✔ Focus on single leaf  
✔ Use plain background  
✔ Ensure good lighting  
✔ Avoid blur  

--------------------------------------------------
🌱 Supported Plants
--------------------------------------------------

The model supports plants such as:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

⚠️ Other plants are under development

--------------------------------------------------
🐛 Common Issues & Fix
--------------------------------------------------

1. Error: API not working
Fix:
Make sure backend is running

2. CORS issue
Fix:
Ensure backend allows frontend origin

3. Image not uploading
Fix:
Check file type (JPG/PNG)

--------------------------------------------------
🎯 Features
--------------------------------------------------

- Modern dark UI
- Image preview before upload
- Animated plant cards
- RGB hover effects
- Real-time prediction display

--------------------------------------------------
🚀 Future Improvements
--------------------------------------------------

- Add camera capture
- Deploy frontend online
- Add multi-language support
- Improve UI animations

--------------------------------------------------
👨‍💻 Developed By Muhammad Mozzam
--------------------------------------------------

Plant Disease Detection System
Frontend (React + Vite)