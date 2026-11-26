import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("resnet"); // 👈 default model

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", model); // 👈 send model

    try {
      setLoading(true);
      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );
      setResult(res.data);
    } catch (err) {
      alert("Error connecting to API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">

        {/* TITLE */}
        <h1 className="main-title">🌿 Plant Disease Detection</h1>

        {/* INSTRUCTIONS */}
        <div className="glass-card instructions-card">
          <h2>📸 How to Capture a Good Image</h2>
          <ul>
            <li>✔ Use a clear, high-quality image</li>
            <li>✔ Focus on a single leaf</li>
            <li>✔ Plain background (like table or white)</li>
            <li>✔ Good lighting (avoid shadows)</li>
            <li>✔ Capture full leaf (not cropped)</li>
            <li>✔ Avoid blur</li>
          </ul>
        </div>

        {/* UPLOAD */}
        <div className="glass-card upload-card">

          <input type="file" onChange={handleFileChange} />

          {preview && (
            <img src={preview} alt="preview" className="preview" />
          )}

          {/* ✅ MODEL SELECTOR */}
          <div style={{ marginTop: "15px" }}>
            <label style={{ marginRight: "10px", fontWeight: "bold" }}>
              Select Model:
            </label>

            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="model-select"
            >
              <option value="resnet">ResNet18 (Balanced)</option>
              <option value="cnn">CNN (Fast)</option>
              <option value="mobilenet">MobileNetV2 (Best Accuracy)</option>
            </select>
          </div>

          <button onClick={handleUpload}>
            {loading ? "Analyzing..." : "Upload & Predict"}
          </button>

        </div>

        {/* SUPPORTED PLANTS */}
        <div className="glass-card plants-card">
          <h2>🌱 Supported Plants</h2>

          <div className="plant-grid">
            {[
              { name: "Apple", icon: "🍎" },
              { name: "Blueberry", icon: "🫐" },
              { name: "Cherry", icon: "🍒" },
              { name: "Corn", icon: "🌽" },
              { name: "Grape", icon: "🍇" },
              { name: "Orange", icon: "🍊" },
              { name: "Peach", icon: "🍑" },
              { name: "Pepper", icon: "🫑" },
              { name: "Potato", icon: "🥔" },
              { name: "Raspberry", icon: "🍓" },
              { name: "Soybean", icon: "🌿" },
              { name: "Squash", icon: "🎃" },
              { name: "Strawberry", icon: "🍓" },
              { name: "Tomato", icon: "🍅" },
            ].map((plant, i) => (
              <div key={i} className="plant-card">
                <div>{plant.icon}</div>
                <p>{plant.name}</p>
              </div>
            ))}
          </div>

          <p className="note">
            ⚠️ Other plants are currently under development.
          </p>
        </div>

        {/* RESULT */}
        {result && (
          <div className="glass-card result-card">
            <h2>🧠 Result</h2>

            {/* 👇 NEW INFO */}
            <p><strong>Model Used:</strong> {result.model_used}</p>

            <p><strong>Prediction:</strong> {result.prediction}</p>

            <p><strong>Confidence:</strong> {result.confidence}%</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;