```markdown
# Deep Residual Network Video Summarization for Face Detection and Person Re-Identification  

## 📌 Project Overview  
This project processes CCTV footage from multiple cameras to **detect and track a target person** using **YOLO, ResNet, and DeepSORT**.  
The system automatically summarizes the video by:  
- Detecting faces and identifying the main person.  
- Tracking the person across different cameras.  
- Annotating each frame with **camera ID, timestamp, and labels**.  
- Producing a final summarized video showing only relevant clips.  

This approach enhances **surveillance and security efficiency** by reducing manual monitoring time.  

---

## 🚀 Features  
- ✅ Person detection using **YOLO**.  
- ✅ Face re-identification using **ResNet**.  
- ✅ Multi-camera tracking with **DeepSORT**.  
- ✅ Annotated output videos with **timestamps & camera IDs**.  
- ✅ Final summarized video combining important clips.  

---

## 🛠️ Tech Stack  
- **Python**  
- **OpenCV**  
- **YOLO**  
- **ResNet**  
- **DeepSORT**  
- **MoviePy**  
- **NumPy, Pandas, PyTorch**  

---

## 📂 Project Structure  
```

📦 CCTV-Summarization
┣ 📂 input\_videos/       # Raw CCTV videos from multiple cameras
┣ 📂 output\_videos/      # Annotated & summarized videos
┣ 📂 models/             # YOLO & ResNet weights
┣ 📜 main.py             # Main script to run summarization
┣ 📜 requirements.txt    # Dependencies
┣ 📜 README.md           # Project documentation
┗ 📜 .gitignore          # Files to ignore in Git

````

---

## ⚡ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux  
venv\Scripts\activate     # Windows  
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

* Add real-time live Multi-Camera streaming support.
* Improve face re-identification accuracy.
* Integrate with cloud storage for large-scale Multi-Camera management.

---

## 👩‍💻 Contributors

* **M. Surya Sunanda** – VIT AP University

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify it.

````

---
