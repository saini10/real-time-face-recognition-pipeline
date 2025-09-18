# Real-Time Face Recognition Pipeline 👤🎥

A complete **real-time face recognition system** for employee identification and access control. This project shows the full pipeline: from **dataset collection** → **face detection & embeddings** → **custom neural network training** → **real-time recognition** using a webcam.

---

## 🚀 Overview

* **Data collection technique**: faces captured from multiple angles, distances, and lighting conditions for robustness.
* **Detection**: MTCNN for cropping high-quality face regions.
* **Embeddings**: FaceNet (InceptionResnetV1 pretrained on VGGFace2) → 512-D vectors.
* **Classification**: custom PyTorch neural network trained from scratch.
* **Real-time pipeline**: YOLOv8 + MTCNN + classifier + threshold for unknown faces.
* **Accuracy**: ~94–95% on curated test set.

---

## 🧩 Pipeline

1. **Image collection** → capture varied angles/lighting (per person in `dataset/<name>/`).
2. **Step 1**: crop faces → `faces/`
3. **Step 2**: embeddings → `embeddings.npy`, `labels.npy`
4. **Step 3**: train NN classifier → `face_classifier.pth`
5. **Step 4**: run real-time recognition (`FullPipeline_V1.py`).

---

## 📂 Dataset format

```
dataset/
├── person1/
├── person2/
└── unknown/     # negative examples (public celeb dataset or mixed faces)
```

⚠️ For privacy, this repo ignores personal datasets but keeps `unknown` (celeb data).

---

## 📊 Results

* Robust across poses, lighting, and angles.
* Real-time inference on webcam with threshold-based **unknown** handling.

*See the full report for methodology & evaluation:* [📄 Full Project Report](./Project_Report.pdf)


---

## 🛠️ Tech Stack

* PyTorch
* FaceNet (InceptionResnetV1, VGGFace2)
* MTCNN
* YOLOv8 (Ultralytics)
* OpenCV, NumPy, scikit-learn, Matplotlib

---

## 📌 Future Work

* **Deployment** – web or mobile app integration for real-world use.  
* **Scalability** – efficient search in large employee databases (e.g., FAISS).  
* **Robust Unknown Detection** – advanced techniques like OpenMax or metric learning.  
* **Multi-camera Support** – handle multiple video feeds in real-time.  

---

## 📜 License

MIT License — free to use and adapt.