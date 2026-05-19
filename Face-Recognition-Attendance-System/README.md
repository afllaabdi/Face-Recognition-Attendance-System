# 🎓 Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An AI-powered, real-time face recognition system that automatically marks attendance using your webcam.**

</div>

---

## 📌 Overview

This project uses **computer vision** and **deep learning** to identify faces from a live webcam feed and automatically log attendance into a CSV file — no manual input needed. Simply look at the camera and your attendance is marked!

Built as a beginner-friendly Computer Vision project that demonstrates the core concepts of:
- Real-time video processing with **OpenCV**
- Face detection and recognition with **face_recognition** (dlib under the hood)
- Data management with **Pandas**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Real-time detection** | Detects and tracks multiple faces simultaneously |
| 🧠 **Face recognition** | Matches faces against your dataset images |
| 📋 **Auto attendance** | Saves Name, Date, Time to a CSV file automatically |
| 🚫 **No duplicates** | Each person is only marked once per day |
| ❓ **Unknown faces** | Labels unrecognised faces as `UNKNOWN` |
| 📊 **Confidence score** | Shows match confidence percentage on screen |
| 🕐 **Timestamp HUD** | Live date & time overlay on the webcam feed |
| ⚡ **Performance** | Frame downscaling for smooth FPS |
| ⌨️ **Keyboard control** | Press `Q` to cleanly exit the app |

---

## 🖼️ Screenshots

> Add your own screenshots here after running the project!

```
screenshots/
├── detection_known.png      # Known face with green box
├── detection_unknown.png    # Unknown face with red box
└── attendance_csv.png       # Sample attendance CSV output
```

**Example output in the webcam window:**

```
┌─────────────────────────────────────────┐
│ Faces detected: 2               FPS:28  │
│                                         │
│   ┌──── Aflla  97% ─────┐              │
│   │                      │              │
│   │    (face area)       │              │
│   └──────────────────────┘              │
│                       [ Marked ✔ ]      │
│                                         │
│  Monday, 18 May 2026  |  08:21:44  Q=Quit│
└─────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
Face-Recognition-Attendance-System/
│
├── dataset/                  # ← Put face images here
│   ├── aflla.jpg             #   One clear face per image
│   └── budi.jpg              #   Filename = person's name
│
├── attendance/
│   └── attendance.csv        # Auto-generated attendance log
│
├── screenshots/              # Your demo screenshots go here
│
├── main.py                   # Main application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10** or higher
- A working **webcam**
- `cmake` and `dlib` dependencies (see note below)

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Note on `face_recognition` / `dlib`:**
> On some systems you may need to install `cmake` and a C++ compiler before `dlib` can build.
>
> **Windows:** Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
>
> **Ubuntu/Debian:**
> ```bash
> sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev
> ```
>
> **macOS:**
> ```bash
> brew install cmake
> ```

---

## 🚀 How to Run

### 1. Add face images to `dataset/`

- One clear, front-facing photo per person
- Name the file after the person: `john.jpg`, `sarah.png`, etc.
- Supported formats: `.jpg`, `.jpeg`, `.png`

```
dataset/
├── aflla.jpg
├── budi.jpg
└── sarah.png
```

### 2. Run the application

```bash
python main.py
```

### 3. What happens

1. All images in `dataset/` are loaded and encoded
2. Your webcam opens
3. Faces are detected and compared in real time
4. Recognised faces are labelled with name + confidence %
5. Attendance is saved to `attendance/attendance.csv`
6. Press **Q** to quit

### 4. View attendance log

```bash
cat attendance/attendance.csv
```

Example output:
```
Name,Date,Time
Aflla,2026-05-18,08:21:44
Budi,2026-05-18,08:23:11
```

---

## 🛠️ Configuration

Open `main.py` and edit the config section at the top:

```python
TOLERANCE    = 0.50   # Matching strictness: lower = stricter (try 0.4–0.6)
FRAME_SCALE  = 0.5    # Frame downscale ratio for speed (0.25–1.0)
WEBCAM_INDEX = 0      # Webcam ID (0 = default, 1 = external)
PROCESS_EVERY = 2     # Run recognition every N frames (1 = every frame)
```

---

## 📦 Technologies Used

| Library | Version | Purpose |
|---|---|---|
| [OpenCV](https://opencv.org/) | ≥ 4.8 | Webcam capture, frame display, drawing |
| [face_recognition](https://github.com/ageitgey/face_recognition) | ≥ 1.3 | Face detection & encoding (uses dlib) |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Numerical operations, distance calculations |
| [Pandas](https://pandas.pydata.org/) | ≥ 2.0 | CSV read/write for attendance records |

---

## 🔮 Future Improvements

- [ ] **GUI Dashboard** — Build a Tkinter / web UI to view attendance history
- [ ] **Anti-spoofing** — Detect photo attacks (liveness detection)
- [ ] **Multiple cameras** — Support multiple webcam feeds simultaneously
- [ ] **Database backend** — Replace CSV with SQLite or PostgreSQL
- [ ] **Email alerts** — Notify when an unknown face is detected
- [ ] **Export reports** — Generate PDF/Excel attendance reports
- [ ] **Face registration UI** — Add new faces without manually placing files
- [ ] **Mobile app** — React Native or Flutter front-end with API backend

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) — the outstanding face recognition library
- [dlib](http://dlib.net/) — the underlying deep learning model
- [OpenCV](https://opencv.org/) — the backbone of computer vision in Python

---

<div align="center">
Made with ❤️ as a Computer Vision portfolio project
</div>
