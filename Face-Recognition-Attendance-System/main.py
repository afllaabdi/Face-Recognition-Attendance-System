"""
╔══════════════════════════════════════════════════════════════╗
║         Face Recognition Attendance System                   ║
║         Built with OpenCV + face_recognition                 ║
╚══════════════════════════════════════════════════════════════╝

Author  : Your Name
Version : 1.0.0
Date    : 2026
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import datetime
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_DIR       = "dataset"
ATTENDANCE_DIR    = "attendance"
ATTENDANCE_FILE   = os.path.join(ATTENDANCE_DIR, "attendance.csv")
TOLERANCE         = 0.50        # Lower = stricter matching (0.4–0.6 recommended)
FRAME_SCALE       = 0.5         # Downscale frame for faster recognition
WEBCAM_INDEX      = 0           # 0 = default webcam
WINDOW_TITLE      = "Face Recognition Attendance System  |  Press Q to Quit"

# ─────────────────────────────────────────────
# COLORS  (BGR format)
# ─────────────────────────────────────────────
COLOR_KNOWN      = (0, 220, 90)     # Green — recognized face
COLOR_UNKNOWN    = (0, 60, 220)     # Red   — unknown face
COLOR_TEXT_BG    = (20, 20, 20)     # Dark label background
COLOR_OVERLAY    = (0, 0, 0)        # Timestamp bar background
COLOR_WHITE      = (255, 255, 255)
COLOR_YELLOW     = (0, 200, 255)    # Highlight color


# ══════════════════════════════════════════════
# 1. DATASET LOADING
# ══════════════════════════════════════════════

def load_known_faces(dataset_dir: str) -> tuple[list, list]:
    """
    Scan the dataset folder and encode every face image found.
    Supported formats: .jpg, .jpeg, .png
    Returns:
        known_encodings : list of 128-d face encodings
        known_names     : list of names (derived from filename)
    """
    known_encodings = []
    known_names     = []

    supported = (".jpg", ".jpeg", ".png")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset folder '{dataset_dir}' not found.")
        print(f"        Create it and add face images (e.g. john.jpg)")
        return known_encodings, known_names

    image_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(supported)
    ]

    if not image_files:
        print(f"[WARN]  No images found in '{dataset_dir}'.")
        return known_encodings, known_names

    print(f"\n[INFO] Loading {len(image_files)} image(s) from '{dataset_dir}' …")

    for filename in image_files:
        path = os.path.join(dataset_dir, filename)
        name = os.path.splitext(filename)[0].capitalize()

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"  ✔  Loaded: {name}")
        else:
            print(f"  ✘  No face detected in: {filename} — skipped")

    print(f"[INFO] {len(known_names)} face(s) encoded and ready.\n")
    return known_encodings, known_names


# ══════════════════════════════════════════════
# 2. ATTENDANCE MANAGEMENT
# ══════════════════════════════════════════════

def init_attendance_file(filepath: str) -> None:
    """Create the CSV file with headers if it doesn't already exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(filepath, index=False)
        print(f"[INFO] Attendance file created: {filepath}")


def mark_attendance(name: str, filepath: str) -> bool:
    """
    Add an attendance record for today — prevents duplicates.
    Returns True if the record was newly written, False if already exists.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    now   = datetime.datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(filepath)

    # Check for an existing entry for this person today
    already_marked = (
        (df["Name"] == name) & (df["Date"] == today)
    ).any()

    if not already_marked:
        new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filepath, index=False)
        print(f"[✔] Attendance marked — {name}  {today}  {now}")
        return True

    return False   # Already marked today


# ══════════════════════════════════════════════
# 3. DRAWING HELPERS
# ══════════════════════════════════════════════

def draw_face_box(frame, top, right, bottom, left,
                  name: str, confidence: float, marked: bool, color):
    """
    Draw a stylish bounding box with name label and confidence badge.
    """
    # ── Corner-style bounding box ──────────────────────────
    thickness  = 2
    corner_len = 20

    # Top-left corners
    cv2.line(frame, (left, top),  (left + corner_len, top),  color, thickness + 1)
    cv2.line(frame, (left, top),  (left, top + corner_len),  color, thickness + 1)
    # Top-right corners
    cv2.line(frame, (right, top), (right - corner_len, top), color, thickness + 1)
    cv2.line(frame, (right, top), (right, top + corner_len), color, thickness + 1)
    # Bottom-left corners
    cv2.line(frame, (left, bottom),  (left + corner_len, bottom),  color, thickness + 1)
    cv2.line(frame, (left, bottom),  (left, bottom - corner_len),  color, thickness + 1)
    # Bottom-right corners
    cv2.line(frame, (right, bottom), (right - corner_len, bottom), color, thickness + 1)
    cv2.line(frame, (right, bottom), (right, bottom - corner_len), color, thickness + 1)

    # Light rectangle fill for the main box
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, 1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # ── Name label ─────────────────────────────────────────
    conf_text  = f"{confidence:.0f}%"
    label      = f"  {name}  {conf_text}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1

    (label_w, label_h), baseline = cv2.getTextSize(
        label, font, font_scale, font_thick
    )

    label_top    = max(0, top - label_h - baseline - 6)
    label_bottom = top

    # Label background
    cv2.rectangle(
        frame,
        (left, label_top),
        (left + label_w + 4, label_bottom),
        color, -1
    )
    # Label text
    cv2.putText(
        frame, label,
        (left + 2, label_bottom - baseline - 2),
        font, font_scale, COLOR_WHITE, font_thick, cv2.LINE_AA
    )

    # ── "Marked ✔" badge ───────────────────────────────────
    if marked:
        badge = "  Marked "
        (bw, bh), _ = cv2.getTextSize(badge, font, 0.45, 1)
        bx = right - bw - 6
        by = bottom + bh + 6
        cv2.rectangle(frame, (bx - 2, bottom + 2), (right, by + 2),
                      COLOR_KNOWN, -1)
        cv2.putText(frame, badge, (bx, by),
                    font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_hud(frame):
    """
    Draw a semi-transparent HUD bar at the bottom with date & time.
    """
    h, w = frame.shape[:2]
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    now_str = datetime.datetime.now().strftime("  %A, %d %B %Y   |   %H:%M:%S")
    cv2.putText(
        frame, now_str,
        (6, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_YELLOW, 1, cv2.LINE_AA
    )

    # Right-side hint
    hint = "Q = Quit  "
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        frame, hint,
        (w - hw - 4, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA
    )


def draw_status_bar(frame, n_faces: int):
    """Top status bar showing face count."""
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    status = f"  Faces detected: {n_faces}"
    cv2.putText(frame, status, (4, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════
# 4. MAIN RECOGNITION LOOP
# ══════════════════════════════════════════════

def run(known_encodings: list, known_names: list) -> None:
    """
    Open the webcam and continuously detect / recognise faces.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check WEBCAM_INDEX in config.")
        return

    # Small cache: remember who was marked today so we don't re-write CSV
    marked_today: set[str] = set()

    print("[INFO] Webcam started. Press Q to quit.\n")

    # Track FPS for smooth display
    prev_time    = time.time()

    # Throttle recognition (process every N frames for speed)
    frame_count  = 0
    PROCESS_EVERY = 2       # Recognise every 2nd frame

    # Keep last known results to display on skipped frames
    last_boxes   = []       # [(top, right, bottom, left), …]
    last_names   = []       # [name, …]
    last_confs   = []       # [confidence %, …]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — retrying …")
            continue

        frame_count += 1

        # ── FPS calculation ────────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ── Face recognition (every Nth frame) ────────────
        if frame_count % PROCESS_EVERY == 0:
            # Resize for speed, convert BGR→RGB
            small  = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Locate and encode faces in the current frame
            boxes_small   = face_recognition.face_locations(rgb, model="hog")
            face_encodings = face_recognition.face_encodings(rgb, boxes_small)

            last_boxes = []
            last_names = []
            last_confs = []

            for encoding, box_small in zip(face_encodings, boxes_small):
                # Scale bounding box back to original size
                t, r, b, l = [int(v / FRAME_SCALE) for v in box_small]
                last_boxes.append((t, r, b, l))

                # Compare against known faces
                distances = face_recognition.face_distance(known_encodings, encoding)

                name       = "UNKNOWN"
                confidence = 0.0

                if len(distances) > 0:
                    best_idx  = int(np.argmin(distances))
                    best_dist = distances[best_idx]
                    confidence = max(0.0, (1 - best_dist) * 100)

                    if best_dist <= TOLERANCE:
                        name = known_names[best_idx]

                        # Mark attendance if not yet done today
                        if name not in marked_today:
                            was_new = mark_attendance(name, ATTENDANCE_FILE)
                            if was_new:
                                marked_today.add(name)

                last_names.append(name)
                last_confs.append(confidence)

        # ── Draw results ───────────────────────────────────
        for (top, right, bottom, left), name, conf in zip(
            last_boxes, last_names, last_confs
        ):
            color  = COLOR_KNOWN if name != "UNKNOWN" else COLOR_UNKNOWN
            marked = name in marked_today

            draw_face_box(frame, top, right, bottom, left,
                          name, conf, marked, color)

        # ── HUD overlays ───────────────────────────────────
        draw_status_bar(frame, len(last_boxes))
        draw_hud(frame)

        # FPS badge (top-right)
        fps_text = f"FPS: {fps:.1f}  "
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, fps_text,
                    (frame.shape[1] - fw - 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (100, 255, 100), 1, cv2.LINE_AA)

        # ── Show frame ─────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        # Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received — shutting down …")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Attendance saved to:", ATTENDANCE_FILE)


# ══════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("   Face Recognition Attendance System  v1.0")
    print("=" * 62)

    # Step 1 — Prepare attendance CSV
    init_attendance_file(ATTENDANCE_FILE)

    # Step 2 — Load and encode all faces in dataset/
    known_encodings, known_names = load_known_faces(DATASET_DIR)

    if not known_encodings:
        print("\n[WARN] No faces loaded. The system will label everyone UNKNOWN.")
        print("       Add face images to the 'dataset/' folder and restart.\n")

    # Step 3 — Start the webcam recognition loop
    run(known_encodings, known_names)
