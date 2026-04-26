import os, cv2, time, sqlite3, threading
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Use an absolute path to be 100% sure inside the container
CAP_DIR = '/app/data/captures' 
if not os.path.exists(CAP_DIR): 
    os.makedirs(CAP_DIR, exist_ok=True)


# --- 1. SETUP ---
DB_DIR = 'data'
CAP_DIR = 'data/captures'
DB_PATH = os.path.join(DB_DIR, 'business_logic.db')
for d in [DB_DIR, CAP_DIR]: 
    if not os.path.exists(d): os.makedirs(d)

def get_db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# Initial DB Sync
conn = get_db_conn()
conn.cursor().execute('''CREATE TABLE IF NOT EXISTS people 
    (id TEXT PRIMARY KEY, name TEXT, is_monitored BOOLEAN, face_vector BLOB)''')
conn.commit()
conn.close()

# --- 2. DASHBOARD APP ---
app = FastAPI()

# THIS LINE links your /data/captures folder to the /photos URL in your browser
# Ensure the static mount matches this path
app.mount("/photos", StaticFiles(directory=CAP_DIR), name="photos")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, is_monitored FROM people ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    html = """
    <html><head><title>Soayan Vision Hub</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #fff; padding: 40px; }
        table { width: 100%; border-collapse: collapse; background: #2d2d2d; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #444; }
        th { background: #3d3d3d; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; }
        img { border-radius: 4px; border: 1px solid #555; object-fit: cover; }
        .btn { background: #007bff; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        input[type="text"] { padding: 8px; border-radius: 4px; border: 1px solid #444; background: #1a1a1a; color: white; }
    </style></head><body>
    <h2>🚀 Soayan Intelligence Hub: Live Logs</h2>
    <table><tr><th>Capture</th><th>Identity</th><th>Status</th><th>Action</th></tr>
    """
    for uid, name, monitored in rows:
        status_text = "🔴 WATCHED" if monitored else "🟢 CLEAR"
        # Accessing the image via the /photos/ mount
        img_url = f"/photos/{uid}.jpg"
        html += f"""
        <tr>
            <td><img src="{img_url}" width="80" height="80" onerror="this.src='https://via.placeholder.com/80?text=No+Img'"></td>
            <td><strong>{name}</strong><br><small style="color:#888">{uid}</small></td>
            <td>{status_text}</td>
            <td>
                <form action="/rename" method="post" style="display:flex; gap: 10px;">
                    <input type="hidden" name="person_id" value="{uid}">
                    <input type="text" name="new_name" placeholder="Rename...">
                    <button type="submit" class="btn">Identify</button>
                </form>
            </td>
        </tr>
        """
    html += "</table></body></html>"
    return html

@app.post("/rename")
def rename(person_id: str = Form(...), new_name: str = Form(...)):
    conn = get_db_conn()
    cursor = conn.cursor()
    # If naming Aaron, set him to monitored by default
    monitored = 1 if new_name.lower() == "aaron" else 0
    cursor.execute("UPDATE people SET name=?, is_monitored=? WHERE id=?", (new_name, monitored, person_id))
    conn.commit()
    conn.close()
    return RedirectResponse("/", status_code=303)

# --- 3. VISION ENGINE ---
def run_vision():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w]
                res = DeepFace.represent(face_img, model_name='SFace', enforce_detection=False)
                vec = np.array(res[0]["embedding"], dtype=np.float32)
                
                v_conn = get_db_conn()
                v_cur = v_conn.cursor()
                v_cur.execute("SELECT id, name, face_vector FROM people WHERE face_vector IS NOT NULL")
                
                match_id = None
                for pid, name, blob in v_cur.fetchall():
                    stored = np.frombuffer(blob, dtype=np.float32)
                    if (1 - (np.dot(vec, stored)/(np.linalg.norm(vec)*np.linalg.norm(stored)))) < 0.5:
                        match_id = pid
                        break
                
                if not match_id:
                    new_id = f"user_{int(time.time())}"
                    # SAVE THE IMAGE FIRST
                    cv2.imwrite(f"{CAP_DIR}/{new_id}.jpg", frame)
                    v_cur.execute("INSERT INTO people VALUES (?, 'Unknown', 0, ?)", (new_id, vec.tobytes()))
                    v_conn.commit()
                    print(f"📸 Captured New Face: {new_id}")
                v_conn.close()
            except Exception as e:
                print(f"Vision Error: {e}")
        time.sleep(0.2)

if __name__ == "__main__":
    threading.Thread(target=run_vision, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8080)