from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3, os, json
from cryptography.fernet import Fernet

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data.db")
KEY_PATH = os.path.join(BASE_DIR, "secret.key")
OPTIONS_PATH = os.path.join(BASE_DIR, "options.json")

app = Flask(__name__)
app.secret_key = "super-secret-session-key-change-this"  # change in production

# --- Encryption helpers ---
def load_key():
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    else:
        with open(KEY_PATH, "rb") as f:
            key = f.read()
    return key

KEY = load_key()
CIPHER = Fernet(KEY)

def encrypt_password(plain: str) -> bytes:
    return CIPHER.encrypt(plain.encode())

def decrypt_password(token: bytes) -> str:
    return CIPHER.decrypt(token).decode()

# --- Database init ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password BLOB NOT NULL,
        is_admin INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        payload TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def ensure_owner():
    # create owner user 'ushin' with password 'Hell6472' if not exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", ("ushin",))
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?,?,1)",
                  ("ushin", encrypt_password("Hell6472")))
        conn.commit()
    conn.close()

init_db()
ensure_owner()

# --- Options JSON init ---
default_options = {
    "Robux": ["50 Robux (free)", "100 Robux (coming soon)"],
    "Grow a Garden": ["Raccoon", "Dragonfly", "Red Fox"],
    "Blox Fruits": ["Kitsune", "Dragon", "Yeti", "Perm Ice"]
}
if not os.path.exists(OPTIONS_PATH):
    with open(OPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(default_options, f, indent=2)

def load_options():
    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_options(obj):
    with open(OPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if not username or not password:
            flash("Username and password required.", "error")
            return redirect(url_for("register"))
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?,?,0)",
                      (username, encrypt_password(password)))
            conn.commit()
            flash("Registered successfully. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already taken.", "error")
            return redirect(url_for("register"))
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password, is_admin FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row:
            enc_pass, is_admin = row
            try:
                dec = decrypt_password(enc_pass)
            except Exception:
                flash("Server error decrypting password.", "error")
                return redirect(url_for("login"))
            if password == dec:
                session["user"] = username
                session["is_admin"] = bool(is_admin)
                flash("Logged in successfully.", "success")
                return redirect(url_for("dashboard"))
        flash("Invalid username/password.", "error")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/admin-login", methods=["GET","POST"])
def admin_login():
    if request.method=="POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password, is_admin FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row:
            enc_pass, is_admin = row
            try:
                dec = decrypt_password(enc_pass)
            except Exception:
                flash("Server error decrypting password.", "error")
                return redirect(url_for("admin_login"))
            if password == dec:
                # allow admin logins only if is_admin flag set
                if is_admin:
                    session["user"] = username
                    session["is_admin"] = True
                    flash("Admin logged in.", "success")
                    return redirect(url_for("admin_dashboard"))
                else:
                    flash("User is not an admin.", "error")
                    return redirect(url_for("admin_login"))
        flash("Invalid admin credentials.", "error")
        return redirect(url_for("admin_login"))
    return render_template("admin_login.html")

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    opts = load_options()
    if request.method=="POST":
        # collect selections
        payload = {}
        for k in opts.keys():
            val = request.form.get(k)
            if val:
                payload[k] = val
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO requests (username, payload) VALUES (?, ?)",
                  (session["user"], json.dumps(payload)))
        conn.commit()
        conn.close()
        return render_template("request_made.html")
    return render_template("dashboard.html", options=opts, username=session["user"])

@app.route("/admin", methods=["GET","POST"])
def admin_dashboard():
    if "user" not in session or not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    # load users and decrypt passwords
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, password, is_admin FROM users")
    users = []
    for u,p,is_admin in c.fetchall():
        try:
            dec = decrypt_password(p)
        except Exception:
            dec = "<decrypt-error>"
        users.append({"username":u,"password":dec,"is_admin":bool(is_admin)})
    c.execute("SELECT id, username, payload, created_at FROM requests ORDER BY created_at DESC")
    requests_list = [{"id":r[0],"username":r[1],"payload":r[2],"created_at":r[3]} for r in c.fetchall()]
    conn.close()
    # options editing
    if request.method=="POST":
        raw = request.form.get("options_json","")
        try:
            obj = json.loads(raw)
            save_options(obj)
            flash("Options updated.", "success")
            return redirect(url_for("admin_dashboard"))
        except Exception as e:
            flash("Invalid JSON: " + str(e), "error")
            return redirect(url_for("admin_dashboard"))
    with open(OPTIONS_PATH,"r",encoding="utf-8") as f:
        options_text = f.read()
    return render_template("admin_dashboard.html", users=users, requests=requests_list, options_text=options_text)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

# API endpoints for admin to add regular admins later
@app.route("/api/make-admin", methods=["POST"])
def make_admin():
    if "user" not in session or not session.get("is_admin"):
        return jsonify({"error":"unauthorized"}), 401
    username = request.json.get("username")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return jsonify({"ok":True})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
