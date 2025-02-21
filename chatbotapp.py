rom flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import os
from dotenv import load_dotenv
from chatbot import query_knowledge_base, load_pdfs

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")

# ✅ Ensure database is initialized
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()



# ================================
# 🔹 Home Page
# ================================
@app.route("/")
def home():
    return render_template("index.html")

# ================================
# 🔹 Signup Page
# ================================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = ""
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))  # Redirect to login page
        except sqlite3.IntegrityError:
            message = "❌ Username or Email already exists"

    return render_template("signup.html", message=message)

# ================================
# 🔹 Login Page
# ================================
@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["username"] = username
            return redirect(url_for("chatbot_page"))
        else:
            message = "❌ Invalid Credentials"

    return render_template("login.html", message=message)

# ================================
# 🔹 Chatbot Page (Requires Login)
# ================================
@app.route("/chatbot")
def chatbot_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("chatbot.html")

# ================================
# 🔹 Chatbot Query API
# ================================
@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    user_input = request.json.get("message", "")
    response = query_knowledge_base(user_input)  # Get chatbot response

    # ✅ Return JSON response to frontend
    return jsonify({"response": response})


# ================================
# 🔹 Logout
# ================================
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
