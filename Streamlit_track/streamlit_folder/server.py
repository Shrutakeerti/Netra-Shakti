from flask import Flask
from flask_socketio import SocketIO, join_room, leave_room, send, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enables real-time communication

rooms = {}  # Stores active chat rooms
clients = {}  # Stores WebRTC users

@app.route("/")
def home():
    return "Chat & Video Backend is Running!"

# ---------------- Chat Functionality ---------------- #

@socketio.on("join_room")
def handle_join(data):
    username = data["username"]
    room = data["room"]
    join_room(room)

    if room not in rooms:
        rooms[room] = []
    rooms[room].append(username)

    send({"message": f"{username} joined the room!", "username": "System"}, room=room)
    print(f"{username} joined room {room}")

@socketio.on("send_message")
def handle_message(data):
    room = data["room"]
    send({"message": data["message"], "username": data["username"]}, room=room)
    print(f"Message from {data['username']} in room {room}: {data['message']}")

@socketio.on("leave_room")
def handle_leave(data):
    username = data["username"]
    room = data["room"]
    leave_room(room)
    
    if room in rooms and username in rooms[room]:
        rooms[room].remove(username)

    send({"message": f"{username} left the room!", "username": "System"}, room=room)
    print(f"{username} left room {room}")

# ---------------- WebRTC Signaling ---------------- #

@socketio.on("join")
def handle_webrtc_join(data):
    username = data["username"]
    room = data["room"]
    join_room(room)
    clients[username] = room
    emit("user_joined", {"username": username}, room=room)
    print(f"{username} joined video call in room {room}")

@socketio.on("offer")
def handle_offer(data):
    emit("offer", data, room=data["room"])

@socketio.on("answer")
def handle_answer(data):
    emit("answer", data, room=data["room"])

@socketio.on("ice_candidate")
def handle_ice_candidate(data):
    emit("ice_candidate", data, room=data["room"])

@socketio.on("disconnect")
def handle_disconnect():
    for username, room in clients.items():
        leave_room(room)
        print(f"{username} disconnected from room {room}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
