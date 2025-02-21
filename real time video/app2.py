from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room, send, emit

app = Flask(__name__)
socketio = SocketIO(app)

rooms = {}  # Store users in rooms

@app.route('/')
def index():
    return render_template('video_chat.html')

@socketio.on('join')
def handle_join(data):
    room = data['room']
    username = data['username']
    join_room(room)

    if room not in rooms:
        rooms[room] = []
    rooms[room].append(username)

    emit('user_joined', {'username': username}, room=room)

@socketio.on('message')
def handle_message(data):
    room = data['room']
    username = data['username']
    message = data['message']

    emit('message', {'username': username, 'message': message}, room=room, include_self=False)

@socketio.on('offer')
def handle_offer(data):
    room = data['room']
    emit('offer', data, room=room, include_self=False)

@socketio.on('answer')
def handle_answer(data):
    room = data['room']
    emit('answer', data, room=room, include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    room = data['room']
    emit('ice-candidate', data, room=room, include_self=False)

if __name__ == '__main__':
    print("Netra Sanchalaya")
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
