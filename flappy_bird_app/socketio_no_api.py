from fastapi import FastAPI
from fastapi_socketio import SocketManager
from pydantic import BaseModel
from typing import List

app = FastAPI()
socket_manager = SocketManager(app=app)

# Define the game state schema
class Pipe(BaseModel):
    x: int
    y: int

class GameState(BaseModel):
    birdY: int
    pipes: List[Pipe]
    score: int
    isGameOver: bool

class ActionRequest(BaseModel):
    action: str
    state: GameState

# A global variable to store the game state (or use a database)
game_state = {}

# Socket.IO connection event
@app.sio.on('connect')
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await app.sio.emit('message', {'data': 'Welcome to Flappy Bird Game!'})

# Socket.IO disconnect event
@app.sio.on('disconnect')
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Event to receive game state from the client
@app.sio.on('game_state')
async def receive_game_state(sid, data: dict):
    global game_state
    game_state = data
    print(f"Received game state: {game_state}")

    # Here you can process the game state and send back any actions or updates
    await app.sio.emit('game_update', {'status': 'Game state updated', 'game_state': game_state})

# Event to receive actions from the client
@app.sio.on('game_action')
async def receive_action(sid, action_data: dict):
    print(f"Received action: {action_data}")
    # Process action and respond
    await app.sio.emit('action_ack', {'status': 'Action received', 'action': action_data})

