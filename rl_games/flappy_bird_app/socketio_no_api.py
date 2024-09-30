import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi_socketio import SocketManager
from pydantic import BaseModel
from typing import List

# Create FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SocketIO manager and bind it to the app
socket_manager = SocketManager(app=app)

# Mount the static files directory to serve the HTML and assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the game state data model
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

# Global variable to store the latest game state
game_state = {}

# Handle client connection
@app.sio.on('connect')
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await app.sio.emit('message', {'data': 'Welcome to the Flappy Bird Game!'}, room=sid)

# Handle client disconnection
@app.sio.on('disconnect')
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Receive game state from the frontend
@app.sio.on('game_state')
async def receive_game_state(sid, data: dict):
    global game_state
    game_state = data  # Store the received game state
    print(f"Received game state: {game_state}")

    # Send an acknowledgment to the client
    await app.sio.emit('game_update', {'status': 'Game state updated', 'game_state': game_state}, room=sid)

# Receive player action from the frontend
@app.sio.on('game_action')
async def receive_action(sid, action_data: dict):
    print(f"Received action from client {sid}: {action_data}")

    # Process action (e.g., bird flap) if necessary, and update state
    action = action_data.get("action")
    game_state = action_data.get("state")
    
    # Here, you can add custom logic to modify game state based on action
    # For example: action could be 'flap' which might update the bird's position

    # Send an acknowledgment to the client
    await app.sio.emit('action_ack', {'status': 'Action received', 'action': action_data}, room=sid)

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

