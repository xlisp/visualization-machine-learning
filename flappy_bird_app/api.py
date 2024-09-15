from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

# Serve static files (like HTML, JavaScript, images) from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Game state
game_state = {
    "birdY": 0,
    "pipes": [],
    "score": 0,
    "isGameOver": False,
}

# Action sent from the DQN agent (0: no flap, 1: flap)
class ActionRequest(BaseModel):
    action: int

@app.post("/game/action/")
async def take_action(action_request: ActionRequest):
    game_state['action'] = action_request.action
    return {"status": "action_received"}

@app.get("/game/action/")
async def get_action():
    # Return a pre-determined action (0: no flap, 1: flap)
    # In practice, the action should be determined by the DQN agent.
    action = game_state.get('action', 0)
    return {"action": action}

@app.post("/game/state/")
async def receive_state(request: Request):
    data = await request.json()
    game_state.update(data)  # Update the global game state
    print("------receive_state------")
    print(data)  # Debugging: Print received data
    return {"status": "state_received"}

@app.get("/game/state/")
async def get_game_state():
    return game_state

# Serve the main HTML file
@app.get("/")
async def serve_html():
    return FileResponse('static/index.html')


if __name__ == "__main__":
    # Ensure the static folder exists and contains the HTML
    if not os.path.exists('static'):
        os.makedirs('static')
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

