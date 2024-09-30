import requests

action_data = {
    "action": "flap",  # This matches the "action" field expected by FastAPI
    "state": {  # This matches the "state" object expected by FastAPI
        "birdY": 150,  # birdY as an integer
        "pipes": [  # pipes as a list of dictionaries with "x" and "y"
            {"x": 300, "y": 400},
            {"x": 500, "y": 200}
        ],
        "score": 10,  # score as an integer
        "isGameOver": False  # isGameOver as a boolean
    }
}

# Send POST request to the FastAPI server
response = requests.post('http://localhost:8000/game/action/', json=action_data)

if response.status_code == 200:
    print("Action sent successfully:", response.json())
else:
    print("Failed to send action:", response.status_code, response.json())

