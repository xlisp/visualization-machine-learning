// Connect to the FastAPI server via Socket.IO

const socket = io('http://0.0.0.0:8000', {
    transports: ['websocket'],  // Force WebSocket transport
});

// Game variables
let bird = {
    y: 200,
    velocity: 0
};
let pipes = [];
let score = 0;
let isGameOver = false;

// When the client connects successfully
socket.on('connect', () => {
    console.log('Connected to the server via Socket.IO');
});

// Listening for game state updates from the server
socket.on('game_update', (data) => {
    console.log('Game update received from server:', data);
    // Handle server updates to the game state here (optional)
});

// Listening for action acknowledgements from the server
socket.on('action_ack', (data) => {
    console.log('Action acknowledged by server:', data);
});

// Function to send the game state to the server
function sendGameState() {
    const state = {
        birdY: bird.y,
        pipes: pipes.map(pipe => ({ x: pipe.x, y: pipe.y })),  // Sending pipe coordinates
        score: score,
        isGameOver: isGameOver
    };

    socket.emit('game_state', state);  // Emit game state to the server
    console.log('Game state sent:', state);
}

// Function to send player action to the server (like bird flapping)
function sendAction(action) {
    const actionData = {
        action: action,  // e.g., 'flap'
        state: {
            birdY: bird.y,
            pipes: pipes.map(pipe => ({ x: pipe.x, y: pipe.y })),
            score: score,
            isGameOver: isGameOver
        }
    };

    socket.emit('game_action', actionData);  // Emit action to the server
    console.log('Action sent:', actionData);
}

// Game logic (pseudo Flappy Bird)
function gameLoop() {
    if (isGameOver) {
        return;
    }

    // Update bird position
    bird.y += bird.velocity;
    bird.velocity += 0.5;  // Gravity

    // Move pipes
    for (let pipe of pipes) {
        pipe.x -= 5;  // Move pipe left
    }

    // Check for collision, etc.
    checkCollisions();

    // Render game
    drawGame();

    // Send game state to server
    sendGameState();

    // Continue the loop
    requestAnimationFrame(gameLoop);
}

// Function to handle player action (e.g., flap)
function flap() {
    bird.velocity = -8;  // Make bird flap upward
    sendAction('flap');  // Send flap action to server
}

// Event listener for user input (spacebar to flap)
document.addEventListener('keydown', function(event) {
    if (event.code === 'Space') {
        flap();
    }
});

// Function to check collisions (pseudo-code)
function checkCollisions() {
    // Simple logic to detect collisions between bird and pipes or ground
    if (bird.y >= 400 || bird.y <= 0) {
        isGameOver = true;
        console.log('Game Over');
    }
}

// Function to draw the game on canvas (pseudo-code)
function drawGame() {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bird
    ctx.fillStyle = 'yellow';
    ctx.fillRect(50, bird.y, 20, 20);  // Simple bird representation

    // Draw pipes
    for (let pipe of pipes) {
        ctx.fillStyle = 'green';
        ctx.fillRect(pipe.x, pipe.y, 50, 150);  // Simple pipe representation
    }

    // Draw score
    ctx.fillStyle = 'black';
    ctx.font = '24px Arial';
    ctx.fillText(`Score: ${score}`, 10, 30);
}

// Start the game loop
gameLoop();
