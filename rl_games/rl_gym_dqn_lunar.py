# To play the game "LunarLander-v2" using PyTorch, you'll need to integrate the game environment with a PyTorch model. You can train a neural network using reinforcement learning (e.g., Deep Q-Networks, or DQN) and then use it to interact with the environment. Here’s a basic outline of how to set it up:

# ### 1. **Install required libraries:**
# Make sure you have the necessary libraries installed:
# ```bash
#pip install gymnasium torch numpy
# ```

# ### 2. **Define the DQN architecture:**
# Define a simple neural network for the agent using PyTorch. This network will learn to take actions based on the state from the game.

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Define a simple fully connected neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
# ```

# ### 3. **Initialize the environment and model:**

# ```python
import gymnasium as gym
import torch

env = gym.make("LunarLander-v2", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Create the DQN model
model = DQN(input_dim=state_dim, output_dim=action_dim)
# ```

# ### 4. **Define the training loop:**
# In this section, we'll define how the agent interacts with the environment, how rewards are collected, and how the model is updated.

# ```python
# Parameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to choose action (using epsilon-greedy policy)
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_dim)  # Random action
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

# Function to train the model
def train_model(memory, batch_size=64):
    if len(memory) < batch_size:
        return

    # Randomly sample a batch from memory
    batch = random.sample(memory, batch_size)

    # Extract states, actions, rewards, next_states, and dones from the batch
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert them to tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute Q values for the current states
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the maximum Q values for the next states
    next_q_values = model(next_states).max(1)[0]

    # Compute the target Q values
    q_targets = rewards + (1 - dones) * gamma * next_q_values

    # Compute the loss
    loss = F.mse_loss(q_values, q_targets)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

###### -------- lmdb -------
import lmdb
import pickle
import torch

# Function to save model to LMDB
def save_model_lmdb(model, lmdb_path="dqn_model.lmdb"):
    with lmdb.open(lmdb_path, map_size=10**9) as env:  # map_size is in bytes, so 1e9 = 1 GB
        with env.begin(write=True) as txn:
            # Serialize the model
            model_data = pickle.dumps(model.state_dict())
            txn.put(b'model', model_data)

# Function to load model from LMDB
def load_model_lmdb(model, lmdb_path="dqn_model.lmdb"):
    with lmdb.open(lmdb_path, map_size=10**9) as env:
        with env.begin(write=False) as txn:
            model_data = txn.get(b'model')
            model.load_state_dict(pickle.loads(model_data))
###
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# Store rewards for each episode
rewards_per_episode = []
# Initialize TensorBoard writer
writer = SummaryWriter()
###

# Main loop
memory = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0

    for t in range(1000):
        action = choose_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        train_model(memory)

        state = next_state
        total_reward += reward
        if done or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Save the model to LMDB every 10 episodes
    if (episode + 1) % 10 == 0:
        save_model_lmdb(model, "dqn_model.lmdb")

    # Log the total reward for each episode
    writer.add_scalar('Reward per episode', total_reward, episode)

    # Plot the total rewards every 10 episodes
    rewards_per_episode.append(total_reward)
    if (episode + 1) % 10 == 0:
        plt.plot(rewards_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        #plt.show() # will block. TODO: plt.pause(0.001) no effect

writer.close()

env.close()

# ```

# ### 5. **Explanation of Key Components:**
# - **Epsilon-Greedy Policy:** This balances exploration and exploitation. Initially, the agent will explore randomly, but over time it will favor actions suggested by the DQN model.
# - **Training the Model:** The `train_model` function samples random batches from the memory and updates the model using the mean squared error loss between predicted and target Q-values.
# - **Memory Buffer:** Stores the experience (state, action, reward, next state, done) to train the model.

# This is a basic structure to get you started. You can improve this by adding more advanced techniques like experience replay buffers, target networks, or using other reinforcement learning algorithms such as Proximal Policy Optimization (PPO) or Advantage Actor-Critic (A2C).

## ---- run ------- 
# @ prunp rl_gym_dqn_lunar.py
# Skipping virtualenv creation, as specified in config file.
# /Users/emacspy/EmacsPyPro/emacspy-machine-learning/rl_gym_dqn_lunar.py:84: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:281.)
#   states = torch.FloatTensor(states)
# Episode 1, Total Reward: -206.46381967632902
# Episode 2, Total Reward: -344.3542455293683
# Episode 3, Total Reward: -161.91831089803713
# Episode 4, Total Reward: -117.33799333877182
# Episode 5, Total Reward: -66.1021129855811
# Episode 6, Total Reward: -200.1378505393618
# Episode 7, Total Reward: -179.6963105533485
# Episode 8, Total Reward: -120.04215451095169
# Episode 9, Total Reward: -118.14319149955723
# Episode 10, Total Reward: -116.494823369448
# Episode 11, Total Reward: -207.54474416025403
# Episode 12, Total Reward: -262.1394876117348
# Episode 13, Total Reward: -125.41318526404015
# Episode 14, Total Reward: -125.46527563038788
# Episode 15, Total Reward: -154.3327107706895
# Episode 16, Total Reward: -327.5104501905113
# Episode 17, Total Reward: -85.4966336049109
# Episode 18, Total Reward: -131.58566888022824
# Episode 19, Total Reward: -116.95014073396192
# Episode 20, Total Reward: -63.938117031213764
# Episode 21, Total Reward: -337.21758991768127
# Episode 22, Total Reward: -394.90873924659707
# Episode 23, Total Reward: -230.02371165362493
# Episode 24, Total Reward: -61.96788504656981
# Episode 25, Total Reward: -405.872091201956
# Episode 26, Total Reward: -319.01074124946564
# Episode 27, Total Reward: -68.58371687948636
# Episode 28, Total Reward: -208.67964733012053
# Episode 29, Total Reward: -51.827876544341024
# Episode 30, Total Reward: -83.33713889414429
# Episode 31, Total Reward: -94.88980916901622
# Episode 32, Total Reward: -97.37802660223564
# Episode 33, Total Reward: -74.26735628368587
# Episode 34, Total Reward: -71.62012835237458
# Episode 35, Total Reward: -90.68939678071521
# Episode 36, Total Reward: -144.93212368356276
# Episode 37, Total Reward: -126.80286955688659
# Episode 38, Total Reward: -110.53375400554708
# Episode 39, Total Reward: -76.4054877469668
# Episode 40, Total Reward: -113.52912241287474
# Episode 41, Total Reward: -98.26023458425821
# Episode 42, Total Reward: -119.3297077710765
# Episode 43, Total Reward: -78.18283404151155
# Episode 44, Total Reward: -59.7081668341807
# Episode 45, Total Reward: -118.96543819038791
# Episode 46, Total Reward: -156.59397070150789
# Episode 47, Total Reward: -92.90259940113158
# Episode 48, Total Reward: -166.13484938269352
# Episode 49, Total Reward: -133.59352710439674
# Episode 50, Total Reward: -161.63310229556106
# Episode 51, Total Reward: -89.18466623200541
# Episode 52, Total Reward: -132.3375209916794
# Episode 53, Total Reward: -64.39956452610826
# Episode 54, Total Reward: -88.05751293299858
# Episode 55, Total Reward: -113.00907303839588
# Episode 56, Total Reward: -118.55374841413888
# Episode 57, Total Reward: -128.9050357974574
# Episode 58, Total Reward: -92.64165065455161
# Episode 59, Total Reward: -35.97646775348218
# Episode 60, Total Reward: -138.64319008629758
# Episode 61, Total Reward: -200.11110981890272
# Episode 62, Total Reward: -129.13081572579483
# Episode 63, Total Reward: -48.22786775563175
# Episode 64, Total Reward: -136.42792097748205
# Episode 65, Total Reward: -70.50889857618424
# Episode 66, Total Reward: -88.00121205244315
# Episode 67, Total Reward: -45.024850555153634
# Episode 68, Total Reward: -250.4255292813832
# Episode 69, Total Reward: -85.55111103832017
# Episode 70, Total Reward: -159.08173220736214
# Episode 71, Total Reward: -317.0380575770548
# Episode 72, Total Reward: -102.065356955241
# Episode 73, Total Reward: -30.819462424371906
# Episode 74, Total Reward: -51.97311669397766
# Episode 75, Total Reward: -85.28648577701286
# Episode 76, Total Reward: -111.45070192575673
# Episode 77, Total Reward: -124.01219718652366
# Episode 78, Total Reward: -17.704804004000295
# Episode 79, Total Reward: -90.42438087909693
# Episode 80, Total Reward: -217.09730084260497
# Episode 81, Total Reward: -31.432354365716535
# Episode 82, Total Reward: -45.178909458913736
# Episode 83, Total Reward: -185.97318004533344
# Episode 84, Total Reward: -98.20546600852488
# Episode 85, Total Reward: -92.17456380096888
# Episode 86, Total Reward: -61.6397863396676
# Episode 87, Total Reward: -50.735793967520095
# Episode 88, Total Reward: -93.25671449326505
# Episode 89, Total Reward: -53.97038325339206
# Episode 90, Total Reward: -54.592211608093
# Episode 91, Total Reward: -200.50217539437512
# Episode 92, Total Reward: -12.723538652291666
# Episode 93, Total Reward: -55.86082969748916
# Episode 94, Total Reward: -131.51518749833227
# Episode 95, Total Reward: -90.22362397522049
# Episode 96, Total Reward: -227.29306301610893
# Episode 97, Total Reward: -98.54488366251532
# Episode 98, Total Reward: -60.37347099489859
# Episode 99, Total Reward: -45.703596508265164
# Episode 100, Total Reward: -141.51494485127935
# Episode 101, Total Reward: 58.006696668195076
# Episode 102, Total Reward: -15.841378470795107
# Episode 103, Total Reward: -103.04420263445397
# Episode 104, Total Reward: -58.57480298694616
# Episode 105, Total Reward: -28.31184619123411
# Episode 106, Total Reward: -99.31000572889707
# Episode 107, Total Reward: 1.8163082913429776
# Episode 108, Total Reward: -72.34831852754759
# Episode 109, Total Reward: -201.73274775967687
# Episode 110, Total Reward: -60.68242305908943
# Episode 111, Total Reward: -87.50976161004867
# Episode 112, Total Reward: -101.47894274644922
# Episode 113, Total Reward: -4.634358042774537
# Episode 114, Total Reward: -67.55239603452486
# Episode 115, Total Reward: -30.84659337344894
# Episode 116, Total Reward: -128.42427442010228
# Episode 117, Total Reward: -14.880009678145527
# Episode 118, Total Reward: -57.981821132786585
# Episode 119, Total Reward: 23.500370443403142
# Episode 120, Total Reward: -52.237278299426734
# Episode 121, Total Reward: -24.781210153894463
# Episode 122, Total Reward: -75.32551216731976
# Episode 123, Total Reward: -69.27950472312872
# Episode 124, Total Reward: -200.2586975556564
# Episode 125, Total Reward: -24.129928555862108
# Episode 126, Total Reward: -33.426226005711115
# Episode 127, Total Reward: -48.68274879704098
# Episode 128, Total Reward: 3.6692082303173237
# Episode 129, Total Reward: -14.013051063290519
# Episode 130, Total Reward: -61.66042398423563
# Episode 131, Total Reward: -38.42405503905633
# Episode 132, Total Reward: -20.10694377167303
# Episode 133, Total Reward: -11.170111016802878
# Episode 134, Total Reward: -18.788624931110007
# Episode 135, Total Reward: -75.7960412834663
# Episode 136, Total Reward: -31.295618356784033
# Episode 137, Total Reward: -61.48325544681481
# Episode 138, Total Reward: -11.16533393987244
# Episode 139, Total Reward: -45.157012799998206
# Episode 140, Total Reward: -27.95847198842843
# Episode 141, Total Reward: -54.941363544538746
# Episode 142, Total Reward: 15.229286282870689
# Episode 143, Total Reward: -8.180527299238378
# Episode 144, Total Reward: -59.542186272245466
# Episode 145, Total Reward: -27.85626260542392
# Episode 146, Total Reward: -45.14712478735288
# Episode 147, Total Reward: -68.70880891590164
# Episode 148, Total Reward: -41.752440955535604
# Episode 149, Total Reward: -17.14843778972856
# Episode 150, Total Reward: -240.4192490277996
# Episode 151, Total Reward: -6.961821456294146
# Episode 152, Total Reward: -27.890813737085168
# Episode 153, Total Reward: 51.927925099113025
# Episode 154, Total Reward: -69.08812472909923
# Episode 155, Total Reward: -10.872626104916549
# Episode 156, Total Reward: -15.197761706462813
# Episode 157, Total Reward: -50.73805975881868
# Episode 158, Total Reward: -34.05101719420597
# Episode 159, Total Reward: 12.849559901699337
# Episode 160, Total Reward: 33.537260544406365
# Episode 161, Total Reward: 17.918229614303556
# Episode 162, Total Reward: 9.60774747399961
# Episode 163, Total Reward: -253.84563795997508
# Episode 164, Total Reward: -155.9752598112836
# Episode 165, Total Reward: 35.84001119816645
# Episode 166, Total Reward: -37.12631683031482
# Episode 167, Total Reward: -14.110603234440475
# Episode 168, Total Reward: 122.13914982807816
# Episode 169, Total Reward: -54.592665466954074
# Episode 170, Total Reward: -24.22122645165605
# Episode 171, Total Reward: -6.299663100551385
# Episode 172, Total Reward: -51.141068362305816
# Episode 173, Total Reward: 47.108396089998706
# Episode 174, Total Reward: -25.542764922518245
# Episode 175, Total Reward: -32.73659189942278
# Episode 176, Total Reward: 29.394904910801984
# Episode 177, Total Reward: -0.43303514951222155
# Episode 178, Total Reward: -0.7184658819842582
# Episode 179, Total Reward: -158.00242808955764
# Episode 180, Total Reward: 30.084162799758644
# Episode 181, Total Reward: -48.43486578356108
# Episode 182, Total Reward: 71.91717119143303
# Episode 183, Total Reward: -3.8865914081036124
# Episode 184, Total Reward: -36.13006765243776
# Episode 185, Total Reward: -15.724314365148288
# Episode 186, Total Reward: 3.5728266464666945
# Episode 187, Total Reward: -42.17304084628413
# Episode 188, Total Reward: 10.906386644121014
# Episode 189, Total Reward: -11.970632676822149
# Episode 190, Total Reward: -27.86470201552399
# Episode 191, Total Reward: -39.52299608181113
# Episode 192, Total Reward: -23.801088774802366
# Episode 193, Total Reward: -11.893663942580417
# Episode 194, Total Reward: -53.883269029664945
# Episode 195, Total Reward: -51.363880660701504
# Episode 196, Total Reward: 22.35223195067661
# Episode 197, Total Reward: 29.208270743682988
# Episode 198, Total Reward: -153.17525422626295
# Episode 199, Total Reward: -24.2906187718375
# Episode 200, Total Reward: -23.84461380090447
# Episode 201, Total Reward: -8.453111450940085
# Episode 202, Total Reward: -3.9300373432701434
# Episode 203, Total Reward: -40.90064780701175
# Episode 204, Total Reward: 27.727724191835208
# Episode 205, Total Reward: 30.61144513762656
# Episode 206, Total Reward: 22.250330768883614
# Episode 207, Total Reward: 42.03123283380302
# Episode 208, Total Reward: 7.253647941607738
# Episode 209, Total Reward: -8.323462291930667
# Episode 210, Total Reward: 11.87364586759064
# Episode 211, Total Reward: -27.72292303739421
# Episode 212, Total Reward: 18.045410892847855
# Episode 213, Total Reward: 79.13121774081407
# Episode 214, Total Reward: -37.59745884877506
# Episode 215, Total Reward: -7.593370809760401
# Episode 216, Total Reward: 1.0932122153192836
# Episode 217, Total Reward: 26.13468199888203
# Episode 218, Total Reward: -177.70122095694992
# Episode 219, Total Reward: -32.58616705100866
# Episode 220, Total Reward: 10.380115693629435
# Episode 221, Total Reward: 6.505872303881532
# Episode 222, Total Reward: 113.27824229249691
# Episode 223, Total Reward: 39.80137963937844
# Episode 224, Total Reward: -13.249192714470496
# Episode 225, Total Reward: 101.5388484021519
# Episode 226, Total Reward: -58.29223617649669
# Episode 227, Total Reward: 94.2619098691587
# Episode 228, Total Reward: 41.24947126371063
# Episode 229, Total Reward: 15.605377511471005
# Episode 230, Total Reward: 12.867235277495652
# Episode 231, Total Reward: -200.71334625860965
# Episode 232, Total Reward: 8.00853201188437
# Episode 233, Total Reward: -12.706413716691685
# Episode 234, Total Reward: -34.039902141335034
# Episode 235, Total Reward: 33.72002215949081
# Episode 236, Total Reward: -16.625841177298312
# Episode 237, Total Reward: 13.253233862058465
# Episode 238, Total Reward: 2.811432068415158
# Episode 239, Total Reward: 9.978234298025697
# Episode 240, Total Reward: -2.7033045461509744
# Episode 241, Total Reward: 6.426311719854837
# Episode 242, Total Reward: 135.04897596460697
# Episode 243, Total Reward: -17.30323742106286
# Episode 244, Total Reward: 104.85974341768987
# Episode 245, Total Reward: 88.43725355698886
# Episode 246, Total Reward: -96.80011568903602
# Episode 247, Total Reward: -14.686789458887375
# Episode 248, Total Reward: 86.51949096471479
# Episode 249, Total Reward: 28.405536088831354
# Episode 250, Total Reward: 48.53121936016487
# Episode 251, Total Reward: -9.268060293693978
# Episode 252, Total Reward: 90.94742899987557
# Episode 253, Total Reward: 137.8165179933895
# Episode 254, Total Reward: -12.964844745535004
# Episode 255, Total Reward: 0.49483758162497793
# Episode 256, Total Reward: 10.695862453485674
# Episode 257, Total Reward: 114.9231074770228
# Episode 258, Total Reward: 159.26845158687823
# Episode 259, Total Reward: 11.235512081620996
# Episode 260, Total Reward: 82.37284328751758
# Episode 261, Total Reward: -9.260466837007158
# Episode 262, Total Reward: -37.688152510128226
# Episode 263, Total Reward: 22.934343066943796
# Episode 264, Total Reward: 44.967078105773
# Episode 265, Total Reward: 38.36016260134727
# Episode 266, Total Reward: 34.908689286701616
# Episode 267, Total Reward: 37.96034740354108
# Episode 268, Total Reward: 60.22568176134783
# Episode 269, Total Reward: 39.00337963243983
# Episode 270, Total Reward: -31.526835856248326
# Episode 271, Total Reward: 132.60825021162924
# Episode 272, Total Reward: 31.689188437641235
# Episode 273, Total Reward: 158.37781523348792
# Episode 274, Total Reward: 52.80569126904463
# Episode 275, Total Reward: 97.6930110132432
# Episode 276, Total Reward: 89.91757644869698
# Episode 277, Total Reward: -24.66048863241562
# Episode 278, Total Reward: -5.551723320741587
# Episode 279, Total Reward: 104.24251160657877
# Episode 280, Total Reward: 28.674660476577316
# Episode 281, Total Reward: 78.51666763789318
# Episode 282, Total Reward: -5.775791047465788
# Episode 283, Total Reward: 59.143463898185615
# Episode 284, Total Reward: 76.76903590016295
# Episode 285, Total Reward: 46.07678788606531
# Episode 286, Total Reward: 93.36215426516048
# Episode 287, Total Reward: 16.71003448405102
# Episode 288, Total Reward: -215.024680927417
# Episode 289, Total Reward: 70.9259276236696
# Episode 290, Total Reward: 164.14541148699595
# Episode 291, Total Reward: 94.54281852533755
# Episode 292, Total Reward: 126.45746030233104
# Episode 293, Total Reward: 98.60464743046958
# Episode 294, Total Reward: 11.706703970804568
# Episode 295, Total Reward: 41.60258553998335
# Episode 296, Total Reward: 81.53986540633193
# Episode 297, Total Reward: 65.29424369527761
# Episode 298, Total Reward: -199.04141408566755
# Episode 299, Total Reward: 114.19359313282962
# Episode 300, Total Reward: 7.0674670474933805
# Episode 301, Total Reward: 30.06169670996502
# Episode 302, Total Reward: 82.95350830592461
# Episode 303, Total Reward: 137.16964551419514
# Episode 304, Total Reward: -9.943569616406549
# Episode 305, Total Reward: 64.55369870978939
# Episode 306, Total Reward: 162.3219932487027
# Episode 307, Total Reward: 85.98167604670871
# Episode 308, Total Reward: 25.47377242589795
# Episode 309, Total Reward: -27.31031865631644
# Episode 310, Total Reward: 20.20894951805799
# Episode 311, Total Reward: 52.24168528484506
# Episode 312, Total Reward: 60.78149056844063
# Episode 313, Total Reward: 60.49568788431173
# Episode 314, Total Reward: 16.43523119703903
# Episode 315, Total Reward: 129.12729326443417
# Episode 316, Total Reward: 115.49759355156536
# Episode 317, Total Reward: 49.685197468260924
# Episode 318, Total Reward: 48.53682828154897
# Episode 319, Total Reward: 28.55205065408174
# Episode 320, Total Reward: 18.82736017920041
# Episode 321, Total Reward: 72.05720019147121
# Episode 322, Total Reward: 95.34852784266288
# Episode 323, Total Reward: 90.31123431483131
# Episode 324, Total Reward: -6.136428299014412
# Episode 325, Total Reward: -20.236631282106444
# Episode 326, Total Reward: 45.69062070758997
# Episode 327, Total Reward: 79.8592976614471
# Episode 328, Total Reward: 127.46047624986815
# Episode 329, Total Reward: 55.419295192730736
# Episode 330, Total Reward: -184.5916904564163
# Episode 331, Total Reward: 76.26356169825097
# Episode 332, Total Reward: 115.72228332321288
# Episode 333, Total Reward: 57.497269308218605
# Episode 334, Total Reward: 15.020491954630439
# Episode 335, Total Reward: 19.316391332028985
# Episode 336, Total Reward: 77.90390814396493
# Episode 337, Total Reward: -27.60216981038765
# Episode 338, Total Reward: 34.50849879536918
# Episode 339, Total Reward: 59.042548134799375
# Episode 340, Total Reward: 71.28898352123417
# Episode 341, Total Reward: -24.768940247895117
# Episode 342, Total Reward: 70.62697953834258
# Episode 343, Total Reward: 5.44524191429862
# Episode 344, Total Reward: 79.62867964641725
# Episode 345, Total Reward: 121.1613101384256
# Episode 346, Total Reward: 45.56626363493278
# Episode 347, Total Reward: 114.40777573676345
# Episode 348, Total Reward: 82.37261831154285
# Episode 349, Total Reward: 11.94451425780042
# Episode 350, Total Reward: 79.33895373499604
# Episode 351, Total Reward: 111.4138700812364
# Episode 352, Total Reward: 23.340775899770435
# Episode 353, Total Reward: 126.07171845601444
# Episode 354, Total Reward: 71.29487972334717
# Episode 355, Total Reward: 94.7546810349611
# Episode 356, Total Reward: 68.93797177089718
# Episode 357, Total Reward: 120.0996871724657
# Episode 358, Total Reward: 136.89189350224203
# Episode 359, Total Reward: 92.90996604719452
# Episode 360, Total Reward: 62.89124444488347
# Episode 361, Total Reward: 101.13166197160287
# Episode 362, Total Reward: 127.09576843745275
# Episode 363, Total Reward: 129.32187308078866
# Episode 364, Total Reward: 34.21218570140937
# Episode 365, Total Reward: 76.65187555960887
# Episode 366, Total Reward: -31.885620990216296
# Episode 367, Total Reward: 126.03746349919084
# Episode 368, Total Reward: 125.65100478766944
# Episode 369, Total Reward: 140.73537147665644
# Episode 370, Total Reward: 120.33517554728343
# Episode 371, Total Reward: 110.24091560650855
# Episode 372, Total Reward: 181.64656533815543
# Episode 373, Total Reward: 116.51040174896954
# Episode 374, Total Reward: -28.561707094852565
# Episode 375, Total Reward: 72.25088126430258
# Episode 376, Total Reward: 21.7989718177613
# Episode 377, Total Reward: 101.47710045512527
# Episode 378, Total Reward: 97.5159894321356
# Episode 379, Total Reward: 1.257272418601313
# Episode 380, Total Reward: 100.87921141510405
# Episode 381, Total Reward: 91.88739099856299
# Episode 382, Total Reward: 53.9806939549435
# Episode 383, Total Reward: 132.48703092262224
# Episode 384, Total Reward: 112.16761796669722
# Episode 385, Total Reward: 144.07261411522
# Episode 386, Total Reward: 74.14700820061339
# Episode 387, Total Reward: 104.24675389651738
# Episode 388, Total Reward: 40.333120786943354
# Episode 389, Total Reward: -2.7153016308316777
# Episode 390, Total Reward: 89.15526557288959
# Episode 391, Total Reward: 36.01566917617319
# Episode 392, Total Reward: 132.04427226287305
# Episode 393, Total Reward: 89.15376064224995
# Episode 394, Total Reward: 114.29913027617143
# Episode 395, Total Reward: 120.50894342403134
# Episode 396, Total Reward: 119.44379321116894
# Episode 397, Total Reward: 82.59963934361193
# Episode 398, Total Reward: 146.89137703145462
# Episode 399, Total Reward: -4.938977183727914
# Episode 400, Total Reward: 34.67625180260714
# Episode 401, Total Reward: 56.52237938964919
# Episode 402, Total Reward: 94.47947672882508
# Episode 403, Total Reward: 57.97089144216508
# Episode 404, Total Reward: 108.95811664157335
# Episode 405, Total Reward: 93.57906602358818
# Episode 406, Total Reward: 112.57212909385085
# Episode 407, Total Reward: 23.236339907881046
# Episode 408, Total Reward: 119.78573718504003
# Episode 409, Total Reward: 99.71417365063648
# Episode 410, Total Reward: 10.555132998647395
# Episode 411, Total Reward: 100.79778975208725
# Episode 412, Total Reward: 99.41620803555875
# Episode 413, Total Reward: 53.950142251044376
# Episode 414, Total Reward: 35.037535475346424
# Episode 415, Total Reward: 78.7358808287711
# Episode 416, Total Reward: 2.50571643794261
# Episode 417, Total Reward: 113.07011116873052
# Episode 418, Total Reward: 97.19006330065494
# Episode 419, Total Reward: 114.47140404911211
# Episode 420, Total Reward: 124.89614836948476
# Episode 421, Total Reward: 5.754622006058071
# Episode 422, Total Reward: 64.17770709539602
# Episode 423, Total Reward: 46.764352477906925
# Episode 424, Total Reward: 41.97673856302835
# Episode 425, Total Reward: 213.334561353269
# Episode 426, Total Reward: 86.2557895117447
# Episode 427, Total Reward: 110.30272676167387
# Episode 428, Total Reward: 76.70713794216579
# Episode 429, Total Reward: 84.62052279590715
# Episode 430, Total Reward: 111.24697704010141
# Episode 431, Total Reward: 130.87573878759721
# Episode 432, Total Reward: 150.90174082383865
# Episode 433, Total Reward: 100.47721869410718
# Episode 434, Total Reward: 143.90360279611
# Episode 435, Total Reward: 37.44300263245427
# Episode 436, Total Reward: 127.03965071573828
# Episode 437, Total Reward: 79.54746327609745
# Episode 438, Total Reward: 122.7638638799965
# Episode 439, Total Reward: 109.0482776480523
# Episode 440, Total Reward: 98.95487113015076
# Episode 441, Total Reward: 144.00889174908747
# Episode 442, Total Reward: 122.26579635628568
# Episode 443, Total Reward: 101.6873105054449
# Episode 444, Total Reward: 102.29157551994827
# Episode 445, Total Reward: 53.68755706458747
# Episode 446, Total Reward: 32.402672083366184
# Episode 447, Total Reward: 18.98682018005676
# Episode 448, Total Reward: 159.33961429484032
# Episode 449, Total Reward: 70.93505651937281
# Episode 450, Total Reward: 78.33786066440229
# Episode 451, Total Reward: 142.54078245766416
# Episode 452, Total Reward: 31.21813826645206
# Episode 453, Total Reward: 71.18139626250647
# Episode 454, Total Reward: -150.99019848234485
# Episode 455, Total Reward: 114.38644454308732
# Episode 456, Total Reward: 65.32951796647107
# Episode 457, Total Reward: 78.1224988714656
# Episode 458, Total Reward: 156.11442967948517
# Episode 459, Total Reward: 125.80796794835592
# Episode 460, Total Reward: 91.79773703344505
# Episode 461, Total Reward: 153.39850960234176
# Episode 462, Total Reward: 128.48058500521836
# Episode 463, Total Reward: 112.91761462389849
# Episode 464, Total Reward: 83.85529263405763
# Episode 465, Total Reward: 112.05761950977852
# Episode 466, Total Reward: 77.74479435409081
# Episode 467, Total Reward: 123.48569263904957
# Episode 468, Total Reward: 132.46477579991063
# Episode 469, Total Reward: 136.23909708619118
# Episode 470, Total Reward: 35.14062458562972
# Episode 471, Total Reward: 114.32386860925808
# Episode 472, Total Reward: 123.40498161067241
# Episode 473, Total Reward: 14.164737550442108
# Episode 474, Total Reward: 147.31574251087892
# Episode 475, Total Reward: 74.58857037856406
# Episode 476, Total Reward: 131.62996096346296
# Episode 477, Total Reward: 81.97062316114977
# Episode 478, Total Reward: 43.797097572845274
# Episode 479, Total Reward: 121.36160415270729
# Episode 480, Total Reward: 154.3046995508979
# Episode 481, Total Reward: -78.77358145214257
# Episode 482, Total Reward: 146.51146607127106
# Episode 483, Total Reward: 99.90403756221696
# Episode 484, Total Reward: 101.28023913770872
# Episode 485, Total Reward: 75.46812268388507
# Episode 486, Total Reward: 45.94073642543725
# Episode 487, Total Reward: 162.85977446475408
# Episode 488, Total Reward: 126.75971216807274
# Episode 489, Total Reward: 132.66278636488994
# Episode 490, Total Reward: -52.02063139111326
# Episode 491, Total Reward: 52.00209101470103
# Episode 492, Total Reward: 114.24040397834783
# Episode 493, Total Reward: 85.4753350891554
# Episode 494, Total Reward: 103.25482738178198
# Episode 495, Total Reward: 86.65143028842112
# Episode 496, Total Reward: 95.18913887482321
# Episode 497, Total Reward: 109.31301407119949
# Episode 498, Total Reward: 98.84312383900425
# Episode 499, Total Reward: 59.89703039051626
# Episode 500, Total Reward: 66.99724347652008
# 

## --- emacspy-machine-learning  master @ tensorboard --logdir=runs => rl_gym_dqn_lunar_tensorboard_log.png
## du -sh dqn_model.lmdb => 424K	dqn_model.lmdb

