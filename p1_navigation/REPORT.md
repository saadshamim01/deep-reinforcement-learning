##Project Navigation Report

###Code
 
For training the agent, we used Deep-Q-Learning network, the code is present in dqn_agent.py, model.py, Navigation_Pixels.ipynb.

In model.py, we created a neural network to train the agent, first and second layer both contained 64 units. Input layers takes the state size and output returns the best action once the neural network is completely trained.
  
```
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

The class Agent is is the well-known class implementing the following mechanisms:

**Two Q-Networks (local and target) using the simple neural network.**
```
  self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
  self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
```
**Replay memory (using the class ReplayBuffer)**

```
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
```

**Epsilon-greedy mechanism**

```
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
```

**Epsilon**

The epsilon become a bit smaller with each episode:
```
  eps = max(eps_end, eps_decay*eps)
```
where eps_end=0.01, eps_decay = 0.996.

**Loss Function**

Minimize the loss by gradient descend mechanism using the ADAM optimizer

```
 loss = F.mse_loss(Q_expected, Q_targets)
```

---

###Model Q-Network

The code for `QNetwork` is written in PyTorch and implemented in model.py. Q-Network consist is a neural network which contains 3 fully connected layer with 2 layer rectified non linear layers.
The layers are constructed in the following way:

- Layer fc1 has 64 neurons and it maps state_size x fc1_units
- Layer fc12 has 64 neurons and it maps fc1_units x fc2_units
- Layer fc1 has 64 neurons and it maps fc2_units x input parameters

where state_size = 37, action_size = 8, fc1_units and fc2_units are the input parameters.


###Training & Testing

We run 5 training sessions with different parameters fc1_units, fc2_units, eps_start, and we save obtained weights by the function of PyTorch:
```
file_weights = 'weights_'+str(train_n)+'.trn'
```

After that we go to the testing session. For each session, we load saved weights by the function of PyTorch as follows:

```
agent.qnetwork_local.load_state_dict(torch.load(file_weights))
torch.save(agent.qnetwork_local.state_dict(), 'weights_'+str(train_numb)+'.trn') 
```


Train: 0, Test: 0, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 17.0

Train: 0, Test: 1, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 14.0

Train: 0, Test: 2, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 12.0

Train: 0, Test: 3, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 9.0

Train: 0, Test: 4, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 15.0

Train: 0, Test: 5, Episode: 561, fc1_units: 48, fc2_units: 56, eps_start: 0.993, Score: 20.0

       Average Score:  14.5
=========================================================


Train: 1, Test: 0, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 11.0

Train: 1, Test: 1, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 11.0

Train: 1, Test: 2, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 19.0

Train: 1, Test: 3, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 13.0

Train: 1, Test: 4, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 11.0

Train: 1, Test: 5, Episode: 651, fc1_units: 112, fc2_units: 104, eps_start: 0.991, Score: 12.0

       Average Score:  12.833333333333334
=========================================================


Train: 2, Test: 0, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 16.0

Train: 2, Test: 1, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 18.0

Train: 2, Test: 2, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 14.0

Train: 2, Test: 3, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 16.0

Train: 2, Test: 4, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 13.0

Train: 2, Test: 5, Episode: 600, fc1_units: 80, fc2_units: 80, eps_start: 0.991, Score: 16.0

       Average Score:  15.5
=========================================================

Train: 3, Test: 0, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 20.0

Train: 3, Test: 1, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 10.0

Train: 3, Test: 2, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 19.0

Train: 3, Test: 3, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 11.0

Train: 3, Test: 4, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 16.0

Train: 3, Test: 5, Episode: 525, fc1_units: 64, fc2_units: 56, eps_start: 0.991, Score: 23.0

       Average Score:  16.5
=========================================================


Train: 4, Test: 0, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 7.0

Train: 4, Test: 1, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 17.0

Train: 4, Test: 2, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 14.0

Train: 4, Test: 3, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 18.0

Train: 4, Test: 4, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 8.0

Train: 4, Test: 5, Episode: 572, fc1_units: 64, fc2_units: 56, eps_start: 0.988, Score: 16.0

       Average Score:  13.333333333333334
=========================================================


- Solved Using Local Environment: After running the Deep-Q-Learning, we usually reached the desired average score greater than 13 within 650 epochs. 


###Future ideas

The performance of the Deep-Q-Learning algorithm can be improveed by following ways:

- Adding one or more non linear (kernel) layers to neural network.
- Further adjusting fc1_units, fc2_units and epsilon parameters.


Q-Learning algorithm are prone to overestimation error, there are other version of Deep-Q-Learning which fix this issue like:

- Dueling Double Deep Neural Network
- Prioritized Experience Replay
- Fixed Q-Target


The recent achievement
Open AI group to play Dota 2 using Reinforcement Learning. They have created a bot which beats the world’s top professionals at 1v1 matches of Dota 2 under standard tournament rules. The bot learned the game from scratch by self-play, and does not use imitation learning or tree search. This is a step towards building AI systems which accomplish well-defined goals in messy, complicated situations involving real humans.


