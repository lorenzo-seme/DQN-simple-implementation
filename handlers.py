import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory_size = 2000
        self.model = self._build_model()

    def _build_model(self):
        # Create a simple neural network model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    # il nostro handler, dato lo stato decide l'azione
    def act(self, state):
        # Decide whether to explore or exploit
        state = np.array(state)  # Ensure state is an array
        if len(state.shape) == 1:
            state = np.reshape(state, [1, self.state_size])  # Reshape state if necessary

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        # Train the model using a batch of experiences
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            state = np.array(state)  # Ensure state is an array
            next_state = np.array(next_state)  # Ensure next_state is an array

            if len(state.shape) == 1:
                state = np.reshape(state, [1, self.state_size])
            if len(next_state.shape) == 1:
                next_state = np.reshape(next_state, [1, self.state_size])

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            q_values = self.model.predict(state, verbose=0)
            q_values[0][action] = target
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

