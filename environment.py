import numpy as np

class CustomEnvironment:
    def __init__(self):
        self.state_size = 4  # Stato simulato con 4 dimensioni
        self.action_size = 2  # Due possibili azioni
        self.state = None
        self.steps = 0
        self.max_steps = 200  # Numero massimo di passi in un episodio

    def reset(self):
        """Resetta l'ambiente all'inizio di un nuovo episodio"""
        self.state = np.random.uniform(-1, 1, self.state_size)
        self.steps = 0
        return self.state

    #Questa dovrebbe essere come update e venir chiamata dentro simulate
    def step(self, action):
        """Avanza di uno step dato un'azione"""
        # Simuliamo una transizione arbitraria dello stato
        next_state = self.state + np.random.uniform(-0.05, 0.05, self.state_size)

        # Ricompensa basata su un'azione (esempio semplice)
        reward = 1 if action == np.argmax(next_state) else -1

        # L'episodio termina dopo un certo numero di passi
        self.steps += 1
        done = self.steps >= self.max_steps
        return next_state, reward, done