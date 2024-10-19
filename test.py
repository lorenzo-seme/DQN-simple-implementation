import gym
from handlers import *

def train_dqn(episodes=1000):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for episode in range(episodes):
        state, _ = env.reset()  # Estrarre solo l'array delle osservazioni, ignorando il secondo elemento (dizionario)

        # Ora lo stato è solo l'array di osservazioni
        print(f"Episode {episode + 1}, Original state from env.reset(): {state}")

        state = np.reshape(state, [1, state_size])

        total_reward = 0

        for time in range(500):
            # Uncomment to render the environment (useful for debugging)
            # env.render()

            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Anche qui stampiamo lo stato successivo
            print(f"Next state: {next_state}")

            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done or truncated:
                print(f"Episode: {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            agent.replay()


if __name__ == "__main__":
    train_dqn(episodes=500)
