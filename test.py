from tqdm import trange
from handlers import *
from environment import *

def train_dqn(episodes=1000):
    env = CustomEnvironment()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    with trange(episodes, desc="Training DQN Agent", position=0, leave=True) as pbar:
        #pbar.set_description("Training DQN Agent")

        for episode in pbar:
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            total_reward = 0

            with trange(500, desc=f"Episode {episode + 1}/{episodes}", position=0, leave=False) as pbar_inside:
                for time in pbar_inside:
                    # Uncomment to render the environment (useful for debugging)
                    # env.render()

                    action = agent.act(state)
                    next_state, reward, done = env.step(action)

                    next_state = np.reshape(next_state, [1, state_size])

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward

                    if done:
                        break

                    agent.replay()

                pbar.set_postfix({"Total reward": total_reward})


if __name__ == "__main__":
    train_dqn(episodes=2)
