import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(self, env, lr=0.2, disc=0.99, eps=1.0):
        self.a_space = env.action_space
        self.qtab = np.zeros((env.observation_space.n, env.action_space.n))
        self.lr = lr
        self.disc = disc
        self.eps = eps

    def act(self, s):
        if np.random.random() < self.eps:
            return self.a_space.sample()
        return int(np.argmax(self.qtab[s]))

    def step(self, s, a, r, s2, done=False):
        best_future = np.max(self.qtab[s2])
        target = r if done else (r + self.disc * best_future)
        self.qtab[s, a] = self.qtab[s, a] + self.lr * (target - self.qtab[s, a])


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")

    rounds = 4000
    max_steps = 200
    bot = QLearningAgent(env, lr=0.2, disc=0.99, eps=1.0)

    for ep in range(rounds):
        st, _ = env.reset()
        total = 0

        for _ in range(max_steps):
            mv = bot.act(st)
            st2, pts, terminated, truncated, _ = env.step(mv)
            done = terminated or truncated

            bot.step(st, mv, pts, st2, done=done)

            total += pts
            st = st2

            if done:
                break

        bot.eps = max(0.01, bot.eps * 0.999)

        if ep % 200 == 0:
            print(f"Ep {ep} | retorno: {total} | eps: {bot.eps:.3f}")

    env.close()

    env2 = gym.make("CliffWalking-v1", render_mode="human")
    st, _ = env2.reset()

    bot.eps = 0.0  

    for _ in range(200):
        mv = bot.act(st)
        st, _, terminated, truncated, _ = env2.step(mv)
        if terminated or truncated:
            break

    env2.close()
