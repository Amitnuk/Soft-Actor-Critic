# Soft-Actor-Critic

I Coded SAC to train the Vehicles in [Test Drive Unlimited: Solar Crown](https://store.steampowered.com/agecheck/app/1249970/)
This is just the base code used, it only attempts to [reproduces the SAC paper](https://arxiv.org/abs/1801.01290).
You will find checkpoints for *Ant-v5, Bipedalwalker-v3, HalfCheetah-v5, InvertedPendulum-v5, LunarLander-v3* 


# Usage 

* python3 main.py (evaluates Lunar Lander Continuous, change main.py to train or even evaluating another env)

# TODO
- [ ] Restructuring
- [ ] Adding results(graphs, images, etc) and a conclusions.
- [ ] Automate Training and Evaluation (command line)
- [ ] Show the results on the Test Drive Unlimited: Solar Crown
- [ ] And More


# Notes
SAC fails to solve bipedalwalker
