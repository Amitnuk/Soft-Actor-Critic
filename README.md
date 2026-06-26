# Soft-Actor-Critic

I Coded SAC to train the Vehicles in [Test Drive Unlimited: Solar Crown](https://store.steampowered.com/agecheck/app/1249970/).

This repo its only has the base code, it reproduces [SAC](https://arxiv.org/abs/1801.01290).
You will find checkpoints for *Ant-v5, Bipedalwalker-v3, HalfCheetah-v5, InvertedPendulum-v5, LunarLander-v3* etc 


# Usage 
## Evaluation
* python3 main.py 
## Training
* python3 main.py --training_mode

# TODO
- [ ] Restructuring
- [ ] Adding results(graphs, images, etc) and a conclusions.
- [X] Automate Training and Evaluation (command line)
- [ ] Show the results on the Test Drive Unlimited: Solar Crown
- [ ] And More


# Notes
SAC fails to solve bipedalwalker
