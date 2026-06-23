

import gymnasium as gym 
from Agent import SACAgent
from Models import SACActor, SACCritic
from ReplayBuffer import ReplayBuffer
import torch 
from torch.optim import Adam


if __name__ == "__main__" :

    print(f"{gym.pprint_registry()}")
    print(f"GPU Available :{torch.cuda.is_available()}")
    print(f"Pytorch Version :{torch.__version__}")

    env_name = ["LunarLander-v3",'BipedalWalker-v3', "Ant-v5", 'HalfCheetah-v5'][0]
    if env_name == "LunarLander-v3" :
        Env = gym.make(env_name, continuous=True,render_mode="human")
    else :
        Env = gym.make(env_name,render_mode="human")

    action_bounds = Env.action_space.low, Env.action_space.high
    print(f"action_bounds ={action_bounds}")
    
    replay_buffer_fn = lambda : ReplayBuffer(max_size=1000000, batch_size=64)


    policy_model_fn  =   lambda nS, bounds, : SACActor(nS, bounds,entropy_lr=0.0003, hidden_dims=(256,256))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda network, learning_rate: Adam(params=network.parameters(),lr=learning_rate)
    policy_optimizer_lr = 0.0003

    value_model_fn  =  lambda nS, nA : SACCritic(nS, nA, hidden_dims=(256,256))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda network, learning_rate: Adam(params=network.parameters(),lr=learning_rate)
    value_optimizer_lr = 0.0003  


    max_episodes = 40000
    n_warmup_batches = 20
    tau = 0.005
    gamma = 0.99
    target_update_interval=1
    
    agent = SACAgent(
                     Env=Env, 
                     policy_model=policy_model_fn, 
                     policy_optimizer=policy_optimizer_fn,
                     policy_optimizer_lr=policy_optimizer_lr,
                     policy_max_grad_norm=policy_max_grad_norm,
                     value_model=value_model_fn, 
                     value_optimizer=value_optimizer_fn,
                     value_optimizer_lr=value_optimizer_lr,
                     value_max_grad_norm=value_max_grad_norm,
                     replay_buffer = replay_buffer_fn, 
                     max_episodes= max_episodes, 
                     n_warmup_batches=n_warmup_batches,
                     tau=tau,
                     gamma=gamma,
                     target_update_interval=target_update_interval,
                     env_name=env_name )
    
    #agent.basic_setup()
    path = "./checkpoint/sac_checkpoint_" + env_name + ".pth"
    #agent.train(ckpt_path=path, finetune=False)
    agent.load_checkpoint(ckpt_path="./checkpoint/sac_checkpoint_" + env_name + ".pth", evaluate=True)
    agent.evaluate(collect_data=False)
    

    



