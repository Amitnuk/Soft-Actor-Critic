

import gymnasium as gym 
from Agent import SACAgent
from Models import SACActor, SACCritic
from ReplayBuffer import ReplayBuffer
import torch 
from torch.optim import Adam
import argparse

if __name__ == "__main__" :



    parser = argparse.ArgumentParser(description="Soft Actor Critic Algorithm") 
    parser.add_argument("--env_index",type=int, choices=[0,1,2,3], default=0,help="The environment to choose")
    parser.add_argument("--training_mode", action="store_true", help="The mode, if not specified evaluation mode is called")
    parser.add_argument("--max_size", type=int,default=1000000 ,help="The size of the replay buffer, 1000000 is default")
    parser.add_argument("--batch_size",  type=int,default=64 ,help="The size of batch, 64 is default")
    parser.add_argument("--hidden_units", type=int,default=256 ,help="Hidden units, default is 256")
    parser.add_argument("--policy_lr", type=float,default=0.0003 ,help="The policy learning rate, default is 0.0003")
    parser.add_argument("--value_lr", type=float, default=0.0003, help="The Value function learning rate, default is 0.0003")
    parser.add_argument("--entopy_lr", type=float, default=0.0003, help="The entropy learning rate, default is 0.0003")
    
    parser.add_argument("--max_episodes",  type=int,default=40000 ,help="The number of episodes for training, 40000 is the default")
    parser.add_argument("--n_warmup_batches", type=int,default=20 ,help="number of warmup batches, default is 20")
    parser.add_argument("--tau", type=float,default=0.005 ,help="tau polyak avering ?, default is 0.005")
    parser.add_argument("--gamma", type=float,default=0.99 ,help="gamma, discounting, default is 0.99")
    parser.add_argument("--target_update_interval", type=int,default=1 ,help="interval for updating, default is 1")


   
    args = parser.parse_args() 

    env_index= args.env_index
    
    print(f"{gym.pprint_registry()}")
    print(f"GPU Available :{torch.cuda.is_available()}")
    print(f"Pytorch Version :{torch.__version__}")

    env_name = ["LunarLander-v3",'BipedalWalker-v3', "Ant-v5", 'HalfCheetah-v5'][env_index]
    if env_name == "LunarLander-v3" :
        Env = gym.make(env_name, continuous=True,render_mode="human")
    else :
        Env = gym.make(env_name,render_mode="human")

    action_bounds = Env.action_space.low, Env.action_space.high
    print(f"action_bounds ={action_bounds}")
    
    replay_buffer_fn = lambda : ReplayBuffer(max_size=args.max_size, batch_size=args.batch_size)


    policy_model_fn  =   lambda nS, bounds, : SACActor(nS, bounds,entropy_lr=args.entopy_lr, hidden_dims=(args.hidden_units,args.hidden_units))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda network, learning_rate: Adam(params=network.parameters(),lr=learning_rate)
    policy_optimizer_lr = args.policy_lr

    value_model_fn  =  lambda nS, nA : SACCritic(nS, nA, hidden_dims=(args.hidden_units,args.hidden_units))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda network, learning_rate: Adam(params=network.parameters(),lr=learning_rate)
    value_optimizer_lr = args.value_lr 


    max_episodes = args.max_episodes
    n_warmup_batches = args.n_warmup_batches
    tau = args.tau
    gamma = args.gamma
    target_update_interval=args;target_update_interval
    
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

    if args.training_mode :
        print("Training")
        #agent.train(ckpt_path=path, finetune=False)
        #agent.load_checkpoint(ckpt_path="./checkpoint/sac_checkpoint_" + env_name + ".pth", evaluate=True)
    else :
        print("Evaluating")
        agent.load_checkpoint(ckpt_path="./checkpoint/sac_checkpoint_" + env_name + ".pth", evaluate=True)
        agent.evaluate(collect_data=False)
    

    



