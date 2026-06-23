from Models import SACActor, SACCritic
from ReplayBuffer import ReplayBuffer
from gymnasium import Env
import numpy as np
from itertools import count
import torch
from torch.optim import Adam
import os

class SACAgent :
    
    def __init__(self, Env:Env, 
                 policy_model:object, 
                 policy_optimizer:object,
                 policy_optimizer_lr:float,
                 policy_max_grad_norm:float,
                 value_model:object, 
                 value_optimizer:object,
                 value_optimizer_lr:float,
                 value_max_grad_norm:float,
                 replay_buffer:object, 
                 max_episodes:int= 1000, 
                 n_warmup_batches:int=1,
                 tau:float=0.001,
                 gamma:float=0.99,
                 target_update_interval:int=1,
                 env_name:str="" ) :
        
        self.Env = Env

        self.policy_model_fn = policy_model 
        self.policy_optimizer_fn = policy_optimizer
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_max_grad_norm = policy_max_grad_norm

        self.value_model_fn = value_model
        self.value_optimizer_fn = value_optimizer
        self.value_optimizer_lr = value_optimizer_lr
        self.value_max_grad_norm=value_max_grad_norm

        self.replay_buffer_fn = replay_buffer

        
        self.max_episodes =  max_episodes
        self.n_warmup_batches = n_warmup_batches
        self.tau = tau
        self.gamma = gamma
        self.target_update_interval=target_update_interval
        self.nb_updates = 0
        self.first_checkpoint_save = True
        self.env_name = env_name
        self.ckpt_path = ""
        self.reward = float('-inf')
        self.automatic_entropy_tuning = False

        nS, nA = self.Env.observation_space.shape[0], self.Env.action_space.shape[0]
        action_bounds = (self.Env.action_space.low, self.Env.action_space.high)

        # actor 
        self.policy_model:SACActor = self.policy_model_fn(nS, action_bounds)
        self.policy_optimizer:Adam = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)
        
      
        #critic 1
        self.target_value_model_a:SACCritic = self.value_model_fn(nS, nA)
        self.online_value_model_a:SACCritic = self.value_model_fn(nS, nA)
        self.value_optimizer_a:Adam         = self.value_optimizer_fn(self.online_value_model_a, self.value_optimizer_lr)
        
        #critic 2
        self.target_value_model_b:SACCritic = self.value_model_fn(nS, nA)
        self.online_value_model_b:SACCritic = self.value_model_fn(nS, nA)
        self.value_optimizer_b:Adam         = self.value_optimizer_fn(self.online_value_model_b, self.value_optimizer_lr)
    
        # replay buffer
        self.replay_buffer:ReplayBuffer     = self.replay_buffer_fn()
        

    def interaction_step(self, state:np.ndarray) -> tuple: 
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        if len(self.replay_buffer) > min_samples :
            action = self.policy_model.select_action(state)
        else :
            action = self.policy_model.select_random_action(state)

        
        next_state, reward, is_terminal, truncated, info = self.Env.step(action) 
       
        is_terminated = is_terminal or truncated
        experience = ( state, action, reward, next_state, is_terminal)
        self.replay_buffer.store( experience )
        return next_state, is_terminated
    

    def optimize_model(self, experiences:tuple) -> None :
        
        
        states, actions, rewards, next_states, terminals = experiences
        #torch.autograd.set_detect_anomaly(True)
     
        # Q loss
        with torch.no_grad():

            next_actions, next_log_pi = self.policy_model.full_pass(state=next_states)
            self.policy_model.alpha = torch.max( self.policy_model.alpha, torch.tensor(float("-inf")))
            next_q_sa_a = self.target_value_model_a(state=next_states, action=next_actions)
            next_q_sa_b = self.target_value_model_b(state=next_states, action=next_actions)
            next_q_sa   = torch.min(next_q_sa_a,next_q_sa_b)
            target_q_sa = rewards + self.gamma*(1 - terminals)*( next_q_sa - self.policy_model.alpha*next_log_pi )
        
        q_sa_a  = self.online_value_model_a(state=states, action=actions)
        q_sa_b  = self.online_value_model_b(state=states, action=actions)
        # gradient descent  : JQ 1,2 = 𝔼(st,at)~D[0.5(Q 1, 2(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qa_loss = ( target_q_sa - q_sa_a).pow(2).mul(0.5).mean()
        qb_loss = ( target_q_sa - q_sa_b).pow(2).mul(0.5).mean()

        self.value_optimizer_a.zero_grad()
        qa_loss.backward()
        torch.nn.utils.clip_grad_norm_( self.online_value_model_a.parameters(), self.value_max_grad_norm )
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        qb_loss.backward()
        torch.nn.utils.clip_grad_norm_( self.online_value_model_b.parameters(), self.value_max_grad_norm )
        self.value_optimizer_b.step()
        
        # policy loss 
       
        current_actions, log_pi = self.policy_model.full_pass(state=states)
        current_q_sa_a = self.online_value_model_a(state=states, action=current_actions )
        current_q_sa_b = self.online_value_model_b(state=states, action=current_actions )
        current_q_sa   = torch.min(current_q_sa_a, current_q_sa_b )
        # gradient ascent --> Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − min 1,2 {Q(st,f(εt;st))}] 
        policy_loss    = ( self.policy_model.alpha * log_pi - current_q_sa ).mean() 

        self.policy_optimizer.zero_grad()
        policy_loss.backward() 
        torch.nn.utils.clip_grad_norm_( self.policy_model.parameters(), self.policy_max_grad_norm )
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning : 
            
            alpha_loss = -( self.policy_model.log_alpha*(log_pi + self.policy_model.target_entropy).detach()).mean()
            self.policy_model.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.policy_model.alpha_optimizer.step()
            self.policy_model.alpha = self.policy_model.log_alpha.exp()
        
            
        

    
    def models_equal(self, model1, model2, tol=1e-6):
        for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(model1_param.data, model2_param.data, atol=1e-6)
      
    def update_target_network(self, tau:float=None) :
        #polyak averaging 
        tau = self.tau if tau is None else tau
        # critic a 
        for target_a, online_a in zip(self.target_value_model_a.parameters(), self.online_value_model_a.parameters()) :
            target_a.data.copy_( target_a.data * (1 - tau) + online_a.data*tau)

        # critic b 
        for target_b, online_b in zip(self.target_value_model_b.parameters(), self.online_value_model_b.parameters()) :
            target_b.data.copy_( target_b.data * (1 - tau) + online_b.data*tau)
             
    def train(self, ckpt_path="", finetune=False) -> None:

        if ckpt_path != "" and finetune :
            self.load_checkpoint(ckpt_path=ckpt_path, evaluate=False)

        
        self.update_target_network(tau=1.0)
        self.models_equal(self.target_value_model_a, self.online_value_model_a)
        self.models_equal(self.target_value_model_b, self.online_value_model_b)
        
        self.steps = 0
        for i in range( 1, self.max_episodes + 1) : 
            state, _  = self.Env.reset()
            is_terminal = False
            #print("train :", i)
            for step in count() :
                self.steps += 1
                state, is_terminal = self.interaction_step(state=state)
                
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples :
                    experiences = self.replay_buffer.sample()
                    experiences = self.policy_model.load(experiences=experiences)
                    self.optimize_model(experiences=experiences)
                    self.nb_updates += 1
                    
                if  self.nb_updates != 0 :
                    self.update_target_network()
                
                if is_terminal :
                    break

            if i == 50 :
                self.automatic_entropy_tuning = True
                #self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
                #self.policy_model.alpha = torch.tensor(1.0).to(self.device)
            if  self.nb_updates and i % 10 == 0 and i >= 50 :
                final_eval_score, score_std = self.evaluate(collect_data=False)        
                self.policy_model.train()     

            print(f"episode :{i} steps :{self.steps}, replayBuffer Size :{len(self.replay_buffer)}")
            

    def evaluate(self, suffix=".pth", collect_data=False):
        #rewards = []
        #print("eval")
        self.policy_model.eval()
        state, _ = self.Env.reset()
        rewards=0
        for _ in count():
            action = self.policy_model.select_greedy_action(state)
            next_state, reward, terminated, truncated,_ = self.Env.step(action)
            rewards += reward
            is_terminated = terminated or truncated
            if collect_data :
                experience = ( state, action, reward, next_state, terminated)
                self.replay_buffer.store( experience )
            state = next_state
            
            if is_terminated: 
                break
         
        if self.first_checkpoint_save  :
            self.save_checkpoint(env_name=self.env_name, reward=np.mean(rewards), suffix=suffix)
            self.first_checkpoint_save =  False
            self.reward = rewards
            print(f"{self.reward,np.std(rewards) }")   

        else :
            
            
            if self.reward < rewards :
                self.save_checkpoint(env_name=self.env_name, reward=np.mean(rewards), suffix=suffix)
                self.reward = rewards
                print(f"{self.reward,np.std(rewards) }")      
        
        return rewards, np.std(rewards)


    def save_checkpoint(self, env_name, reward, suffix="", ckpt_path=None) :

        if not os.path.exists("./checkpoint/") :
            os.makedirs("/home/kuntima/Workspace/UniverseRL/SAC/checkpoint/")
        if ckpt_path is None :
            self.ckpt_path = "/home/kuntima/Workspace/UniverseRL/SAC/checkpoint/sac_checkpoint_{}{}".format(env_name, suffix)
        
        print(f"Saving models to {self.ckpt_path}")
        torch.save({'policy_state_dict': self.policy_model.state_dict(),
                    'critic_online_a_state_dict': self.online_value_model_a.state_dict(),
                    'critic_online_b_state_dict': self.online_value_model_b.state_dict(),
                    'critic_target_a_state_dict': self.target_value_model_a.state_dict(),
                    'critic_target_b_state_dict': self.target_value_model_b.state_dict(),
                    'critic_optimizer_a_state_dict': self.value_optimizer_a.state_dict(),
                    'critic_optimizer_b_state_dict': self.value_optimizer_b.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                    "cumulative_expected_reward":reward}, self.ckpt_path)


    def load_checkpoint(self, ckpt_path, evaluate=True) :

        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
        print(f"Loading models from {ckpt_path}")
        if ckpt_path is not None :
            checkpoint = torch.load(ckpt_path, weights_only=False)
            
            
            """
            nS, nA = self.Env.observation_space.shape[0], self.Env.action_space.shape[0]
            action_bounds = (self.Env.action_space.low, self.Env.action_space.high)

            # actor 
            self.policy_model:SACActor = self.policy_model_fn(nS, action_bounds)
            self.policy_optimizer:Adam = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)
            
        
            #critic 1
            self.target_value_model_a:SACCritic = self.value_model_fn(nS, nA)
            self.online_value_model_a:SACCritic = self.value_model_fn(nS, nA)
            self.value_optimizer_a:Adam         = self.value_optimizer_fn(self.online_value_model_a, self.value_optimizer_lr)
            
            #critic 2
            self.target_value_model_b:SACCritic = self.value_model_fn(nS, nA)
            self.online_value_model_b:SACCritic = self.value_model_fn(nS, nA)
            self.value_optimizer_b:Adam         = self.value_optimizer_fn(self.online_value_model_b, self.value_optimizer_lr)
            """

            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
            self.online_value_model_a.load_state_dict(checkpoint['critic_online_a_state_dict'])
            self.online_value_model_b.load_state_dict(checkpoint['critic_online_b_state_dict'])
            self.target_value_model_a.load_state_dict(checkpoint['critic_target_a_state_dict'])
            self.target_value_model_b.load_state_dict(checkpoint['critic_target_b_state_dict'])
            self.value_optimizer_a.load_state_dict(checkpoint['critic_optimizer_a_state_dict'])
            self.value_optimizer_b.load_state_dict(checkpoint['critic_optimizer_b_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy_model.eval()
                self.online_value_model_a.eval()
                self.online_value_model_b.eval()
                self.target_value_model_a.eval()
                self.target_value_model_b.eval()
                
            else:
                self.policy_model.train()
                self.online_value_model_a.train()
                self.online_value_model_b.train()
                self.target_value_model_a.train()
                self.target_value_model_b.train()

        return checkpoint
    
    def basic_setup(self) :
        obs, _ = self.Env.reset()
        
        done  = False 

        while not done :
            
            action  =  self.Env.action_space.sample()
            print(f"action ={action}")
            obs, reward, terminated, truncated, info =  self.Env.step( action=action )
            print(f"observation ={obs}")
            episode_over = terminated or truncated