import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from torch.distributions import Normal
from torch.optim import Adam 


class SACActor(nn.Module) :

    def __init__(self, input_dim:int, 
                 action_bounds:tuple, 
                 hidden_dims:tuple=(32,32), 
                 activation_fc=F.relu,
                 entropy_lr:float=0.000003,
                 automatic_entropy_tuning:bool=True,
                 log_std:tuple=(-20,2)) :
        super(SACActor, self).__init__()

        self.log_std_min, self.log_std_max = log_std # from original code SAC (-20,2)
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds
       
        self.entropy_lr = entropy_lr
        
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.input_layer = nn.Linear(input_dim, 
                                     hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1) :
            hidden_layer = nn.Linear( hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.nn_min = F.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = F.tanh(torch.Tensor([float('inf')])).to(self.device)

        
        self.target_entropy = -torch.prod(torch.tensor(self.env_max.shape)).to(self.device)
        print(f"Target Entropy : {self.target_entropy}")
        #self.log_alpha = torch.zeros(1, requires_grad=True,device=self.device)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = Adam([self.log_alpha],lr=self.entropy_lr)
        self.alpha = self.log_alpha.exp().to(self.device)

        self.action_scale = 0.5 * (self.env_max - self.env_min)
        self.action_bias = 0.5 * (self.env_max + self.env_min)

        self.rescale = lambda x: x*self.action_scale + self.action_bias
        

    def rescale_(self, x:torch.tensor) -> torch.tensor: 
        # Assumes nn_min = -1 and nn_max = 1
        #Normalize 
        x_normalized = (  x - self.nn_min )/( self.nn_max - self.nn_min)
        #Scale back to env values
        x_normalized_scaled = x_normalized *(self.env_max -  self.env_min)
        # Shift to the origin 
        x_normalized_scaled += self.env_min

        return x_normalized_scaled
           
    def _format(self, x:np.ndarray) -> torch.Tensor: 

        if not isinstance(x, torch.Tensor) :
            x = torch.tensor(data=x, dtype=torch.float32, device=self.device )
            x = x.unsqueeze(0)
        return x
    
    def forward(self, state:np.ndarray) -> tuple:
        
        state = self._format( state )
        x = self.activation_fc(self.input_layer( state )) 

        for hidden_layer in self.hidden_layers :
            x = self.activation_fc(hidden_layer(x))

        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, self.log_std_min, self.log_std_max) # from original code SAC (-20,2)
        return x_mean, x_log_std
    
    

    def full_pass(self, state:np.ndarray, epsilon:float=1e-6) -> tuple:

        
        mean, log_std = self.forward(state=state)
        std = log_std.exp()
        pi = Normal(mean, std )
        pre_tanh_action = pi.rsample() # reparametrization trick  mean + variance * epsilon, where epsilon 1 (?)
        tanh_action =  torch.tanh( pre_tanh_action )
        action = self.rescale(tanh_action)
        
        # Change of Variables
        # x = action
        # y =  tanh(x)
        # dy/dx =  1 - tanh²(x)
        # p(y) = p(x)dx/dy
        # log{p(y)} = log{p(x)} + log{dx/dy} =  log{p(x)} - log{dy/dx} = log{p(x)} + log{1 - tanh²(x)}
        # log{p(y)} = log{p(x)} + clmap(log{1 - tanh²(x) + epsilon}, 0,1) 
        #log_prob = pi.log_prob(pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0,1)   + epsilon) 
        correction = torch.log(self.action_scale * (1 - tanh_action.pow(2)) + epsilon).clamp(min=1e-6)
        l_prob = pi.log_prob(pre_tanh_action) 
        log_prob = l_prob - correction
        
        
        # a = [ a1, a2, a3,......, an] 
        # p(a) = p(a1)*p(a2)*p(a3)*.......*p(an) because the action are consider independent 
        # log{p(a)} = log{p(a1)} + log{p(a2)} + log{p(a3)} + .......+ log{p(an)}

        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    

    def get_actions(self, x:np.ndarray) -> tuple:
        
        action_shape = self.env_max.cpu().numpy().shape
        mean, std = self.forward(state=x) 

        action = self.rescale(torch.tanh( Normal(mean, std.exp()).sample() )) # No reparametridzation trick : mean + variance * epsilon, where epsilon follows a Normal distribution with zero mean and 1 variance
        action = action.detach().cpu().numpy().reshape(action_shape)
        
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(), high=self.env_max.cpu().numpy()).reshape(action_shape)

        greedy_action = self.rescale(torch.tanh(mean)).detach().cpu().numpy().reshape(action_shape)
        
        return action, random_action, greedy_action
    
    def select_random_action(self, state:np.ndarray) -> np.ndarray:
        _, random_action, _ = self.get_actions(x=state)
        return random_action
    
    def select_greedy_action(self, state:np.ndarray) -> np.ndarray:
        _, _, greedy_action = self.get_actions(x=state)
        return greedy_action
    
    def select_action(self, state:np.ndarray) -> np.ndarray:
        action, _, _ = self.get_actions(x=state)
        return action

    def load(self, experiences):

        states, actions, rewards, next_states, is_terminals = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)

        return states, actions, rewards, next_states, is_terminals
    
class SACCritic(nn.Module) :

    def __init__(self, input_dim:int, out_dim:int, hidden_dims:tuple=(32,32), activation_fc:object=F.relu ) :
        super(SACCritic, self).__init__()

        self.input_layer = nn.Linear( input_dim + out_dim,  hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) -1 ) :
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1] )
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear( hidden_dims[-1], 1)
        self.activation_fc = activation_fc 

        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        self.to(self.device)

    def _format(self, x:np.ndarray, u:np.ndarray) -> torch.Tensor :

        if not isinstance(x, torch.Tensor) :
            x = torch.tensor(data=x,dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor) :
            u = torch.tensor(data=u,dtype=torch.float32, device=self.device)
            u = u.unsqueeze(0)
        return x, u

    def forward( self,state:np.ndarray, action:np.ndarray) -> torch.Tensor :

        state, action = self._format(x=state,u=action)

        x = torch.cat((state, action), dim=1)
        x = self.activation_fc(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers :
            x =  self.activation_fc(hidden_layer(x))
        
        x = self.output_layer(x)

        return x


   
