
import numpy as np 

class ReplayBuffer :
    def __init__(self, max_size:int=10000,batch_size:int= 512) :
        #state, action, reward, next_state, terminal
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ns_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ts_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        

        self.max_size = max_size
        self.batch_size = batch_size
        self.idx = 0
        self.size = 0

    def sample(self, batch_size:int = None) -> np.ndarray:
        
        if batch_size ==  None :
             batch_size = self.batch_size
            
        _idx =  np.random.choice( self.size, size=self.batch_size, replace=True) 
        experiences = np.vstack(self.ss_mem[_idx]), \
                      np.vstack(self.as_mem[_idx]), \
                      np.vstack(self.rs_mem[_idx]), \
                      np.vstack(self.ns_mem[_idx]), \
                      np.vstack(self.ts_mem[_idx])
        
        return experiences
    

    def store(self, sample) -> None:
        state, action, reward, next_state, is_terminal = sample
        self.ss_mem[self.idx] = state
        self.as_mem[self.idx] = action
        self.rs_mem[self.idx] = reward
        self.ns_mem[self.idx] = next_state
        self.ts_mem[self.idx] = is_terminal

        self.idx += 1
        self.idx = self.idx % self.max_size
        self.size += 1
        self.size = min(self.size, self.max_size)

    def __len__(self) -> int:
        return self.size
