import random as rand
from tkinter.ttk import setup_master  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
import pandas as pd
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    def __init__(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_actions,  		  	   		  	  		  		  		    	 		 		   		 		  
        alpha,  		  	   		  	  		  		  		    	 		 		   		 		  
        gamma,  		  	   		  	  		  		  		    	 		 		   		 		  
        rar,  		  	   		  	  		  		  		    	 		 		   		 		  
        radr,  		  	   		  	  		  		  		    	 		 		   		 		  
        dyna,  		  	   		  	  		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):  				  	   		  	  		  		  		    	 		 		   		 		  
        """
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar =  rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.a = None
        self.s = None
        self.q = np.zeros((num_states,num_actions))
        if dyna>0:        
            self.T = np.full((num_states,num_actions,num_states), 0.00001)
            self.R = np.zeros((num_states,num_actions))
            self.visited = set()
        
    def querysetstate(self, s, learning=False):
        """ 		  	   		  	  		  		  		    	 		 		   		 		  
        Given this state save/update that.
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s: The new state
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        # """
        # if self.verbose:
        #     print('Entering query state with',s)		  	  		  		  		    	 		 		   		 		  
        arg_max_a = None
        if learning:
            if self.rar > np.random.uniform(0,1):
                arg_max_a = rand.randint(0, self.num_actions - 1)
            else:
                arg_max_a=np.argmax(self.q[s,:], axis=0)
        else:
            arg_max_a=np.argmax(self.q[s,:], axis=0)
        self.s = s
        self.a = arg_max_a
        return arg_max_a

  		  	   		  	  		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):
        """
        
        Return action to take    	 		 		   		 		  
        Update the Q table and return an action. If using dyna hallucinate as many iterations as required. 		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        #update T and visited for Dyna
        if self.dyna>0:  
            self.T[self.s,self.a,s_prime] += 1
            self.visited.add((self.s,self.a,s_prime))
        
        arg_max_a = None
        if self.rar > np.random.uniform(0,1):
            arg_max_a = rand.randint(0, self.num_actions - 1)
            self.rar = self.rar * self.radr
        else:
            arg_max_a = np.argmax(self.q[s_prime,:], axis=0)

        value = ((1-self.alpha) * self.q[self.s,self.a]) + self.alpha*(r + self.gamma*self.q[s_prime,arg_max_a])
        # if self.verbose:
        #     print('New value of q table for state:',s_prime,value)
        self.q[self.s,self.a] = value
        self.s = s_prime
        self.a = arg_max_a
        
        if self.dyna>0:
            #update rule for R
            # for _ in self.dyna:
            #     randomly select an experience tuple from visited
            #     infer s_prime for T pick max value s_prime
            #     infer r_1 = R[s,a]
            #     update Q with hallucinated experience tuple
            
            self.R[self.s,self.a]= (1-self.alpha)*self.R[self.s,self.a] + r*self.alpha
            for _ in range(self.dyna):
                [(s,a,_)] = rand.sample(self.visited,1)
                s_prime = np.argmax(self.T[s,a,:],axis=0) 
                r = self.R[s,a]
                self.q[s,a] = (1-self.alpha) * self.q[s,a] + self.alpha*(r + self.gamma*self.q[s_prime,a])
            
        return arg_max_a
