
# coding: utf-8

# ### Demonstration of an evaluation based on the MCTSAgent

# In[1]:


#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')


# In[2]:


#get_ipython().magic('reload_ext autoreload')


# In[3]:


import chess
import chess.variant
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'../../../')
import DeepCrazyhouse.src.runtime.Colorer
from DeepCrazyhouse.src.domain.agent.NeuralNetAPI import NeuralNetAPI
from DeepCrazyhouse.src.domain.agent.player.MCTSAgent import MCTSAgent
from DeepCrazyhouse.src.domain.agent.player.RawNetAgent import RawNetAgent
from DeepCrazyhouse.src.domain.crazyhouse.GameState import GameState
from time import time
#get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')


# In[4]:


net = NeuralNetAPI(ctx='gpu')


# In[5]:


raw_agent = RawNetAgent(net)


# In[6]:


mcts_agent = MCTSAgent(net, threads=8, playouts_empty_pockets=128, playouts_filled_pockets=256,
                 playouts_update=512, cpuct=1, dirichlet_epsilon=.1, dirichlet_alpha=0.2, max_search_time_s=300,
                 max_search_depth=15, temperature=0., clip_quantil=0., virtual_loss=3, verbose=True)


# In[7]:


board = chess.variant.CrazyhouseBoard()

board.push_uci('e2e4')
#board.push_uci('e7e6')

#fen = 'rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR/P w KQkq - 1 3'
#fen = 'r4rk1/ppp2pp1/3p1q1p/n1bPp3/2B1B1b1/3P1N2/PPP2PPP/R2Q1RK1[Nn] w - - 2 13'
#fen = 'rnb2rk1/p3bppp/2p5/3p2P1/4n3/8/PPPPBPPP/RNB1K1NR/QPPq w KQ - 0 11'
#fen = 'r1b1kbnr/ppp1pppp/2n5/3q4/3P4/8/PPP1NPPP/RNBQKB1R/Pp b KQkq - 1 4'
#fen = 'r1b1k2r/ppp2ppp/2n5/3np3/3P4/2PBP3/PpPB1PPP/1Q2K1NR/QNrb b Kkq - 27 14'
#board.set_fen(fen)

state = GameState(board)
board


# In[8]:


len(list(state.get_legal_moves()))


# In[9]:


def plot_moves_with_prob(moves, probs, only_top_x=None):
    
    # revert the ordering afterwards
    idx_order = np.argsort(probs)[::-1]
    
    if only_top_x is not None and only_top_x < len(idx_order):
        idx_order = idx_order[:only_top_x]
    
    #moves_ordered = moves[range(len(moves))] #idx_order[::-1]]
    probs_ordered = [] #probs[idx_order]
    
    moves_ordered = []
    for idx in idx_order:
        probs_ordered.append(probs[idx])
        moves_ordered.append(moves[idx])
        
    plt.barh(range(len(probs_ordered)), probs_ordered)
    plt.yticks(range(len(moves_ordered)), moves_ordered)


# ### Evalution using the raw network

# In[10]:


t_s = time()
value, legal_moves, p_vec_small = raw_agent.evaluate_board_state(state)
print('Elapsed time: %.4fs' % (time()-t_s))


# In[11]:


plot_moves_with_prob(legal_moves, p_vec_small, only_top_x=10)


# ### Evalution using the MCTS-Agent

# In[12]:


t_s = time()
value, legal_moves, p_vec_small = mcts_agent.evaluate_board_state(state)
print('Elapsed time: %.4fs' % (time()-t_s))


# In[13]:


mcts_agent.get_calclated_line()


# In[14]:


plot_moves_with_prob(legal_moves, p_vec_small, only_top_x=10)

