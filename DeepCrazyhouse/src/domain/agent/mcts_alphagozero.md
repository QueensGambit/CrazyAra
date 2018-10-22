
Monte Carlo Tree Search (Alpha Go Zero / Alpha Zero):
=====================================================

_"AlphaZero uses a generalpurpose
Monte-Carlo tree search (MCTS) algorithm. Each search consists of a series of simulated
games of self-play that traverse a tree from root sroot to leaf. Each simulation proceeds by
selecting in each state s a move a with low visit count, high move probability and high value
(averaged over the leaf states of simulations that selected a from s) according to the current
neural network f. The search returns a vector pi representing a probability distribution over
moves, either proportionally or greedily with respect to the visit counts at the root state."_ (p. 3, Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm)

Each Node s in the tree contains edges (s, a) for all legal actions.

Each Node stores the following stats:

N(s, a) - Visit count

W(s, a) - Total action value / Monte Carlo estimates of the total action value

Q(s, a) - (Combined) mean action value

P(s, a) - Prior probabiblity selecting that edge / move


* Multiple simulation are executed in parallel.

* Algortihm iterates over three phases:


Step I: Select()
----------------

* Starts at the root node s_0 and finishes at leaf node s_l at time step L.
* At each of the time-steps, t<T an action is selected according to the statistics
  in the search tree:

  a_t = argmax(Q(s_t, a) + U(s_t, a))

  u(s, a) = c_puct * P(s, a) * (sqrt(sum(N(s, b))) / (1 + N(s, a))

This search control strategy prefers actions with high prior probability low visit count,
but asymptotially prefers actions with high action value.


Step II: Expand_and_evaluate()
------------------------------

The lef node s_l is added to a queue for neural network evaluation using a mini-batch size of 8.
The search thread is locked until evaluation completes.
The leaf node is expanded and each edge (s_l, a) is initialized to:

* N(s_l, a) = 0
* W(s_l, a) = 0
* Q(s_l, a) = 0
* P(s_l, a) = p_a

The value v is then backed up.


Step III: Backup()
------------------

The edge statistics are updated in a backward pass through each step t <= L.
The visit counts are incremented:

* N(s_t, a_t) += 1

and the action value is updated to the mean value:

* W(s_t, a_t) += v
* Q(s_t, a_t) = W(s_t, a_t) / N(s_t, a_t)

Virtual loss is used to ensure each thread evaluates different nodes.


Step IV: Play()
---------------

At the end of the search AlphaGoZero selects a move top lay in the root position s_0,
proportional to its exponentatial visit count:

* pi(a | s_0) = N(s_0, a)^(1/T) / sum( N(s_0, b)^(1/T) )

Where T is a temperature parameters that controls the level of exploration.
The search tree is reused at subsequent time-steps:
The child node corresponding to the played action becomes the new root node;
the subtree below this child is retained along with all its statistics, while the remainder
of the tree is discard. AlphaGo Zero resigns if its root value and best child value are lower
than a threshold value v_resign.


Differences to older versions:
------------------------------

* Compared to older MCTS versions here no rolloouts are used.
* It uses a single net for both policy and value predicitons.
* Leaf nodes are always expanded
* Each search thread waits for the neural net evaluation
* A transposition table was used fo rthe large (40 blocks) instance


Additional References:
----------------------

* https://medium.com/@jonathan_hui/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a







