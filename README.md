| Demo | Description |
| ----------- | ----------- |
| <img src="https://github.com/m-sher/QTris/blob/main/Demo.gif" width="150"> | <p>The gif to the left is a 1000 piece demo of the model playing. The set of pieces on the right represent, in order, the active piece, the currently held piece, and the next five pieces. It is apparent that the model still requires substantial training to be competitive, but the keen eye will notice a few T-spins and quads. It's also interesting to note that the model tends to save T pieces and I pieces for longer than it saves others.</p><p>The current version of the model is only able to "see" the board, the active piece, the hold piece, and the queue of the next 5 pieces. It has no way to consider Combo, Back-to-Back (B2B), or Garbage Queue (yet :D).</p><p>The model processes the available information as follows: </p><ol><li>Linear convolution to make board patches</li><li>Perform self-attention on patches</li><li>Encode each piece as a vector</li><li>Run decoder steps on board patches and piece vectors:</li><ol><li>Perform self-attention on the sequence of piece vectors</li><li>Perform cross-attention between the sequence of piece vectors and board patch vectors</li><li>Repeat decoder steps</li></ol><li>Flatten to form latent state vector</li><li>Generate hold action distribution based on latent state vector</li><li>Sample from distribution to get hold action</li><li>Embed hold action as vector and combine with latent state vector</li><li>Generate position action distribution based on new state vector</li><li>Sample from distribution to get position action</li><li>Embed position action as vector and combine with state vector</li><li>Generate spin action distribution based on new state vector</li><li>Sample from distribution to get spin action</li></ol> |

# Todo Section: #

### Todo Items: ###

- Redo priority colors in ReadMe
- Give the model more information (e.g. b2b, combo, garbage queue)
- 1v1 environment or periodic garbage sent to queue
- Create demo/explanation videos w/ manim

### Completed Items: ###

- ~~Check piece/death calculations for logging~~
- ~~Check if tf.data.Dataset.save maintains pipeline operations when loading~~
- ~~Add custom read function to shuffle shard reading order~~
- ~~Kill for too bumpy (maybe)~~
- ~~Fix method for saving demo gif in Trainer~~
- ~~Rename trainer attributes to follow private convention~~
- ~~Split pretrainer dataset files into smaller chunks to improve loading time - tf.data.Dataset.save instead~~
- ~~Change expert dataset structure to match novice dataset~~
- ~~Fix pretrainer functions to make the class easier to use (why did I make them like that??)~~
- ~~Log more information to WandB - episode lengths, number of deaths, average action prob~~
- ~~Remove reference model to speed up training, but log model output for per-step kl-div~~
- ~~Implement parallel trajectory collection~~
- ~~Revisit encouraging short actions - possibly by re-separating key sequences~~
- ~~Move demo gif creation to its own file to not clutter the notebook~~
- ~~Consider reducing gamma - Failures later in episodes are affecting early placements too heavily~~
- ~~Adjust reward design to directly penalize holes~~
- ~~Make the model larger~~
- ~~Make feature extraction deeper~~
- ~~Rescale input to [-1, 1]~~
- ~~Fix logged attention score differences (difference between unnormalized scores instead of normalized)~~
- ~~Log attention score differences~~
- ~~Consider using individual values (already calculated) when computing advantages~~
- ~~Log attention scores to WandB~~
- ~~Reevaluate use of reference model~~
- ~~Consider standardizing/scaling returns to ease value function learning (rejected)~~
- ~~Save optimizer parameters~~
- ~~Add temperature in place of epsilon-greedy/stochastic-sampling~~
- ~~Treat key sequences as actions but maintain separate probabilities (re-separated)~~
- ~~Fix value function learning~~
- ~~Make entire model trainable with one optimizer (re-separated)~~
- ~~Adjust reward scaling such that attacks are favored more heavily~~
- ~~Fix last value for GAE calculation~~