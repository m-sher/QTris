| Demo | Description |
| ----------- | ----------- |
| <img src="https://github.com/m-sher/QTris/blob/main/Demo.gif" width="150"> | <p>The gif to the left is a 1000 piece demo of the model playing. The set of pieces on the right represent, in order, the active piece, the currently held piece, and the next five pieces. It is apparent that the model still requires substantial training to be competitive, but the keen eye will notice a few T-spins and quads. It's also interesting to note that the model tends to save T pieces and I pieces for longer than it saves others.</p><p>The current version of the model is only able to "see" the board, the active piece, the hold piece, and the queue of the next 5 pieces. It has no way to consider Combo, Back-to-Back (B2B), or Garbage Queue (yet :D).</p><p>The model processes the available information as follows: </p><ol><li>Linear convolution to make board patches</li><li>Perform self-attention on patches</li><li>Encode each piece as a vector</li><li>Run decoder steps on board patches and piece vectors:</li><ol><li>Perform self-attention on the sequence of piece vectors</li><li>Perform cross-attention between the sequence of piece vectors and board patch vectors</li><li>Repeat decoder steps</li></ol><li>Flatten to form latent state vector</li><li>Generate hold action distribution based on latent state vector</li><li>Sample from distribution to get hold action</li><li>Embed hold action as vector and combine with latent state vector</li><li>Generate position action distribution based on new state vector</li><li>Sample from distribution to get position action</li><li>Embed position action as vector and combine with state vector</li><li>Generate spin action distribution based on new state vector</li><li>Sample from distribution to get spin action</li></ol> |

# Todo Section: #

### Priority Colors: ###

$$\color{red}\text{●}$$ High priority

$$\color{orange}\text{●}$$ Medium priority

$$\color{yellow}\text{●}$$ Low priority

$$\color{green}\text{●}$$ In progress

### Todo Items: ###

$$\color{green}\text{●}$$ Refer to this every day: JUST LET IT TRAIN SERIOUSLY IT'S WORKING, JUST LET IT LEARN

$$\color{green}\text{●}$$ Write this ReadMe

$$\color{green}\text{●}$$ Redo pretrainer for new model architecture 

$$\color{orange}\text{●}$$ Organize new working training script into separate files

$$\color{orange}\text{●}$$ Give the model more information (e.g. b2b, combo, garbage queue)

$$\color{orange}\text{●}$$ Instead of 1v1 environment, periodic garbage sent to queue

$$\color{yellow}\text{●}$$ Create demo/explanation videos w/ manim

### Completed Items: ###

$$\color{orange}\text{●}$$ ~~Check piece/death calculations for logging~~

$$\color{orange}\text{●}$$ ~~Check if tf.data.Dataset.save maintains pipeline operations when loading~~

$$\color{orange}\text{●}$$ ~~Add custom read function to shuffle shard reading order~~

$$\color{yellow}\text{●}$$ ~~Kill for too bumpy (maybe)~~

$$\color{yellow}\text{●}$$ ~~Fix method for saving demo gif in Trainer~~

$$\color{yellow}\text{●}$$ ~~Rename trainer attributes to follow private convention~~

$$\color{orange}\text{●}$$ ~~Split pretrainer dataset files into smaller chunks to improve loading time - tf.data.Dataset.save instead~~

$$\color{yellow}\text{●}$$ ~~Change expert dataset structure to match novice dataset~~

$$\color{yellow}\text{●}$$ ~~Fix pretrainer functions to make the class easier to use (why did I make them like that??)~~

$$\color{yellow}\text{●}$$ ~~Log more information to WandB - episode lengths, number of deaths, average action prob~~

$$\color{orange}\text{●}$$ ~~Remove reference model to speed up training, but log model output for per-step kl-div~~

$$\color{orange}\text{●}$$ ~~Implement parallel trajectory collection~~

$$\color{orange}\text{●}$$ ~~Revisit encouraging short actions - possibly by re-separating key sequences~~

$$\color{orange}\text{●}$$ ~~Move demo gif creation to its own file to not clutter the notebook~~

$$\color{yellow}\text{●}$$ ~~Consider reducing gamma - Failures later in episodes are affecting early placements too heavily~~

$$\color{yellow}\text{●}$$ ~~Adjust reward design to directly penalize holes~~

$$\color{orange}\text{●}$$ ~~Make the model larger~~

$$\color{orange}\text{●}$$ ~~Make feature extraction deeper~~

$$\color{orange}\text{●}$$ ~~Rescale input to [-1, 1]~~

$$\color{yellow}\text{●}$$ ~~Fix logged attention score differences (difference between unnormalized scores instead of normalized)~~

$$\color{yellow}\text{●}$$ ~~Log attention score differences~~

$$\color{yellow}\text{●}$$ ~~Consider using individual values (already calculated) when computing advantages~~

$$\color{yellow}\text{●}$$ ~~Log attention scores to WandB~~

$$\color{yellow}\text{●}$$ ~~Reevaluate use of reference model~~

$$\color{yellow}\text{●}$$ ~~Consider standardizing/scaling returns to ease value function learning (rejected)~~

$$\color{orange}\text{●}$$ ~~Save optimizer parameters~~

$$\color{red}\text{●}$$ ~~Add temperature in place of epsilon-greedy/stochastic-sampling~~

$$\color{red}\text{●}$$ ~~Treat key sequences as actions but maintain separate probabilities (re-separated)~~

$$\color{orange}\text{●}$$ ~~Fix value function learning~~

$$\color{orange}\text{●}$$ ~~Make entire model trainable with one optimizer (re-separated)~~

$$\color{orange}\text{●}$$ ~~Adjust reward scaling such that attacks are favored more heavily~~

$$\color{red}\text{●}$$ ~~Fix last value for GAE calculation~~