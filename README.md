| Demo | Description |
| ----------- | ----------- |
| <img src="https://github.com/m-sher/QTris/blob/main/Demo.gif" width="200"> | <p>The gif to the left is a 500 piece demo of the model playing. The set of pieces in the middle represent, in order, the active piece, the currently held piece, and the next five pieces. It is apparent that the model still requires substantial training to be competitive, but the keen eye will notice quite a few T-spins. </p><p>The version playing in the example is only able to "see" the board, the active piece, the hold piece, and the queue of the next 5 pieces. It had no way to consider Combo, Back-to-Back (B2B), or Garbage Queue, but a version with such capabilities is currently training.</p><p>The model processes the available information as follows:</p><ol><li>Convolution layers to make board patches</li><li>Embed each piece as a vector</li><li>Run decoder steps on board patches and piece vectors:</li><ol><li>Perform self-attention on the sequence of board patches</li><li>Perform cross-attention between piece vectors and board patches to update board patches</li><li>Perform self-attention on the sequence of piece vectors</li><li>Perform cross-attention between board patches and piece vectors to update piece vectors</li><li>Repeat decoder steps</li></ol><li>Autoregressively generate key presses:</li><ol><li>Embed each key press as a vector (just START key initially)</li><li>Run decoder steps on latent state representation (final piece vectors) and key vectors</li><ol><li>Perform self-attention of sequence of key vectors</li><li>Perform cross-attention between latent state representation and key vectors</li></ol><li>Pass final key vector sequence through densely connected layers to generate next-key distribution</li><li>Sample from next-key distribution and append result to key sequence</li></ol></ol>

# Todo Section: #

### Todo Items: ###

- Move Runner from environment repo to this one
- Remove or update the Oneshot model
   - Probably remove since Autoregressive can be sped up significantly for inference
- Give the model more information (garbage queue)
- Consider 1v1 environment
- Create demo/explanation videos w/ manim

### Completed Items: ###

- ~~Periodic garbage sent to queue~~
- ~~Give the model more information (b2b and combo)~~
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
