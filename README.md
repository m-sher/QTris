# Description #

The gif below is a 1000 piece demo of the model playing. The set of pieces on the right represent, in order, the active piece, the currently held piece, and the next five pieces. It is apparent that the model still requires substantial training to be competitive, but the keen eye will notice a few T-spins and quads. It's also interesting to note that the model tends to save T pieces and I pieces for longer than it saves others.

## Demo ##

![](https://github.com/m-sher/QTris/blob/main/Demo.gif)

# Todo Section: #

### Priority Colors: ###

$$\color{red}\text{●}$$ High priority

$$\color{orange}\text{●}$$ Medium priority

$$\color{yellow}\text{●}$$ Low priority

$$\color{green}\text{●}$$ In progress

### Todo Items: ###

$$\color{green}\text{●}$$ Refer to this every day: JUST LET IT TRAIN SERIOUSLY IT'S WORKING, JUST LET IT LEARN

$$\color{green}\text{●}$$ Create demo/explanation videos w/ manim

$$\color{orange}\text{●}$$ Move demo gif creation to its own file to not clutter the notebook

$$\color{yellow}\text{●}$$ Implement parallel trajectory collection

$$\color{yellow}\text{●}$$ Give the model more information (e.g. b2b, combo, garbage queue)

$$\color{yellow}\text{●}$$ Setup 1v1 environment w/ garbage queue 

$$\color{green}\text{●}$$ Write this ReadMe

$$\color{yellow}\text{●}$$ Revisit encouraging short actions

### Completed Items: ###

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