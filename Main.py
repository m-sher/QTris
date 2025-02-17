from TetrisModel import TetrisModel
from TrainerSeparateParallel import Trainer
from GenerateMoves import generate_key_dict
from Pretrainer import Pretrainer
import tensorflow as tf
from tensorflow import keras

piece_dim = 8
key_dim = 12
max_len = 7
depth = 32
num_heads = 4
num_layers = 4
dropout_rate = 0.1
trunk_dim = 64
out_dims = [2, 60, 8, 1]

entropy_coef = 0.0243
entropy_decay = 0.9
value_coef = 0.5
gamma = 0.95
lam = 0.95
temperature = 1.0
num_players = 32

ind_to_str, str_to_ind = generate_key_dict()

# pretrainer = Pretrainer(gamma=gamma, str_to_ind=str_to_ind)

model = TetrisModel(piece_dim=piece_dim,
                    key_dim=key_dim,
                    max_len=max_len,
                    depth=depth,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate,
                    trunk_dim=trunk_dim)

optimizer = keras.optimizers.Adam(3e-4, clipnorm=1)
model.compile(optimizer=optimizer)

all_actions, all_values, piece_scores = model((tf.random.uniform((num_players, 28, 10, 1)),
                                               tf.random.uniform((num_players, 7), minval=0, maxval=8, dtype=tf.int32),
                                               tf.random.uniform((num_players, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
model.summary()
print(tf.shape(all_actions))
print(tf.shape(all_values))
print(tf.shape(piece_scores))

if __name__ == '__main__':
    # pretrainer.train(model, 100, None)
    
    trainer = Trainer(model=model,
                      ind_to_str=ind_to_str,
                      entropy_coef=entropy_coef,
                      entropy_decay=entropy_decay,
                      value_coef=value_coef,
                      num_players=num_players,
                      players_to_render=1,
                      gamma=gamma,
                      lam=lam,
                      ckpt_type=None,
                      temperature=1,
                      max_holes=10,
                      max_height=15,
                      max_diff=10,
                      max_episode_steps=100)
    
    trainer.train(gens=10000, update_steps=4)
