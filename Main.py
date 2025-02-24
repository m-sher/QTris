from TetrisModel import TetrisModel
from TrainerSeparateParallel import Trainer
from GenerateMoves import generate_all_finesse_moves_dict
# from Pretrainer import Pretrainer
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

piece_dim = 8
depth = 32
num_heads = 4
num_layers = 4
dropout_rate = 0.1
out_dims = [2, 37, 8, 1]

entropy_coef = 0.026181
entropy_decay = 0.9999
value_coef = 0.5
gamma = 0.95
lam = 0.95
temperature = 1.0
num_players = 32

str_to_ind, ind_to_str = generate_all_finesse_moves_dict()

# pretrainer = Pretrainer(gamma=gamma, str_to_ind=str_to_ind)

model = TetrisModel(piece_dim=piece_dim,
                    depth=depth,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate,
                    out_dims=out_dims)

optimizer = keras.optimizers.Adam(3e-4, clipnorm=1)
model.compile(optimizer=optimizer)

all_logits, all_values, piece_scores = model((tf.random.uniform((num_players, 28, 10, 1)),
                                              tf.random.uniform((num_players, 7), minval=0, maxval=8, dtype=tf.int32),
                                              tf.random.uniform((num_players, len(out_dims)-1), minval=0, maxval=2, dtype=tf.int32)), return_scores=True)
model.summary()
print(tf.shape(all_logits))
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
                      ckpt_type='finetuned',
                      temperature=1.0,
                      max_holes=5,
                      max_height=15,
                      max_diff=10,
                      max_episode_steps=500)
    
    trainer.train(gens=10000, update_steps=4)
