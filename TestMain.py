from TetrisModel import TetrisModel
from TrainerSeparateParallel import Trainer
import tensorflow as tf
from tensorflow import keras

piece_dim = 8
key_dim = 12
depth = 32
gamma = 0.99
lam = 0.95
temperature = 1.0
num_players = 16
display_rows = 4

max_len = 10

actor = TetrisModel(piece_dim=piece_dim,
                    key_dim=key_dim,
                    depth=depth,
                    num_heads=4,
                    num_layers=4,
                    max_length=max_len,
                    out_dim=key_dim)

actor_optimizer = keras.optimizers.Adam(3e-4, clipnorm=1.5)
actor.compile(optimizer=actor_optimizer)

logits, piece_scores, key_scores = actor((tf.random.uniform((32, 28, 10, 1)),
                                          tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32),
                                          tf.random.uniform((32, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
actor.summary(), tf.shape(logits), tf.shape(piece_scores), tf.shape(key_scores)



critic = TetrisModel(piece_dim=piece_dim,
                     key_dim=key_dim,
                     depth=depth,
                     num_heads=4,
                     num_layers=4,
                     max_length=max_len,
                     out_dim=1)

critic_optimizer = keras.optimizers.Adam(3e-4, clipnorm=1.5)
critic.compile(optimizer=critic_optimizer)

values, piece_scores, key_scores = critic((tf.random.uniform((32, 28, 10, 1)),
                                           tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32),
                                           tf.random.uniform((32, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
critic.summary(), tf.shape(values), tf.shape(piece_scores), tf.shape(key_scores)


ref_model = TetrisModel(piece_dim=piece_dim,
                        key_dim=key_dim,
                        depth=depth,
                        num_heads=4,
                        num_layers=4,
                        max_length=max_len,
                        out_dim=key_dim)

logits, piece_scores, key_scores = ref_model((tf.random.uniform((32, 28, 10, 1)),
                                              tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32),
                                              tf.random.uniform((32, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
ref_model.summary(), tf.shape(logits), tf.shape(piece_scores), tf.shape(key_scores)


actor_checkpoint = tf.train.Checkpoint(model=actor, optim=actor.optimizer)
actor_checkpoint.restore('good_checkpoints/agent_checkpoint/ckpt-9')

critic_checkpoint = tf.train.Checkpoint(model=critic, optim=critic.optimizer)
critic_checkpoint.restore('good_checkpoints/critic_checkpoint/ckpt-9')

ref_model.set_weights(actor.get_weights())

actor_checkpoint = tf.train.Checkpoint(model=actor, optim=actor.optimizer)
actor_checkpoint_manager = tf.train.CheckpointManager(actor_checkpoint, 'actor_checkpoint/finetuned', max_to_keep=5)

critic_checkpoint = tf.train.Checkpoint(model=critic, optim=critic.optimizer)
critic_checkpoint_manager = tf.train.CheckpointManager(critic_checkpoint, 'critic_checkpoint/finetuned', max_to_keep=5)

trainer = Trainer(actor=actor,
                  critic=critic,
                  ref_model=ref_model,
                  max_len=max_len,
                  num_players=8,
                  gamma=gamma,
                  lam=lam,
                  temperature=1.0,
                  max_episode_steps=1000)

if __name__ == '__main__':
    trainer.train(gens=1000)