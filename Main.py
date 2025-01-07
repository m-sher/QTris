from TetrisModel import TetrisModel
from TrainerSeparateParallel import Trainer
import tensorflow as tf
from tensorflow import keras

piece_dim = 8
key_dim = 12
depth = 16
gamma = 0.99
lam = 0.95
temperature = 1.0
num_players = 8
display_rows = 4

max_len = 10



actor = TetrisModel(piece_dim=piece_dim,
                    key_dim=key_dim,
                    depth=depth,
                    num_heads=4,
                    num_layers=4,
                    max_length=max_len,
                    out_dim=key_dim)

actor_optimizer = keras.optimizers.Adam(3e-4, clipnorm=0.5)
actor.compile(optimizer=actor_optimizer)

actor_logits, piece_scores, key_scores = actor((tf.random.uniform((32, 28, 10, 1)),
                                                tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32),
                                                tf.random.uniform((32, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
actor.summary(), tf.shape(actor_logits), tf.shape(piece_scores), tf.shape(key_scores)



critic = TetrisModel(piece_dim=piece_dim,
                     key_dim=key_dim,
                     depth=depth,
                     num_heads=4,
                     num_layers=4,
                     max_length=max_len,
                     out_dim=1)

critic_optimizer = keras.optimizers.Adam(3e-4, clipnorm=0.5)
critic.compile(optimizer=critic_optimizer)

values, piece_scores, key_scores = critic((tf.random.uniform((32, 28, 10, 1)),
                                           tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32),
                                           tf.random.uniform((32, max_len), minval=0, maxval=key_dim, dtype=tf.int32)), return_scores=True)
critic.summary(), tf.shape(values), tf.shape(piece_scores), tf.shape(key_scores)



disc = TetrisModel(piece_dim=piece_dim,
                   key_dim=None,
                   depth=depth,
                   num_heads=4,
                   num_layers=4,
                   max_length=None,
                   out_dim=2)

disc_optimizer = keras.optimizers.Adam(3e-4, clipnorm=0.5)
disc.compile(optimizer=disc_optimizer)

disc_logits, piece_scores = disc((tf.random.uniform((32, 28, 10, 1)),
                                  tf.random.uniform((32, 7), minval=0, maxval=8, dtype=tf.int32)), return_scores=True)
disc.summary(), tf.shape(disc_logits), tf.shape(piece_scores)



actor_checkpoint = tf.train.Checkpoint(model=actor, optim=actor.optimizer)
actor_checkpoint.restore('actor_checkpoint/finetuned/small/ckpt-31')

critic_checkpoint = tf.train.Checkpoint(model=critic, optim=critic.optimizer)
critic_checkpoint.restore('critic_checkpoint/finetuned/small/ckpt-31')

actor_checkpoint = tf.train.Checkpoint(model=actor, optim=actor.optimizer)
actor_checkpoint_manager = tf.train.CheckpointManager(actor_checkpoint, 'actor_checkpoint/finetuned/small', max_to_keep=5)

critic_checkpoint = tf.train.Checkpoint(model=critic, optim=critic.optimizer)
critic_checkpoint_manager = tf.train.CheckpointManager(critic_checkpoint, 'critic_checkpoint/finetuned/small', max_to_keep=5)

disc_checkpoint = tf.train.Checkpoint(model=disc, optim=disc.optimizer)
disc_checkpoint_manager = tf.train.CheckpointManager(disc_checkpoint, 'disc_checkpoint/finetuned/small', max_to_keep=5)



trainer = Trainer(actor=actor,
                  critic=critic,
                  disc=disc,
                  max_len=max_len,
                  num_players=8,
                  gamma=gamma,
                  lam=lam,
                  temperature=1.0,
                  max_episode_steps=1000)

if __name__ == '__main__':
    while True:
        trainer.train(gens=100, update_steps=4)
        actor_checkpoint_manager.save()
        critic_checkpoint_manager.save()
