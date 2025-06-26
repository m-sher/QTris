import tensorflow as tf
from TetrisModel import PolicyModel
from TetrisEnvs.PyTetrisEnv.PyTetrisEnv import PyTetrisEnv
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
import imageio
import numpy as np
import time

# Model params
num_envs = 1
piece_dim = 8
key_dim = 12
depth = 32
num_heads = 4
num_layers = 4
dropout_rate = 0.1
max_len = 9

num_steps = 100
queue_size = 5
max_holes = 10
max_height = 20

p_model = PolicyModel(batch_size=num_envs,
                      piece_dim=piece_dim,
                      key_dim=key_dim,
                      depth=depth,
                      max_len=max_len,
                      num_heads=num_heads,
                      num_layers=num_layers,
                      dropout_rate=dropout_rate,
                      output_dim=key_dim)

p_checkpoint = tf.train.Checkpoint(model=p_model)
p_checkpoint_manager = tf.train.CheckpointManager(p_checkpoint, './policy_checkpoints_6', max_to_keep=3)
p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()

p_model.build(input_shape=[(None, 24, 10, 1),
                           (None, queue_size + 2),
                           (None, max_len)])

p_model.summary()

py_env = PyTetrisEnv(queue_size=queue_size,
                     max_holes=max_holes,
                     max_height=max_height,
                     seed=123,
                     idx=0)
env = TFPyEnvironment(py_env)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((625, 600))
pygame.display.set_caption("Tetris")
font = pygame.font.Font(None, 30)

time_step = env.reset()

frames = []
attacks = []
clears = []

piece_display = np.load('PieceDisplay.npy')

start = time.time()
for t in range(num_steps):
    board = time_step.observation['board']
    pieces = time_step.observation['pieces']
    attack = time_step.reward['attack']
    clear = time_step.reward['clear']

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    key_sequence, log_probs, masks, scores = p_model.predict((board, pieces), greedy=True)

    board_scores = tf.reshape(tf.reduce_sum(scores, axis=[0, 2, 3])[0], (12, 5))
    board_scores = (board_scores - tf.reduce_min(board_scores)) / (tf.reduce_max(board_scores) - tf.reduce_min(board_scores))

    piece_sidebar = piece_display[pieces[0].numpy()].reshape((28, 5))

    screen.fill((255, 255, 255))

    board_surf = pygame.Surface((10, 24))
    piece_surf = pygame.Surface((5, 28))
    scores_surf = pygame.Surface((5, 12))

    pygame.surfarray.blit_array(board_surf, board[0, ..., 0].numpy().T * 255)
    pygame.surfarray.blit_array(piece_surf, piece_sidebar.T * 255)
    pygame.surfarray.blit_array(scores_surf, board_scores.numpy().T * 255)

    board_surf = pygame.transform.scale(board_surf, (250, 600))
    piece_surf = pygame.transform.scale(piece_surf, (125, 600))
    scores_surf = pygame.transform.scale(scores_surf, (250, 600))

    screen.blit(board_surf, (0, 0))
    screen.blit(piece_surf, (250, 0))
    screen.blit(scores_surf, (375, 0))
    pygame.display.update()

    frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))
    attacks.append(attack.numpy()[0])
    clears.append(clear.numpy()[0])

    time_step = env.step(key_sequence)

print(f"Time taken: {time.time() - start:.2f} seconds")

# imageio.mimsave('demo.gif', frames, fps=3)

slider = Slider(screen, x=10, y=5, width=600, height=10, min=0, max=num_steps - 1, step=1, colour=(125, 125, 125), handleColour=(50, 50, 50))

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill((255, 255, 255))

    ind = slider.getValue()
    frame = frames[ind]

    pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
    pygame_widgets.update(events)
    
    attack_text = font.render(f"Attack: {attacks[ind]}", True, (255, 255, 255))
    clear_text = font.render(f"Clear: {int(clears[ind] * 5)}", True, (255, 255, 255))
    
    # Position text below the slider
    screen.blit(attack_text, (10, 35))
    screen.blit(clear_text, (10, 70))

    pygame.display.update()