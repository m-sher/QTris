import tensorflow as tf
from tensorflow import keras
import time
import glob

class Pretrainer():
    def __init__(self, gamma):

        self.gamma = gamma
        self.scc = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.mse = keras.losses.MeanSquaredError(reduction='none')
        
        self.move_dict = {
            '0': 11,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 5,
            '7': 6,
            '8': 7,
            '9': 8,
            '10': 9,
            '11': 10,
        }

    def _load_data(self):
        players_data = [[], []]
        for file in glob.glob('C:\\Users\\micha\\Downloads\\MisaMino-Tetrio-copy\\MisaMino-Tetrio-master\\tetris_ai\\logs\\game_att*.txt'):
            with open(file) as f:
                contents = f.readlines()
                for line in contents:
                    players_data[int(line[0])].append(line[2:])
        return players_data

    def _get_discounted_returns(self, rewards, gamma):
        returns = []
    
        rewards = rewards[::-1]
        discounted_sum = 0
        
        for reward in rewards:
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)
            
        returns = returns[::-1]
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        return returns # .tolist()

    def _load_dset(self, players_data):
        self._dset_pieces = []
        self._dset_boards = []
        self._dset_actions = []
        self._dset_attacks = []
        self._max_len = 0
        
        for player_data in players_data:
            episode_pieces = []
            episode_boards = []
            episode_actions = []
            episode_attacks = []
            
            for i, line in enumerate(player_data[:-1]):
                current, hold, queue, raw_board, raw_action, total_att = line.strip().split('#')
                _, _, _, _, _, next_att = player_data[i+1].strip().split('#')
                
                piece_seq = [int(current)] + [int(hold)] + [int(piece) for piece in queue.split(',')[:5]]
                board = [[float(bit) for bit in '{:032b}'.format(int(row))[-10:][::-1]] for row in raw_board.split(',')[1:-4]]
                action = [self.move_dict[move_char] for move_char in raw_action.split(',')[:-1]]
                attack = int(next_att) - int(total_att)
                
                if len(action) > 0 and attack >= 0:
                    action = [11] * (action[0] != 11) + action
                    self._max_len = max(self._max_len, len(action))
                else:
                    episode_attacks = self._get_discounted_returns(episode_attacks, self.gamma)
            
                    self._dset_pieces += episode_pieces
                    self._dset_boards += episode_boards
                    self._dset_actions += episode_actions
                    self._dset_attacks += episode_attacks
                    
                    episode_pieces = []
                    episode_boards = []
                    episode_actions = []
                    episode_attacks = []
                    
                    continue
            
                episode_pieces.append(piece_seq)
                episode_boards.append(board)
                episode_actions.append(action)
                episode_attacks.append(attack / 4.0 + 0.01)
        
                if i % 10000 == 0:
                    print(f'\r{(i+1)/len(player_data[:-1]):1.2f}', end='', flush=True)

    def _dset_generator(self):
        for sample in zip(self._dset_pieces, self._dset_boards, self._dset_actions, self._dset_attacks):
            yield sample

    def _pad(self, item, length, pad_value=0):
        num_valid = tf.shape(item)[0]
        if num_valid > length:
            padded = item[:length]
        else:
            padded = tf.concat([item, tf.zeros((length - num_valid), dtype=item.dtype) + pad_value], axis=0)
        return padded

    def _pad_and_split(self, piece, board, action, att):
        padded_action = self._pad(action, self._max_len)
        inp = tf.ensure_shape(padded_action[:-1], (self._max_len-1,))
        tar = tf.ensure_shape(padded_action[1:], (self._max_len-1,))
        return (board, piece, inp), (tar, att)

    def _cache_dset(self):
        gt_dset = (tf.data.Dataset.from_generator(self._dset_generator,
                                                  output_signature=(tf.TensorSpec(shape=(7,), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(28, 10), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                                                    tf.TensorSpec(shape=(), dtype=tf.float32)))
                   .map(self._pad_and_split,
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False)
                   .cache()
                   .shuffle(100000)
                   .batch(128,
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False,
                          drop_remainder=True)
                   .prefetch(tf.data.AUTOTUNE))

        for i, ((board, piece, inp), (tar, att)) in enumerate(gt_dset):
            if i % 100 == 0:
                print(f'\r{i}', end='', flush=True)

        print(f'\rDone Caching')
        
        return gt_dset

    @tf.function
    def _masked_logit_loss(self, true, pred, mask):
        raw_loss = self.scc(true, pred)
        
        loss = tf.reduce_sum(raw_loss * mask) / tf.reduce_sum(mask)
        return loss

    @tf.function
    def _masked_value_loss(self, true, pred, mask):
        raw_loss = self.mse(true[..., None, None], pred)
        
        loss = tf.reduce_sum(raw_loss * mask) / tf.reduce_sum(mask)
        return loss

    @tf.function
    def _masked_logit_acc(self, true, pred, mask):
        preds = tf.argmax(pred, axis=-1, output_type=tf.int32)
        true = tf.cast(true, tf.int32)
        
        raw_acc = tf.cast(preds == true, tf.float32)
        
        acc = tf.reduce_sum(raw_acc * mask) / tf.reduce_sum(mask)
        return acc

    @tf.function
    def _train_step(self, actor, critic, board, piece, inp, tar, att):
        mask = tf.cast(tar != 0, tf.float32)
        
        with tf.GradientTape() as actor_tape:
            logits = actor((board, piece, inp), training=True)
            actor_loss = self._masked_logit_loss(tar, logits, mask)
        actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
        actor.optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        
        acc = self._masked_logit_acc(tar, logits, mask)

        with tf.GradientTape() as critic_tape:
            values = critic((board, piece, inp), training=True)
            critic_loss = self._masked_value_loss(att, values, mask)
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
        
        return actor_loss, critic_loss, acc

    def train(self, actor, critic, gt_dset, epochs):
        actor_losses, critic_losses, accs = [], [], []
        for epoch in range(epochs):
            print()
            last_time = time.time()
            for i, ((board, piece, inp), (tar, att)) in enumerate(gt_dset):
                actor_loss, critic_loss, acc = self._train_step(actor, critic, board, piece, inp, tar, att)
                if i % 10 == 0:
                    cur_time = time.time()
                    print(f'\r{i}\t|\tActor Loss: {actor_loss:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\tAccuracy: {acc:1.2f}\t|\tStep Time: {(cur_time - last_time) * 100:3.0f}ms\t|\t', end='', flush=True)
                    last_time = cur_time
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    accs.append(acc)
        return actor_losses, critic_losses, accs

