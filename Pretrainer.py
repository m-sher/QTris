import tensorflow as tf
from tensorflow import keras
import glob

class Pretrainer():
    def __init__(self, gamma, str_to_ind):

        self.gamma = gamma
        self.scc = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mse = keras.losses.MeanSquaredError()
        self.str_to_ind = str_to_ind
        self.move_dict = {
            '0': 'S',
            '1': 'l',
            '2': 'r',
            '3': 'L',
            '4': 'R',
            '5': 's',
            '6': 's',
            '7': 'a',
            '8': 'c',
            '9': 'H',
            '10': 'h',
            '11': '1',
        }

        def custom_reader_func(datasets):
            datasets = datasets.shuffle(10)
            return datasets.interleave(lambda x: x,
                                       deterministic=False,
                                       num_parallel_calls=tf.data.AUTOTUNE)
        
        try:
            self.gt_dset = tf.data.Dataset.load('saved_expert_dset', reader_func=custom_reader_func)
        except:
            self._load_dset()
            self.gt_dset = tf.data.Dataset.load('saved_expert_dset', reader_func=custom_reader_func)
            
        self.gt_dset = (self.gt_dset
                        .cache()
                        .shuffle(100000)
                        .batch(512,
                               deterministic=False,
                               drop_remainder=True,
                               num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    def _load_data(self):
        self.players_data = [[], []]
        for file in glob.glob('C:\\Users\\micha\\Downloads\\MisaMino-Tetrio-copy\\MisaMino-Tetrio-master\\tetris_ai\\logs\\game_att*.txt'):
            with open(file) as f:
                contents = f.readlines()
                for line in contents:
                    self.players_data[int(line[0])].append(line[2:])

    def _get_discounted_returns(self, rewards, gamma):
        returns = []
    
        rewards = rewards[::-1]
        discounted_sum = 0
        
        for reward in rewards:
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)
            
        returns = returns[::-1]

        return returns

    def _load_dset(self):
        self._load_data()
        
        self._dset_pieces = []
        self._dset_boards = []
        self._dset_actions = []
        self._dset_attacks = []
        
        for player_data in self.players_data:
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
                
                if len(action) == 0 or attack < 0:
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
                
                try:
                    action = self.str_to_ind[''.join(action)]
                except:
                    action = (-1, -1, -1)
                
                episode_pieces.append(piece_seq)
                episode_boards.append(board)
                episode_actions.append(action)
                episode_attacks.append(attack / 32)

        
                if i % 10000 == 0:
                    print(f'\r{(i+1)/len(player_data[:-1]):1.2f}', end='', flush=True)

        self.gt_dset = (tf.data.Dataset.from_generator(self._dset_generator,
                                                       output_signature=(tf.TensorSpec(shape=(28, 10), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(7,), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(3,), dtype=tf.int32),
                                                                         tf.TensorSpec(shape=(), dtype=tf.float32)))
                        .filter(self._filter_fn)
                        .enumerate())
        
        self.gt_dset.save('saved_expert_dset', shard_func=lambda idx, sample: idx % 10)

    def _dset_generator(self):
        for sample in zip(self._dset_boards, self._dset_pieces, self._dset_actions, self._dset_attacks):
            yield sample

    def _filter_fn(self, piece, board, action, att):
        return tf.reduce_all(action != -1)

    def cache_dset(self):
        for i, batch in enumerate(self.gt_dset):
            if i % 100 == 0:
                print(f'\rCaching {i}', end='', flush=True)

    @tf.function
    def _logit_loss(self, true, pred):
        loss = self.scc(true, pred)
        
        return loss

    @tf.function
    def _value_loss(self, true, pred):
        loss = self.mse(true, pred)
        
        return loss

    @tf.function
    def _logit_acc(self, true, pred):
        chosen = tf.argmax(pred, axis=-1, output_type=tf.int32)
        
        acc = chosen == true
        return acc

    @tf.function
    def _train_step(self, model, board, piece, action, target_value):
        
        with tf.GradientTape() as tape:
            _, all_log_probs, values = model(board, piece, gt_actions=action, training=True)
            actor_loss = tf.constant(0.0, tf.float32)
            for i, log_probs in enumerate(all_log_probs):
                actor_loss += self._logit_loss(action[:, i], log_probs)
            critic_loss = self._value_loss(target_value, values)
            total_loss = actor_loss + 0.5 * critic_loss            
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        accs = []
        for i, log_probs in enumerate(all_log_probs):
            accs.append(self._logit_acc(action[:, i], log_probs))
        acc = tf.reduce_mean(tf.cast(tf.reduce_all(accs, axis=0), tf.float32))
        
        return actor_loss, critic_loss, acc

    def train(self, model, epochs, ckpt_type='pretrained'):

        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)        

        if not ckpt_type:
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/pretrained', max_to_keep=5)
        else:
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'combined_checkpoints/{ckpt_type}', max_to_keep=5)
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(self.checkpoint_manager.latest_checkpoint)
        
        if ckpt_type == 'finetuned':
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/pretrained', max_to_keep=5)

        for epoch in range(epochs):
            for i, (_, (board, piece, action, target_value)) in enumerate(self.gt_dset):
                step_out = self._train_step(model, board, piece, action, target_value)
                if i % 100 == 0:
                    actor_loss, critic_loss, acc = step_out
                    print(f"\r{i}\t|\tActor Loss: {actor_loss:1.2f}\t|\t" + 
                          f"Critic Loss: {critic_loss:1.2f}\t|\t" +
                          f"Accuracy: {acc:1.2f}\t|\t", end='', flush=True)
        
            self.checkpoint_manager.save()

        return step_out