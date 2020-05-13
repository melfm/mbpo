import argparse
import gym
import numpy as np
import os

import utils
from shaping import NFShaping
from normalizer import Normalizer
import tensorflow as tf

import pdb


def train_shaping(state_dim, action_dim, max_action, args):

	# Load buffer
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.load(f"./buffers/{buffer_name}")

    # Get a tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    demo_shapes = {}
    demo_shapes["obs1"] = state_dim
    demo_shapes["acts"] = action_dim

    gamma = 0.99
    norm_eps= 0.01
    norm_clip = 5

    num_demo = 1000
    eps_length = 1000 
    max_u = max_action
    max_num_transitions = num_demo * eps_length
    batch_size = 256
    num_bijectors= 4
    layer_sizes  =[256, 256]
    prm_loss_weight= 1.0
    reg_loss_weight= 200.0
    potential_weight= 3.0
    lr= 5e-4
    num_masked=2

    def generate_demo_data():
        demo_data = replay_buffer.sample()
        
        num_transitions = demo_data["acts"].shape[0]
        print('Num transitions ', num_transitions)
        assert all([demo_data[k].shape[0] == num_transitions for k in demo_data.keys()])
        for i in range(num_transitions):
            yield {k: demo_data[k][i] for k in demo_shapes.keys()}

    demo_dataset = (
        tf.data.Dataset.from_generator(
            generate_demo_data, output_types={k: tf.float32 for k in demo_shapes.keys()}, output_shapes=demo_shapes
        )
        .take(max_num_transitions)
        .shuffle(max_num_transitions)
        .repeat(1)
    )

    state_dim_tuple = (state_dim,)
    o_stats = Normalizer(state_dim_tuple, norm_eps, norm_clip, sess=sess)
    g_stats = None

    demo_shaping = NFShaping(sess,
                        gamma,
                        max_u,
                        max_num_transitions,
                        batch_size,
                        demo_dataset,
                        o_stats,
                        g_stats,
                        num_bijectors,
                        layer_sizes,
                        num_masked,
                        potential_weight,
                        prm_loss_weight,
                        reg_loss_weight)
    init=tf.global_variables_initializer()
    sess.run(init)
    #
    # train normalizing flow or gan for 1 epoch
    loss = 0
    demo_shaping.initialize_dataset()
    losses = np.empty(0)
    while True:
        #try:
        #pdb.set_trace() 
        loss = demo_shaping.train()
        print('Loss ', loss)
        losses = np.append(losses, loss)
        # except tf.errors.OutOfRangeError:
        #     pdb.set_trace() 
        #     loss = np.mean(losses)
        #     break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
    args = parser.parse_args()

    env = gym.make(args.env)

    env.seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    train_shaping(state_dim, action_dim, max_action, args)