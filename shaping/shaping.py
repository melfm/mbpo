import itertools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from normalizing_flows import MAF, RealNVP

tfd = tfp.distributions


class Shaping:
    def __init__(self, gamma):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g
            u_2 - output from the actor of the main network
        """
        self.gamma = gamma

    def potential(self, o, g, u):
        raise NotImplementedError

    def reward(self, o, g, u, o_2, g_2, u_2):
        potential = self.potential(o, g, u)
        next_potential = self.potential(o_2, g_2, u_2)
        assert potential.shape[1] == next_potential.shape[1] == 1
        return self.gamma * next_potential - potential

    def train(self, batch):
        pass

    def evaluate(self, batch):
        pass

    def post_training_update(self, batch):
        """
        """
        pass

    def _concat_normalize_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = self.o_stats.normalize(o)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, self.g_stats.normalize(g)])
        state_tf = tf.concat(axis=1, values=[state_tf, u / self.max_u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _concat_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = o
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, g])
        state_tf = tf.concat(axis=1, values=[state_tf, u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _cast_concat_normalize_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = tf.cast(self.o_stats.normalize(o), tf.float64)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(self.g_stats.normalize(g), tf.float64)])
        state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(u / self.max_u, tf.float64)])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _cast_concat_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = tf.cast(o, tf.float64)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(g, tf.float64)])
        state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(u, tf.float64)])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf


class NFShaping(Shaping):
    def __init__(
        self,
        sess,
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
        reg_loss_weight,
    ):
        """
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            prm_loss_weight  (float)
            reg_loss_weight  (float)
            potential_weight (float)
        """
        super().__init__(gamma)

        # Prepare parameters
        self.sess = sess
        self.max_u = max_u
        self.num_bijectors = num_bijectors
        self.layer_sizes = layer_sizes
        self.prm_loss_weight = prm_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.potential_weight = tf.constant(potential_weight, dtype=tf.float64)

        #
        self.learning_rate = 2e-4
        self.scale = tf.constant(5, dtype=tf.float64)

        import pdb;pdb.set_trace()
        demo_dataset = demo_dataset.shuffle(max_num_transitions).batch(batch_size)
        demo_iter_tf = demo_dataset.make_initializable_iterator()
        self.demo_iter_init_tf = demo_iter_tf.initializer
        self.demo_inputs_tf = demo_iter_tf.get_next()

        # normalizer for goal and observation.
        self.o_stats = o_stats
        self.g_stats = g_stats

        # training
        demo_state_tf = self._cast_concat_normalize_inputs(
            self.demo_inputs_tf["obs1"],
            None,
            self.demo_inputs_tf["acts"],
        )

        # normalizing flow
        demo_state_dim = int(demo_state_tf.shape[1])
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([demo_state_dim], tf.float64))
        self.nf = MAF(
            base_dist=self.base_dist, dim=demo_state_dim, num_bijectors=num_bijectors, layer_sizes=layer_sizes
        )
        # loss function that tries to maximize log prob
        # log probability
        neg_log_prob = tf.clip_by_value(-self.nf.log_prob(demo_state_tf), -1e5, 1e5)
        neg_log_prob = tf.reduce_mean(tf.reshape(neg_log_prob, (-1, 1)))
        # regularizer
        jacobian = tf.gradients(neg_log_prob, demo_state_tf)
        regularizer = tf.norm(jacobian[0], ord=2)
        self.loss = prm_loss_weight * neg_log_prob + reg_loss_weight * regularizer
        # optimizers
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def potential(self, o, g, u):
        state_tf = self._cast_concat_normalize_inputs(o, g, u)

        potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
        potential = tf.math.log(potential + tf.exp(-self.scale))
        potential = potential + self.scale  # shift
        potential = self.potential_weight * potential / self.scale  # scale
        return tf.cast(potential, tf.float32)

    def train(self, feed_dict={}):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def initialize_dataset(self):
        self.sess.run(self.demo_iter_init_tf)


class EnsNFShaping(Shaping):
    def __init__(
        self,
        sess,
        num_ens,
        gamma,
        max_u,
        max_num_transitions,
        batch_size,
        demo_dataset,
        o_stats,
        g_stats,
        num_masked,
        num_bijectors,
        layer_sizes,
        prm_loss_weight,
        reg_loss_weight,
        potential_weight,
    ):
        """
        Args:
            num_ens          (int)   - number of nf ensembles
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            prm_loss_weight  (float)
            reg_loss_weight  (float)
            potential_weight (float)
        """
        self.sess = sess
        self.learning_rate = 2e-4
        # setup ensemble
        self.nfs = []
        for i in range(num_ens):
            self.nfs.append(
                NFShaping(
                    sess=sess,
                    gamma=gamma,
                    max_u=max_u,
                    max_num_transitions=max_num_transitions,
                    batch_size=batch_size,
                    demo_dataset=demo_dataset,
                    o_stats=o_stats,
                    g_stats=g_stats,
                    num_masked=num_masked,
                    num_bijectors=num_bijectors,
                    layer_sizes=layer_sizes,
                    prm_loss_weight=prm_loss_weight,
                    reg_loss_weight=reg_loss_weight,
                    potential_weight=potential_weight,
                )
            )
        # loss
        self.loss = tf.reduce_sum([ens.loss for ens in self.nfs], axis=0)
        # optimizers
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # dataset initializer
        self.demo_iter_init_tf = [ens.demo_iter_init_tf for ens in self.nfs]

        super().__init__(gamma)

    def potential(self, o, g, u):
        # return the mean potential of all ens
        potential = tf.reduce_mean([ens.potential(o=o, g=g, u=u) for ens in self.nfs], axis=0)
        assert potential.shape[1] == 1
        return potential

    def train(self, feed_dict={}):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def initialize_dataset(self):
        self.sess.run(self.demo_iter_init_tf)


class GANShaping(Shaping):
    def __init__(
        self,
        sess,
        gamma,
        max_u,
        max_num_transitions,
        batch_size,
        demo_dataset,
        o_stats,
        g_stats,
        potential_weight,
        layer_sizes,
        latent_dim,
        gp_lambda,
        critic_iter,
    ):

        """
        GAN with Wasserstein distance plus gradient penalty.
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            potential_weight (float)
        """
        super().__init__(gamma)

        # Prepare parameters
        self.sess = sess
        self.max_u = max_u
        self.layer_sizes = layer_sizes
        self.critic_iter = critic_iter
        self.grad_target = 0.1
        self.potential_weight = potential_weight

        demo_dataset = demo_dataset.shuffle(max_num_transitions).batch(batch_size)
        demo_iter_tf = demo_dataset.make_initializable_iterator()
        self.demo_iter_init_tf = demo_iter_tf.initializer
        self.demo_inputs_tf = demo_iter_tf.get_next()
        # normalizer for goal and observation.
        self.o_stats = o_stats
        self.g_stats = g_stats

        demo_state_tf = self._concat_normalize_inputs(  # remove _normalize to not normalize the inputs
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )

        self.potential_weight = potential_weight
        self.critic_iter = critic_iter
        self.train_gen = 0  # counter

        # Generator & Discriminator
        self.generator = Generator(fc_layer_params=layer_sizes + [demo_state_tf.shape[-1]])
        self.discriminator = Discriminator(fc_layer_params=layer_sizes + [1])

        # Loss functions
        assert len(demo_state_tf.shape) >= 2
        fake_data = self.generator(tf.random.normal([tf.shape(demo_state_tf)[0], latent_dim]))
        disc_fake = self.discriminator(fake_data)
        disc_real = self.discriminator(demo_state_tf)
        # discriminator loss on generator (including gp loss)
        alpha = tf.random.uniform(
            shape=[tf.shape(demo_state_tf)[0]] + [1] * (len(demo_state_tf.shape) - 1), minval=0.0, maxval=1.0
        )
        interpolates = alpha * demo_state_tf + (1.0 - alpha) * fake_data
        disc_interpolates = self.discriminator(interpolates)
        gradients = tf.gradients(disc_interpolates, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - self.grad_target) ** 2)
        self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + gp_lambda * gradient_penalty
        # generator loss
        self.gen_cost = -tf.reduce_mean(disc_fake)

        # Train
        self.disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.disc_cost, var_list=self.discriminator.trainable_variables
        )
        self.gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.generator.trainable_variables
        )

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        state_tf = self._concat_normalize_inputs(o, g, u)  # remove _normalize to not normalize the inputs
        potential = self.discriminator(state_tf)
        potential = potential * self.potential_weight
        return potential

    def train(self, feed_dict={}):
        # train critic
        disc_cost, _ = self.sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
        # train generator
        if self.train_gen == 0:
            self.sess.run(self.gen_train_op, feed_dict=feed_dict)
        self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
        return disc_cost

    def initialize_dataset(self):
        self.sess.run(self.demo_iter_init_tf)


class EnsGANShaping(Shaping):
    def __init__(
        self,
        sess,
        num_ens,
        gamma,
        max_u,
        max_num_transitions,
        batch_size,
        demo_dataset,
        o_stats,
        g_stats,
        layer_sizes,
        latent_dim,
        gp_lambda,
        critic_iter,
        potential_weight,
    ):
        """
        Args:
            num_ens          (int)   - number of ensembles
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            potential_weight (float)
        """
        # Parameters for training
        self.sess = sess
        self.critic_iter = critic_iter
        self.train_gen = 0  # counter
        # Setup ensemble
        self.gans = []
        for i in range(num_ens):
            self.gans.append(
                GANShaping(
                    sess=sess,
                    gamma=gamma,
                    max_u=max_u,
                    max_num_transitions=max_num_transitions,
                    batch_size=batch_size,
                    demo_dataset=demo_dataset,
                    o_stats=o_stats,
                    g_stats=g_stats,
                    layer_sizes=layer_sizes,
                    latent_dim=latent_dim,
                    gp_lambda=gp_lambda,
                    critic_iter=critic_iter,
                    potential_weight=potential_weight,
                )
            )

        # Loss functions
        self.disc_cost = tf.reduce_sum([ens.disc_cost for ens in self.gans], axis=0)
        self.gen_cost = tf.reduce_sum([ens.gen_cost for ens in self.gans], axis=0)

        # Training
        self.disc_vars = list(itertools.chain(*[ens.discriminator.trainable_variables for ens in self.gans]))
        self.gen_vars = list(itertools.chain(*[ens.generator.trainable_variables for ens in self.gans]))
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.disc_cost, var_list=self.disc_vars
        )
        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.gen_vars
        )

        # dataset initializer
        self.demo_iter_init_tf = [ens.demo_iter_init_tf for ens in self.gans]

        super().__init__(gamma)

    def potential(self, o, g, u):
        # return the mean potential of all ens
        potential = tf.reduce_mean([ens.potential(o=o, g=g, u=u) for ens in self.gans], axis=0)
        assert potential.shape[1] == 1
        return potential

    def train(self, feed_dict={}):
        # train critic
        disc_cost, _ = self.sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
        # train generator
        if self.train_gen == 0:
            self.sess.run(self.gen_train_op)
        self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
        return disc_cost

    def initialize_dataset(self):
        self.sess.run(self.demo_iter_init_tf)