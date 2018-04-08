import os
import scipy.misc
from nets.util.layers import *
from nets.util.read_anime_face import *
from nets.util.mnist import *
from nets.util.anime import *

def save_images(images, size, path):
    height, width = images.shape[1], images.shape[2]
    merged_image = np.zeros((height * size[0], width * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merged_image[j * height: (j + 1) * height, i * width: (i + 1) * width, :] = image
    return scipy.misc.imsave(path, merged_image)

class WGAN_GP(object):
    def __init__(self):
        self.IMAGE_SIZE = 28
        self.IMAGE_CHANNEL = 1
        self.g_reuse = False
        self.d_reuse = False

    def generator(self, name, size):
        with tf.variable_scope(name, reuse=self.g_reuse):
            NOISE_SIZE = 128
            noise = tf.random_normal([size, NOISE_SIZE])
            fc = fully_connected(noise, output_size=2 * 2 * 8 * self.IMAGE_SIZE, name='g_fc')
            reshaped_fc = tf.reshape(fc, [size, 2, 2, 8 * self.IMAGE_SIZE])
            deconv1 = deconv2d(reshaped_fc, output_shape=[size, 4, 4, 4 * self.IMAGE_SIZE], name='g_deconv1')
            deconv1 = tf.nn.relu(deconv1)
            deconv2 = deconv2d(deconv1, output_shape=[size, 7, 7, 2 * self.IMAGE_SIZE], name='g_deconv2')
            deconv2 = tf.nn.relu(deconv2)
            deconv3 = deconv2d(deconv2, output_shape=[size, 14, 14, self.IMAGE_SIZE], name='g_deconv3')
            deconv3 = tf.nn.relu(deconv3)
            deconv4 = deconv2d(deconv3, output_shape=[size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_CHANNEL], name='g_deconv4')
            output = tf.nn.sigmoid(deconv4)
            self.g_reuse = True
            return tf.reshape(output, [size, self.IMAGE_SIZE * self.IMAGE_SIZE * self.IMAGE_CHANNEL])

    def discriminator(self, name, image):
        with tf.variable_scope(name, reuse=self.d_reuse):
            reshaped_image = tf.reshape(image, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_CHANNEL])
            conv1 = conv2d(reshaped_image, output_channel=self.IMAGE_SIZE, name='d_conv1')
            conv1 = leaky_relu(conv1)
            conv2 = conv2d(conv1, output_channel=2 * self.IMAGE_SIZE, name='d_conv2')
            conv2 = leaky_relu(conv2)
            conv3 = conv2d(conv2, output_channel=4 * self.IMAGE_SIZE, name='d_conv3')
            conv3 = leaky_relu(conv3)
            shape = conv3.get_shape().as_list()
            flatten = tf.reshape(conv3, [-1, shape[1] * shape[2] * shape[3]])
            output = fully_connected(flatten, output_size=1, name='d_fc')
            self.d_reuse = True
            return output

    def train(self, data_source, save_path, epoch = 400, iters = 100, batch_size = 64, ratio = 10):
        with tf.variable_scope(tf.get_variable_scope()):
            real_image = tf.placeholder(tf.float32, shape=[batch_size, self.IMAGE_SIZE * self.IMAGE_SIZE * self.IMAGE_CHANNEL])
            with tf.variable_scope(tf.get_variable_scope()):
                fake_image = self.generator('generator', batch_size)
                disc_real = self.discriminator('discriminator', image=real_image)
                disc_fake = self.discriminator('discriminator', image=fake_image)
            train_vars = tf.trainable_variables()
            d_vars = [var for var in train_vars if 'd_' in var.name]
            g_vars = [var for var in train_vars if 'g_' in var.name]
            d_loss = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
            g_loss = tf.reduce_mean(disc_fake)
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
            differences = fake_image - real_image
            interpolates = real_image + (alpha * differences)
            gradients = tf.gradients(self.discriminator('discriminator', image=interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
            d_loss += ratio * gradient_penalty
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                g_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                    beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)
                d_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                    beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
            saver = tf.train.Saver()
            sess = tf.InteractiveSession()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                for j in range(iters):
                    image = data_source.next_batch(batch_size)
                    for k in range(5):
                        _, d_cost = sess.run([d_train_op, d_loss], feed_dict={real_image: image})
                    _, g_cost = sess.run([g_train_op, g_loss])
                    print('[%4d:%4d/%4d] d_loss: %.8f, g_loss: %.8f' % (i + 1, j + 1, iters, d_cost, g_cost))
                with tf.variable_scope(tf.get_variable_scope()):
                    samples = self.generator('generator', 64)
                    samples = tf.reshape(samples, shape=[64, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_CHANNEL])
                    samples = sess.run(samples)
                    save_images(samples, [8, 8], save_path + '/' + 'sample_%d_epoch.png' % (i + 1))
                    if i == epoch - 1:
                        checkpoint_path = os.path.join(os.getcwd(), 'wgan-gp.ckpt')
                        saver.save(sess, checkpoint_path, global_step=i + 1)
            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    wgan = WGAN_GP()
    anime = Anime('../../datasets/anime/anime.npy')
    save_path = '../../generated_images/face'
    wgan.train(anime, save_path)