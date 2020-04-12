import numpy as np
import matplotlib.pyplot as plt
from simple_GAN.utils import sample_data, relu, sigmoid, tanh, DataChunk

EPSILON = 1e-8


class GANSubnet:
    def __init__(self, batch_size=32, layer_sizes=(64, 32, 2), output_activation_func=None):
        self.batch_size = batch_size
        if output_activation_func is None:
            self.output_activation_func = sigmoid
        else:
            self.output_activation_func = output_activation_func
        self.input_shape = None
        self.initialized = False
        self.layer_sizes = list(layer_sizes)
        self.weights = []
        self.biases = []
        self.logits = []
        self.activations = []

    def _initialize(self):
        # Xavier parameter initialization
        self.layer_sizes = (self.layer_sizes[::-1] + [self.input_shape])[::-1]
        for idx in range(len(self.layer_sizes) - 1):
            input_dim = self.layer_sizes[idx]
            output_dim = self.layer_sizes[idx + 1]
            w = np.random.randn(input_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / output_dim)
            self.weights.append(w)
            b = np.zeros(output_dim).astype(np.float32)
            self.biases.append(b)

        self.initialized = True

    def infer(self, data_input):
        self.input_shape = data_input.shape[1]
        if not self.initialized:
            self._initialize()

        self.logits = []
        self.activations = []

        output = data_input
        for idx in range(len(self.weights)):
            output = (output.dot(self.weights[idx])) + self.biases[idx]
            self.logits.append(output)
            if idx < len(self.weights) - 1:
                output = relu(output)
                self.activations.append(output)

        output = self.output_activation_func(output)
        self.activations.append(output)

        return output


class GAN:
    def __init__(self, batch_size=32, learning_rate=1e-3, gen_layers=(64, 32, 2),
                 dis_layers=(64, 32, 1)):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.generator = GANSubnet(batch_size=self.batch_size, layer_sizes=gen_layers,
                                   output_activation_func=tanh)
        self.discriminator = GANSubnet(batch_size=self.batch_size, layer_sizes=dis_layers,
                                       output_activation_func=sigmoid)

        self.generator_loss = lambda z: -np.mean(np.log(z + EPSILON))
        self.discriminator_loss_fake = lambda z: -np.mean(np.log(1 - z + EPSILON))
        self.discriminator_loss_real = lambda x: -np.mean(np.log(x + EPSILON))

    def generate(self, data_input):
        reshaped_input = np.reshape(data_input, (self.batch_size, -1))
        result = self.generator.infer(reshaped_input)

        return result

    def discriminate(self, data_input):
        reshaped_input = np.reshape(data_input, (self.batch_size, -1))
        result = self.discriminator.infer(reshaped_input)

        return result

    def _back_propagation_generator(self, noise_input, fake_logits, fake_activations):
        dis_output_fake = fake_activations[-1]
        dis_logits_fake = fake_logits[-1]

        # get the derivative of -log[D(G(z))] w.r.t. to parameters W and b of G
        # where D is the discriminator and G the generator

        # first get the derivative of -log[D(G(z))] w.r.t. G(z)
        outer_derivative = -1.0 / (dis_output_fake + EPSILON)
        # iterate over discriminator layers, starting with output one
        for num, w in enumerate(reversed(self.discriminator.weights)):
            activation_func = relu
            if num == 0:
                activation_func = self.discriminator.output_activation_func
            logit = list(reversed(fake_logits))[num]
            outer_derivative *= activation_func(logit, derivative=True)
            outer_derivative = outer_derivative.dot(w.T)

        # second get the derivatives of G(z) w.r.t. to the parameters W and b
        derivative = outer_derivative
        weight_gradients = []
        bias_gradients = []
        for num, w in enumerate(reversed(self.generator.weights)):
            activation_func = relu
            if num == 0:
                activation_func = self.generator.output_activation_func

            gen_logits = list(reversed(self.generator.logits))[num]

            derivative *= activation_func(gen_logits, derivative=True)
            if num < len(self.generator.weights) - 1:
                prev_layer_activation = np.expand_dims(list(reversed(self.generator.activations))[num + 1], axis=-1)
            else:
                prev_layer_activation = np.expand_dims(noise_input, axis=-1)
            _derivative = np.expand_dims(derivative, axis=1)

            weight_gradients.append(prev_layer_activation @ _derivative)
            bias_gradients.append(derivative.copy())

            derivative = derivative.dot(w.T)

        weight_gradients = weight_gradients[::-1]
        bias_gradients = bias_gradients[::-1]

        # update generator parameters with gradients
        for idx in range(len(self.generator.weights)):
            for num in range(self.batch_size):
                self.generator.weights[idx] = self.generator.weights[idx] \
                                              - self.learning_rate * weight_gradients[idx][num]
                self.generator.biases[idx] = self.generator.biases[idx] - self.learning_rate * bias_gradients[idx][num]

    def _back_propagation_discriminator(self, fake_input, real_input,
                                        dis_logits_real, dis_activations_real,
                                        dis_logits_fake, dis_activations_fake):

        # get the derivative of -log[D(x)] - log[1 - D(G(z))]  w.r.t. to parameters W and b of D
        # the term -log[D(x)] and according derivatives will be indexed with 1
        # same for -log[1 - D(G(z))] with index 2

        dis_output_real = dis_activations_real[-1]
        dis_output_fake = dis_activations_fake[-1]

        outer_derivative_1 = -1. / (dis_output_real + EPSILON)
        outer_derivative_2 = -1. / (dis_output_fake - 1 + EPSILON)

        # get gradients according to first term -log[D(x)]
        weight_gradients_1 = []
        bias_gradients_1 = []
        for num, w in enumerate(reversed(self.discriminator.weights)):
            activation_func = relu
            if num == 0:
                activation_func = self.discriminator.output_activation_func

            logits = list(reversed(dis_logits_real))[num]

            outer_derivative_1 *= activation_func(logits, derivative=True)
            if num < len(self.discriminator.weights) - 1:
                prev_layer_activation = np.expand_dims(list(reversed(dis_activations_real))[num + 1], axis=-1)
            else:
                prev_layer_activation = np.expand_dims(real_input, axis=-1)
            _derivative = np.expand_dims(outer_derivative_1, axis=1)

            weight_gradients_1.append(prev_layer_activation @ _derivative)
            bias_gradients_1.append(outer_derivative_1.copy())

            outer_derivative_1 = outer_derivative_1.dot(w.T)

        weight_gradients_1 = weight_gradients_1[::-1]
        bias_gradients_1 = bias_gradients_1[::-1]

        weight_gradients_2 = []
        bias_gradients_2 = []
        for num, w in enumerate(reversed(self.discriminator.weights)):
            activation_func = relu
            if num == 0:
                activation_func = self.discriminator.output_activation_func

            logits = list(reversed(dis_logits_fake))[num]

            outer_derivative_2 *= activation_func(logits, derivative=True)
            if num < len(self.discriminator.weights) - 1:
                prev_layer_activation = np.expand_dims(list(reversed(dis_activations_fake))[num + 1], axis=-1)
            else:
                prev_layer_activation = np.expand_dims(fake_input, axis=-1)
            _derivative = np.expand_dims(outer_derivative_2, axis=1)

            weight_gradients_2.append(prev_layer_activation @ _derivative)
            bias_gradients_2.append(outer_derivative_2.copy())

            outer_derivative_2 = outer_derivative_2.dot(w.T)

        weight_gradients_2 = weight_gradients_2[::-1]
        bias_gradients_2 = bias_gradients_2[::-1]

        weight_gradients = []
        bias_gradients = []
        for i in range(len(weight_gradients_1)):
            weight_gradients.append(weight_gradients_1[i] + weight_gradients_2[i])
            bias_gradients.append(bias_gradients_1[i] + bias_gradients_2[i])

        for idx in range(len(self.discriminator.weights)):
            for num in range(self.batch_size):
                self.discriminator.weights[idx] = self.discriminator.weights[idx] \
                                                  - self.learning_rate * weight_gradients[idx][num]
                self.discriminator.biases[idx] = self.discriminator.biases[idx] - self.learning_rate \
                                                 * bias_gradients[idx][num]

    def train(self, data_chunk, epochs=100, evaluation_step=100):
        batch_index = 0
        gen_loss = []
        dis_loss_real = []
        dis_loss_fake = []
        for epoch in range(epochs):
            for train_batch, _ in data_chunk.iter_batches(self.batch_size):
                batch_index += 1
                noise = np.random.uniform(-1, 1, [self.batch_size, 100]).astype(np.float32)
                gen_output = self.generate(noise)

                dis_output_real = self.discriminate(train_batch)
                dis_logits_real, dis_activations_real = self.discriminator.logits, self.discriminator.activations
                dis_output_fake = self.discriminate(gen_output)
                dis_logits_fake, dis_activations_fake = self.discriminator.logits, self.discriminator.activations

                # NOW: Backpropagation for generator and discriminator
                self._back_propagation_discriminator(gen_output, train_batch, dis_logits_real, dis_activations_real,
                                                         dis_logits_fake, dis_activations_fake)

                self._back_propagation_generator(noise, dis_logits_fake, dis_activations_fake)

                if batch_index % evaluation_step == 0:
                    gen_loss.append(self.generator_loss(dis_output_fake))
                    dis_loss_real.append(self.discriminator_loss_real(dis_output_real))
                    dis_loss_fake.append(self.discriminator_loss_fake(dis_output_fake))

        X = np.array(range(batch_index // evaluation_step)) * evaluation_step
        plt.plot(X, gen_loss, "r-", label="gen")
        plt.plot(X, dis_loss_real, "b-", label="dis real")
        plt.plot(X, dis_loss_fake, "g-", label="dis fake")
        plt.xlabel("Number of processed batches")
        plt.ylabel("Error")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    data, labels = sample_data(1000, 3)
    max_value = np.max(np.abs(data))
    normed_data = data / max_value
    chunk = DataChunk(normed_data, labels)
    net = GAN()
    net.train(chunk, 2000)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=3)
    plt.show()
