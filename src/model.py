import tensorflow as tf

class TVAE(tf.keras.Model):
    """Tabular Variational Autoencoder (TVAE) in modern TensorFlow 2.x execution."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super(TVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(units=input_dim, activation="tanh"),
            tf.keras.layers.Dense(units=latent_dim, activation="relu"),
            tf.keras.layers.Dense(units=latent_dim * 2),  # Mean (mu) and Logvar
        ], name="inference_net")

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim, activation="relu"),
            tf.keras.layers.Dense(units=input_dim, activation="tanh"),
            tf.keras.layers.Dense(units=input_dim),
        ], name="generative_net")

    def encode(self, x: tf.Tensor):
        model_output = self.inference_net(x)
        mean, logvar = tf.split(model_output, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
        logits = self.generative_net(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def sample(self, eps: tf.Tensor = None) -> tf.Tensor:
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
