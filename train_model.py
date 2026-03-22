# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# import mlflow
# import mlflow.tensorflow
# mlflow.set_experiment("Assignment3_Rofida")

# #  Data Loading and Preprocessing
# def load_and_preprocess_data():
#     (X_train, _), (_, _) = mnist.load_data()
#     # Normalize to [-1, 1]
#     X_train = X_train.astype('float32')
#     X_train = (X_train - 127.5) / 127.5
#     # Add channel dimension (28,28) → (28,28,1)
#     X_train = np.expand_dims(X_train, axis=-1)
#     return X_train

# # Model Architecture Definitions
# def build_generator():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((7,7,128)))

#     model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same',
#                                      use_bias=False, activation='tanh'))
#     return model

# def build_discriminator():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1)) # No sigmoid for use with from_logits=True
#     return model

# # Loss and Optimizers
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     return real_loss + fake_loss

# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

# #  Training Step 
# @tf.function
# def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, noise_dim):
#     noise = tf.random.normal([batch_size, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)

#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     # --- ADDED: CALCULATE ACCURACY ---
#     # Convert logits to binary predictions (0 or 1)
#     real_preds = tf.cast(real_output > 0, tf.float32)
#     fake_preds = tf.cast(fake_output > 0, tf.float32)
    
#     # Discriminator is correct if it says 1 for real and 0 for fake
#     real_acc = tf.reduce_mean(tf.cast(tf.equal(real_preds, tf.ones_like(real_preds)), tf.float32))
#     fake_acc = tf.reduce_mean(tf.cast(tf.equal(fake_preds, tf.zeros_like(fake_preds)), tf.float32))
    
#     accuracy = (real_acc + fake_acc) / 2
#     return gen_loss, disc_loss, accuracy

# if __name__ == "__main__":
#     mlflow.set_experiment("Assignment3_Rofida")

#     # Hyperparameters
#     BUFFER_SIZE = 60000
#     BATCH_SIZE = 128
#     EPOCHS = 2 # Set low for testing, increase for real runs
#     noise_dim = 100
#     learning_rate = 0.0005

#     X_train = load_and_preprocess_data()
#     dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#     generator = build_generator()
#     discriminator = build_discriminator()
#     generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
#     discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

#     with mlflow.start_run(run_name="GAN_Training") as run:
#         # Capture the Run ID for the GitHub Action
#         current_run_id = run.info.run_id
        
#         mlflow.log_params({"learning_rate": learning_rate, "epochs": EPOCHS})

#         for epoch in range(EPOCHS):
#             epoch_acc = []
#             for image_batch in dataset:
#                 gen_loss, disc_loss, acc = train_step(
#                     image_batch, generator, discriminator, 
#                     generator_optimizer, discriminator_optimizer, 
#                     BATCH_SIZE, noise_dim
#                 )
#                 epoch_acc.append(acc)

#             # Log the average accuracy for the epoch
#             avg_acc = np.mean(epoch_acc)
#             mlflow.log_metric("accuracy", float(avg_acc), step=epoch)
#             mlflow.log_metric("gen_loss", float(gen_loss), step=epoch)
#             mlflow.log_metric("disc_loss", float(disc_loss), step=epoch)

#             print(f"Epoch {epoch+1}: Accuracy: {avg_acc:.4f}")

#         mlflow.tensorflow.log_model(generator, artifact_path="generator_model")
        
#         # CRITICAL: This line allows the GitHub Action to save the Run ID to the text file
#         print(f"RUN_ID: {current_run_id}")
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow

mlflow.set_experiment("Assignment3_Rofida")

def load_and_preprocess_data():
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)
    return X_train

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*128, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7,7,128)),
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    gen_optimizer.apply_gradients(zip(gen_tape.gradient(gen_loss, generator.trainable_variables), generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    
    # Calculate accuracy: How well does the discriminator tell real from fake?
    real_acc = tf.reduce_mean(tf.cast(real_output > 0, tf.float32))
    fake_acc = tf.reduce_mean(tf.cast(fake_output < 0, tf.float32))
    return gen_loss, disc_loss, (real_acc + fake_acc) / 2

if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 2 # Low for CI/CD testing
    noise_dim = 100
    
    dataset = tf.data.Dataset.from_tensor_slices(load_and_preprocess_data()).shuffle(60000).batch(BATCH_SIZE)
    generator, discriminator = build_generator(), build_discriminator()
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    with mlflow.start_run() as run:
        for epoch in range(EPOCHS):
            accuracies = []
            for batch in dataset:
                gl, dl, acc = train_step(batch, generator, discriminator, gen_opt, disc_opt, BATCH_SIZE, noise_dim)
                accuracies.append(acc)
            
            avg_acc = np.mean(accuracies)
            mlflow.log_metric("accuracy", float(avg_acc), step=epoch)
            print(f"Epoch {epoch+1} - Accuracy: {avg_acc:.4f}")

        mlflow.tensorflow.log_model(generator, "model")
        print(f"RUN_ID: {run.info.run_id}")