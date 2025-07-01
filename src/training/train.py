import tensorflow as tf
import os
from config import *
from utils.data_loader import load_images
from models.generator import build_generator
from models.discriminator import build_discriminator
from training.loss_functions import generator_loss, discriminator_loss

BUFFER_SIZE = 10000

images = load_images(DATA_PATH)
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train():
    for epoch in range(EPOCHS):
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Gen Loss: {g_loss.numpy()}, Disc Loss: {d_loss.numpy()}")
            save_image(epoch)
            generator.save(os.path.join(MODEL_SAVE_PATH, f"generator_{epoch}.h5"))

def save_image(epoch):
    noise = tf.random.normal([1, LATENT_DIM])
    image = generator(noise, training=False)[0].numpy()
    image = (image * 255).astype('uint8')
    img = image.fromarray(image)
    img.save(f"{GENERATED_IMG_PATH}img_epoch_{epoch}.png")

if __name__ == '__main__':
    os.makedirs(GENERATED_IMG_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    train()
