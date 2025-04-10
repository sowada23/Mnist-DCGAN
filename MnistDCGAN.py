import glob 
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython import display
from PIL import Image
import datetime   # For a unique run id
import re         # For regular expression matching

# Generate a unique run ID (e.g., 20251009_153045)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define unique output directories based on run_id
BASE_OUTPUT_DIR = f"./Output_{run_id}"
EPOCH_IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "Mnistepoch")
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
LOSS_PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "Lossfolder") 
GRID_IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "GridImages")
GIF_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, f"training_grid_{run_id}.gif")

# Create the base directories
os.makedirs(EPOCH_IMAGES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_PLOTS_DIR, exist_ok=True)
os.makedirs(GRID_IMAGES_DIR, exist_ok=True)

# Load and preprocess MNIST data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  

# Define how many images passes at each epoch
BATCH_SIZE = 200
BUFFER_SIZE = 10000  # Shuffle buffer

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Define Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),  # Input: Noise vector
        layers.Reshape((7, 7, 256)),  # Reshape to small feature map
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(1, (4,4), strides=1, padding="same", activation="tanh")
    ])
    
    return model

generator = build_generator()
generator.summary()


#Define Descriminator
def build_discriminator():
    base_model = keras.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, padding="valid", input_shape=(28, 28, 1)),
        layers.LeakyReLU(0.2),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Conv2D(16, kernel_size=5, strides=1, padding="valid"),
        layers.LeakyReLU(0.2),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(1),  # Output probability (Real or Fake)
    ])
    
    return base_model

discriminator = build_discriminator()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define descriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Define generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, decay=0.5) 


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])  # Match batch size dynamically

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss.numpy(), disc_loss.numpy()


EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 25

# Fixed seed to track generator progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Save checkpoints with unique filenames using the unique output directory
def save_checkpoint(epoch):
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}_{run_id}")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.save(file_prefix=checkpoint_prefix)
    print(f"Checkpoint saved: {checkpoint_prefix}")
    

def get_last_checkpoint_epoch():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_epoch_")]
    if not checkpoint_files:
        return 0
    checkpoint_epochs = []
    for filename in checkpoint_files:
        match = re.search(r"ckpt_epoch_(\d+)_", filename)
        if match:
            checkpoint_epochs.append(int(match.group(1)))
    return max(checkpoint_epochs) if checkpoint_epochs else 0

def restore_latest_checkpoint():
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Checkpoint restored from: {latest_checkpoint}")
        match = re.search(r"ckpt_epoch_(\d+)_", latest_checkpoint)
        last_epoch = int(match.group(1)) if match else 0
        return last_epoch
    else:
        print("No checkpoint found. Training from scratch.")
        return 0


def generate_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        img = predictions[i, :, :, :] * 127.5 + 127.5
        plt.imshow(img.numpy().astype("uint8"), cmap="gray")
        plt.axis('off')
    plt.show()


def plot_loss(generator_losses, discriminator_losses):
    plt.clf()
    epochs_range = range(1, len(generator_losses) + 1)
    plt.plot(epochs_range, generator_losses, label='Generator Loss', color='dodgerblue')
    plt.plot(epochs_range, discriminator_losses, label='Discriminator Loss', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.pause(0.1)


def save_outputs(generator_losses, discriminator_losses, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.clf()
    epochs_range = range(1, len(generator_losses) + 1)
    plt.plot(epochs_range, generator_losses, label='Generator Loss', color='dodgerblue')
    plt.plot(epochs_range, discriminator_losses, label='Discriminator Loss', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator & Discriminator Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, f"loss_plot_{run_id}.png")
    plt.savefig(loss_plot_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Loss plot saved to: {loss_plot_path}")


# Global lists to store the average loss per epoch
generator_losses = []
discriminator_losses = []

def train(dataset, epochs, start_epoch=0):
    total_start = time.time()
    plt.ion()  # Enable interactive plotting
    for epoch in range(start_epoch, epochs):
        start = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)

        generator_losses.append(np.mean(epoch_gen_loss))
        discriminator_losses.append(np.mean(epoch_disc_loss))
            
        # Save a checkpoint every 10 epochs if needed
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch + 1)

        display.clear_output(wait=True)
        generate_images(generator, epoch + 1, seed)

        # Save 16 images for the current epoch in a unique epoch folder
        epoch_folder = os.path.join(EPOCH_IMAGES_DIR, f"epoch_{epoch + 1:04d}")
        os.makedirs(epoch_folder, exist_ok=True)
        predictions = generator(seed, training=False)
        for i in range(predictions.shape[0]):
            img_array = (predictions[i].numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=2)
            img = Image.fromarray(img_array)
            img.save(os.path.join(epoch_folder, f"image_{i+1:02d}.png"))

        print(f"Epoch {epoch + 1} completed. Time: {time.time() - start:.2f} sec")
        plot_loss(generator_losses, discriminator_losses)

    plt.ioff()
    plt.show()
    
    os.makedirs(LOSS_PLOTS_DIR, exist_ok=True)
    save_outputs(generator_losses, discriminator_losses, LOSS_PLOTS_DIR)
    print("Final epoch image and loss plot successfully saved.")
    
    total_end = time.time()
    total_duration = total_end - total_start
    print(f"\n Total training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


def build_grids_from_epoch_folders(base_dir, grid_output_dir):
    """
    Build a 5x5 grid image from images in each epoch folder.
    The grid image filename includes run_id so it does not conflict.
    """
    os.makedirs(grid_output_dir, exist_ok=True)
    epoch_folders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('epoch_')
    ], key=lambda x: int(x.split('_')[1]))
    for folder in epoch_folders:
        epoch_path = os.path.join(base_dir, folder)
        image_files = [f"image_{i:02d}.png" for i in range(1, 26)]
        missing = [f for f in image_files if not os.path.exists(os.path.join(epoch_path, f))]
        if missing:
            print(f"Skipping {folder}: missing images {missing}")
            continue
        imgs = [Image.open(os.path.join(epoch_path, f)).convert("RGB") for f in image_files]
        w, h = imgs[0].size
        grid_img = Image.new('RGB', (w * 5, h * 5))
        for idx, img in enumerate(imgs):
            row, col = divmod(idx, 5)
            grid_img.paste(img, (col * w, row * h))
        # Append run_id to the grid file to ensure uniqueness even if folder names repeat
        grid_img.save(os.path.join(grid_output_dir, f"{folder}_{run_id}.png"))
        print(f"Grid image saved for {folder}")
        
def create_gif_from_grids(grid_dir, output_gif_path, duration):
    """
    Create a GIF from grid images.
    The output gif filename already includes run_id.
    """
    frames = []
    grid_files = sorted(
        [f for f in os.listdir(grid_dir) if f.endswith('.png') and f.startswith('epoch_')],
        key=lambda x: int(x.split('_')[1])
    )
    for fname in grid_files:
        path = os.path.join(grid_dir, fname)
        img = Image.open(path).convert("RGB")
        frames.append(img)
    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to: {output_gif_path}")
    else:
        print("No grid images found.")

train(train_dataset, EPOCHS)

build_grids_from_epoch_folders(EPOCH_IMAGES_DIR, GRID_IMAGES_DIR)
create_gif_from_grids(GRID_IMAGES_DIR, GIF_OUTPUT_PATH, duration=10)