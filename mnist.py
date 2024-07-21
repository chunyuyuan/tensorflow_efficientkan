from efficient_kan import KAN


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers.experimental import AdamW
from tqdm import tqdm
# Load MNIST
(train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28*28).astype('float32') / 255.0
val_images = val_images.reshape(-1, 28*28).astype('float32') / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(64)

# Define model
model = KAN([28*28, 64, 10])
criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
'''
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)


# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=500, decay_rate=0.8)
'''
optimizer = AdamW(learning_rate=5e-3, weight_decay=1e-5)  # Adjusted values

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=1000,
    decay_rate=0.9
)
# Compile the model
model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

# Training loop
for epoch in range(20):
    # Train
    model.reset_metrics()
    train_loss = 0
    
    train_batches = 0
    for images, labels in tqdm(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = criterion(labels, predictions)
            batch_loss = tf.reduce_mean(loss)
            train_loss += batch_loss.numpy()
            train_batches += 1
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss /= train_batches
    print(f"train Loss: {train_loss}")
    # Validation
    

    # Assuming 'val_dataset' is batched and ready to use
    val_loss = 0
    val_accuracy = 0
    total_batches = 0

    # No gradient computation needed during validation
    for images, labels in val_dataset:
        # Reshape images if necessary (like in your PyTorch example, flatten them)
        images = tf.reshape(images, [-1, 28 * 28])  # Assuming each image is 28x28

        # Get model predictions
        preds = model(images, training=False)  # Set training=False to run the model in inference mode
        labels = tf.cast(labels, tf.int64)
        # Calculate loss (make sure to specify reduction as 'none' to sum it manually later)
        batch_loss = tf.reduce_mean(criterion(labels, preds))

        # Calculate batch accuracy
        batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), labels), tf.float32))

        # Accumulate loss and accuracy
        val_loss += batch_loss.numpy()
        val_accuracy += batch_accuracy.numpy()
        total_batches += 1

    # Average the loss and accuracy over all batches
    val_loss /= total_batches
    val_accuracy /= total_batches

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Update learning rate
    optimizer.learning_rate = lr_schedule(epoch)

