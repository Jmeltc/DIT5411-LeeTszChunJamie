import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

TEST_DIR = "dataset/test"

# Load Test Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Load Model
model = tf.keras.models.load_model(
    "best_model.keras"
)

# Evaluate
loss, accuracy = model.evaluate(
    test_ds
)

print("\n==================")
print("Test Accuracy")
print("==================")
print(f"Accuracy: {accuracy*100:.2f}%")

print(f"Loss: {loss:.4f}")