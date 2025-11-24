# ============ data_loader.py ============
Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
MaxPooling2D(2, 2),


Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),


Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),


Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
return model




# ============ train.py ============


def train_model(model, X_train, y_train, epochs=10):
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
model.save("model.h5")
return history




# ============ evaluate.py ============


from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




# ============ main.py ============


from data_loader import load_dataset
from model import create_cnn
from train import train_model
from evaluate import evaluate_model
import tensorflow as tf


if _name_ == "_main_":
print("Loading dataset...")
X_train, X_test, y_train, y_test = load_dataset("data/")


print("Creating model...")
model = create_cnn(input_shape=X_train[0].shape, num_classes=2)


print("Training model...")
history = train_model(model, X_train, y_train, epochs=10)


print("Evaluating model...")
evaluate_model(model, X_test, y_test)


print("Project execution completed.")

