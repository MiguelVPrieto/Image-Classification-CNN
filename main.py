import numpy as np
import pandas as pd
from database import importData
from model import buildBaselineModel
from model import buildBetterModel
from model import trainModel

print("Loading data...")
(x_train, y_train), (x_test, y_test) = importData()

labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]

#print("Preprocessing images...")
#x_train2 = preprocessImages(x_train)

print("Building models...")
baselineModel = buildBaselineModel()
improvedModel = buildBetterModel()

print("Training models...")
historyBaseline = trainModel(model=baselineModel, images=x_train, labels=y_train, batch_size=64, epochs=5, name='Baseline')
historyImproved = trainModel(model=improvedModel, images=x_train, labels=y_train, batch_size=64, epochs=5, name='Improved')
print("Training complete!")

print("Loading results...")
accuracyBaseline = historyBaseline.history['val_accuracy'][-1]
accuracyBaseline = accuracyBaseline * 100
accuracyBaseline = np.round(accuracyBaseline, 2)
accuracyImproved = historyImproved.history['val_accuracy'][-1]
accuracyImproved = accuracyImproved * 100
accuracyImproved = np.round(accuracyImproved, 2)

print("Saving results...")
try:
    file = pd.read_csv("results.csv")
    lastTrain = file.iloc[-1, 0]
    lastTrain += 1
except (FileNotFoundError, IndexError):
    print("No previous results found. Creating a new file.")
    lastTrain = 1
    file = pd.DataFrame(columns=["Train ID", "Baseline Model Accuracy (%)", "Improved Model Accuracy (%)"])

newData = pd.DataFrame([[lastTrain, accuracyBaseline, accuracyImproved]], columns=["Train ID", "Baseline Model Accuracy (%)", "Improved Model Accuracy (%)"])
file = pd.concat([file, newData], ignore_index=True)
file.to_csv("results.csv", index=False)
print("All processes finished!")