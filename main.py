from database import importData
from model import buildBaselineModel
from model import buildBetterModel
from model import trainModel

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


#print("Training data shape:", x_train.shape)
#print("Test data shape:", x_test.shape)

#print(y_train[0])

#fig = plt.figure(figsize=(12, 8))
#columns = 5
#rows = 3
#for i in range(1, columns*rows +1):
#  img = x_train[i]
#  fig.add_subplot(rows, columns, i)
#  plt.title(labels[y_train[i][0]])
#  plt.imshow(img, cmap='binary')
#plt.show()