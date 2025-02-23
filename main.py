import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from database import importData

(x_train, y_train), (x_test, y_test) = importData()

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)