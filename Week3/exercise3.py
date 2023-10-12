import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import randint
import sys

"""
Als je een foutmelding krijgt met SSL, dan helpt dit onderstaande.
https://github.com/tensorflow/tensorflow/issues/33285
"""
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
"""
Tot hier (SSL error)
"""

from uitwerkingen import *

# ==============================================
# HELPER FUNCTIES
def plot_matrix(data):
    plt.figure()
    plt.matshow(data)
    plt.show()


# ==== Laden van de data en zetten van belangrijke variabelen ====
print ("Laden van de data...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
labels = ['T-shirt/topje', 'Broek', 'Pullover', 'Jurk', 'Jas', 'Sandalen', 'Shirt', 'Sneaker', 'Tas', 'Lage laars'] 
print ("Done.")

print (f"Formaat van de train_images: {train_images.shape}")
print (f"Formaat van de train_labels: {train_labels.shape}")
print (f"Formaat van de test_images: {test_images.shape}")
print (f"Formaat van de test_labels: {test_labels.shape}")
print (f"Grootte van de labels: {len(labels)}")

# ===============  OPGAVE 1 ======================
# ===============  OPGAVE 1a ======================
print ("Plotten van een willekeurig plaatje uit de trainings-dataset")
if (len(sys.argv)>1 and sys.argv[1]=='skip') :
    print ("Slaan we over")
else:
    rnd = randint(0, train_images.shape[0])
    hyp = labels[train_labels[rnd]]
    plot_image(train_images[rnd], hyp)

input ("Druk op enter om verder te gaan...")

# ===============  OPGAVE 1b ======================
X = np.array( ([1,2,3,4],[2,2,4,4],[4,3,2,1]) )
r = X/4
print ("Aanroepen van de methode scale_data met de matrix:")
print (X)
print (scale_data(X))
print ("Het resultaat zou gelijk moeten zijn aan:")
print (r)

train_images = scale_data(train_images)
test_images = scale_data(test_images)

input ("Druk op enter om verder te gaan...")


# ===============  OPGAVE 1c ======================
print ("")
print ("Aanmaken van het model.")
model = build_model()
print ("Trainen van het model...") 
model.fit(train_images, train_labels, epochs=6)
print ("Training afgerond.")

input ("Druk op enter om verder te gaan...")


# ===============  OPGAVE 2 ======================
print ("")
print ("Bepalen van de confusion matrix van het getrainde netwerk.")
pred = np.argmax(model.predict(test_images), axis=1)
conf_mtx = conf_matrix(test_labels, pred)

print ("De confusion matrix:") 
print (conf_mtx)

if (len(sys.argv)>1 and sys.argv[1]=='skip') :
    print ("Tekenen slaan we over")
else:
    plot_matrix(conf_mtx)
  

print ("Bepalen van de tp, tn, fp, fn")
metrics = conf_els(conf_mtx.numpy(), labels)
print (metrics)
print ("Bepalen van de scores:")
scores = conf_data(metrics)
print (scores)

print ("Klaar...")




