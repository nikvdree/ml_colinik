import numpy as np
import sys
import pickle

from uitwerkingen import *

'''
Onderstaande regels laden het pickle-bestand in waarin de data is opgeslagen.
Dit betreft een vector van 97 regels en 2 kolommen: de eerste kolom bevat
de populatie van steden, de tweede kolom bevat de winst van de vervoerder.
Nadat de data is ingeladen, wordt er een aantal variabelen gezet die in de
rest van het programma gebruikt kunnen worden:

  m = aantal datapunten
  X = de input-vector die gebruikt wordt voor de voorspellingen
  y = de werkelijke uitkomsten
  theta = de parameters van de voorspelling

Het is de bedoeling om de optimale waarden voor theta te vinden.
'''

with open('week1_data.pkl','rb') as f:
  data=pickle.load(f) 

m,n = data.shape

#Enen toevoegen als eerste kolom van X, zodat we op elke regel een
#vector hebben waarvan de grootte correspondeert met theta.
m,n = data.shape
X = np.c_[np.ones(m), data[:, [0]]]
y = data[:, [1]]
theta = np.zeros( (2, 1) )

# ========================  OPGAVE 1 ========================
print ("\nOpgave 1: draw_graph")
print ("Tekenen van de scatter plot van de data...")
if (len(sys.argv)>1 and sys.argv[1]=='skip'): print ('Slaan we over!')
else: draw_graph(data)
input ("Druk op return om verder te gaan...");

# ========================  OPGAVE 2 ========================
print ("\nOpgave 2: compute_cost")
J = compute_cost(X, y, theta);
print (f"De gevonden waarde van J is {J}")
print ("Deze waarde zou rond de 32.07 moeten liggen")

input ("Druk op return om verder te gaan...");

# ========================  OPGAVE 3 ========================
print ("\nOpgave 3a: gradient_descent")
alpha = 0.01
num_iters = 1500
theta = np.zeros( (1,2) )
theta, costs = gradient_descent(X, y, theta, alpha, num_iters) 
print (f"De gevonden waarde voor theta={theta[0]}")
print ("Deze waarde zou rond de [-3.63, 1.16] moeten liggen.")
print (f"De kost met deze waarde van theta is {compute_cost(X, y, theta.T)}")
print ("Die zou zo rond de 4.48 moeten liggen.")
input ("Druk op return om verder te gaan...")

# ========================  OPGAVE 4 ========================
print ("\nOpgave 3b: draw_costs")
draw_costs(costs)
input ("Druk op return om verder te gaan...")

# ========================  OPGAVE 5 ========================
print ("\nOpgave 4: contour plot")
contour_plot(X, y)
