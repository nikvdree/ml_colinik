import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    rn = nrVector.reshape(20,20, order='F')
    plt.matshow(rn)
    plt.show()
    pass

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    g = 1/(1+np.exp(-z))
    return g

    pass


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    # Hierbij kun je gebruik maken van de methode csr_matrix uit scipy 

    #YOUR CODE HERE
    
    # De y vector is 1-based, dus we trekken 1 af om het 0-based te maken
    cols = np.array(y - 1).flatten()  # Zorg ervoor dat cols een 1D array is
    
    # De rij indices zijn gewoon de indices van de elementen in y
    rows = np.arange(len(y)).flatten()  # Zorg ervoor dat rows een 1D array is
    
    # De data is gewoon een lijst van enen, omdat we een 1 willen zetten op elke positie (row, col)
    data = np.ones(len(y)).flatten()  # Zorg ervoor dat data een 1D array is
    
    # We vinden de breedte van de matrix door het maximale element in cols te nemen en er 1 bij op te tellen
    width = np.max(cols) + 1
    
    # We maken de csr_matrix en converteren het naar een dichte matrix
    y_matrix = csr_matrix((data, (rows, cols)), shape=(len(rows), width)).toarray()
    
    return y_matrix
    
    
    pass

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    #YOUR CODE HERE
    m = X.shape[0]

    # Voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    a1 = np.hstack((np.ones((m, 1)), X))

    # roep de sigmoid-functie van hierboven aan met a1 als actuele parameter: dit is de variabele a2
    a2 = sigmoid(a1.dot(Theta1.T))

    # voeg enen toe aan de matrix a2, dit is de input voor de laatste laag in het netwerk
    a2 = np.hstack((np.ones((m, 1)), a2))

    # roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke resultaat: de output van het netwerk aan de buitenste laag.
    a3 = sigmoid(a2.dot(Theta2.T))

    return a3
    

    pass



# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 

    #YOUR CODE HERE

    m = X.shape[0]  # aantal trainingsexamples
    num_labels = len(np.unique(y))  # bepaal het aantal unieke labels

    # Converteer y naar binaire matrix
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, int(y[i]) % num_labels] = 1  # We gebruiken % num_labels om label 10 om te zetten naar 0

    # Bereken de voorspellingen van het netwerk
    predictions = predict_number(X, Theta1, Theta2)

    # Bereken de kosten met de logistieke kostenfunctie
    cost = -1/m * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
    
    return cost


    pass



# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.

    pass

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = 1 #voorbeeldwaarde; dit moet je natuurlijk aanpassen naar de echte waarde van m

    for i in range(m): 
        #YOUR CODE HERE
        pass

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad
