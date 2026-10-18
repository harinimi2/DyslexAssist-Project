import tensorflow as tf
from tensorflow.keras import layers


#Objectif: il faut que le graident soit le plus stable possible pour que l'IA soit optimal.

# 1.Création de la cellule LSTM (réseau de neurones, ici il y a 64 réseaux de neurones)
lstm_cell = layers.LSTMCell(64)  

# 2. Création d'un batch/groupe de séquences d'entrée aléatoires
x = tf.random.normal((32, 10, 16))  # (la taille du groupe =32, nb d'étapes dans le temps=10, le nb de valeurs que contient le vecteur =16)


# 3. Initialisation de l'état initial : [h0, c0]
# 3. On crée les états de départ de la mémoire (tout à zéro)
# h0 = état caché à t=0, c0 = mémoire interne à t=0
state = [tf.zeros((32, 64)), tf.zeros((32, 64))]   # forme : (taille du groupe, nombre d'unités), zero car on regarde la mémoire du réseau à l'instant t=0


# 4. Calcul avec suivi des gradients
with tf.GradientTape() as tape:
    tape.watch(lstm_cell.trainable_weights)  # surveille les poids de la cellule (le poids c'est comme un facteur devant un terme mathématique)

    # 4.1. Propagation manuelle sur chaque pas de temps
    for t in range(10):
        output, state = lstm_cell(x[:, t, :], state)

    # 4.2. On calcule une "fausse erreur" (perte) pour voir comment le réseau réagit
    # Ici, on imagine qu'on voulait que la sortie soit nulle
    target = tf.zeros_like(output)  # # cible = sortie attendue (ici que des zéros)
    loss = tf.reduce_mean(tf.square(output - target))  # MSE l'erreur moyen


# 5. Calcul des gradients de la perte par rapport aux poids
grads = tape.gradient(loss, lstm_cell.trainable_weights)

# 6. Affichage de la norme des gradients
print("Norme des gradients :", [tf.norm(g).numpy() for g in grads])
