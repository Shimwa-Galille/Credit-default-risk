import numpy as np
import matplotlib.pyplot as plt

# Création des vecteurs de coordonnées
x = np.linspace(1, 10, 100)
y = np.linspace(1, 10, 100)



# Création de la grille
X, Y = np.meshgrid(x, y)

# Calcul des valeurs correspondantes de Z
Z = np.sin(X) * np.cos(Y)

# Visualisation de la surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Ajouter des labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Titre
ax.set_title('Surface plot of Z = sin(X) * cos(Y)')

#plt.show()

# # Visualisation de la grille
# plt.plot(X, Y, color='blue', marker='o')  # Affiche les points de la grille
# plt.xlabel("X axis")
# plt.ylabel("Y axis")
# plt.title("Grille de points avec meshgrid")
# #plt.grid(True)  # Pour afficher une grille
# plt.show()



arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)