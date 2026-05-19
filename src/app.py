import tkinter as tk
from PIL import Image, ImageDraw
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import charger_modele, predire

# --- Paramètres ---
TAILLE_CANVAS = 280  # pixels affichés à l'écran
TAILLE_IMAGE = 28  # pixels réels de MNIST
EPAISSEUR_TRAIT = 18  # épaisseur du trait de dessin


class ApplicationDessin:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de chiffres MNIST")
        self.modele = charger_modele()

        # Image PIL en arrière-plan (celle qu'on enverra au modèle)
        self.image_pil = Image.new("L", (TAILLE_CANVAS, TAILLE_CANVAS), color=255)
        self.draw = ImageDraw.Draw(self.image_pil)

        # --- Canvas de dessin ---
        self.canvas = tk.Canvas(
            root,
            width=TAILLE_CANVAS,
            height=TAILLE_CANVAS,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack(pady=10)

        # --- Label résultat ---
        self.label_resultat = tk.Label(
            root, text="Dessine un chiffre puis clique sur Prédire", font=("Arial", 16)
        )
        self.label_resultat.pack(pady=5)

        # --- Label confiance ---
        self.label_confiance = tk.Label(root, text="", font=("Arial", 12), fg="gray")
        self.label_confiance.pack()

        # --- Boutons ---
        frame_boutons = tk.Frame(root)
        frame_boutons.pack(pady=10)

        tk.Button(
            frame_boutons,
            text="Prédire",
            font=("Arial", 14),
            bg="#4CAF50",
            fg="white",
            padx=20,
            command=self.predire,
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            frame_boutons,
            text="Effacer",
            font=("Arial", 14),
            bg="#f44336",
            fg="white",
            padx=20,
            command=self.effacer,
        ).pack(side=tk.LEFT, padx=10)

        # --- Événements souris ---
        self.canvas.bind("<B1-Motion>", self.dessiner)
        self.canvas.bind("<ButtonRelease-1>", self.arret_dessin)
        self.dernier_x = None
        self.dernier_y = None

    def dessiner(self, event):
        x, y = event.x, event.y
        if self.dernier_x and self.dernier_y:
            # Dessine sur le canvas affiché
            self.canvas.create_line(
                self.dernier_x,
                self.dernier_y,
                x,
                y,
                width=EPAISSEUR_TRAIT,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True,
            )
            # Dessine aussi sur l'image PIL (celle envoyée au modèle)
            self.draw.line(
                [self.dernier_x, self.dernier_y, x, y], fill=0, width=EPAISSEUR_TRAIT
            )
        self.dernier_x = x
        self.dernier_y = y

    def arret_dessin(self, event):
        self.dernier_x = None
        self.dernier_y = None

    def predire(self):
        prediction, confiance = predire(self.image_pil, self.modele)
        self.label_resultat.config(
            text=f"Chiffre prédit : {prediction}", font=("Arial", 32, "bold")
        )
        self.label_confiance.config(text=f"Confiance : {confiance*100:.1f}%")

    def effacer(self):
        self.canvas.delete("all")
        self.image_pil = Image.new("L", (TAILLE_CANVAS, TAILLE_CANVAS), color=255)
        self.draw = ImageDraw.Draw(self.image_pil)
        self.label_resultat.config(
            text="Dessine un chiffre puis clique sur Prédire", font=("Arial", 16)
        )
        self.label_confiance.config(text="")


# --- Lancement ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationDessin(root)
    root.mainloop()
