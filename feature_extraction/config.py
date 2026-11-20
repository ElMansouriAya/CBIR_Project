"""Paramètres d'implémentation du module B.

Les dimensions et paramètres indiqués comme contractuels viennent de
``CONTRAT.md``. Les réglages de redimensionnement et de quantification sont
des choix provisoires propres à B, centralisés ici pour faciliter leur revue.
"""

from math import pi

# Paramètres explicitement définis par le contrat.
N_DIMS = 172
HSV_BINS = (8, 4, 4)
N_COLOR_MOMENTS = 9
N_DOMINANT_COLORS = 3
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
GLCM_DISTANCE = 1
GLCM_ANGLES = (0.0, pi / 4.0, pi / 2.0, 3.0 * pi / 4.0)
GLCM_PROPERTIES = ("contrast", "correlation", "energy", "homogeneity")

# Réglages choisis par B, car le contrat ne les spécifie pas.
IMAGE_SIZE = (256, 256)  # (largeur, hauteur), avec conservation du ratio.
LETTERBOX_RGB = (0, 0, 0)  # fond noir autour de l'image redimensionnée.
PREPROCESSING_LETTERBOX = "letterbox"
PREPROCESSING_STRETCH = "stretch"
PREPROCESSING_MODES = (PREPROCESSING_LETTERBOX, PREPROCESSING_STRETCH)
GLCM_LEVELS = 32
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 100
KMEANS_MAX_SAMPLES = 10000  # échantillon déterministe pour limiter le coût.

# Contrôle de cohérence local : le membre C possède sa propre configuration.
BLOCK_SLICES = {
    "hsv_hist": slice(0, 128),
    "color_moments": slice(128, 137),
    "dominant_colors": slice(137, 146),
    "glcm": slice(146, 162),
    "lbp": slice(162, 172),
}
