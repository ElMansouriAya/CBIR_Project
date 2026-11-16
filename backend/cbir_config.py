"""Configuration commune du module C (recherche et retour de pertinence).

Les valeurs ci-dessous suivent le contrat de données B <-> C (CONTRAT.md).
Si B modifie un paramètre, on ne change QUE ce fichier : aucun autre module
ne doit écrire "172" en dur.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT_DIR / "features"
SYNTHETIC_DIR = FEATURES_DIR / "synthetic"

# --- Contrat de données (section 2 de CONTRAT.md) --------------------------
# Blocs de descripteurs : nom -> (colonne de début, colonne de fin exclue)
BLOCKS = {
    "hsv_hist": (0, 128),
    "color_moments": (128, 137),
    "dominant_colors": (137, 146),
    "glcm": (146, 162),
    "lbp": (162, 172),
}
N_DIMS = max(end for _, end in BLOCKS.values())  # 172

# --- Recherche et retour de pertinence -------------------------------------
SCOPE = 25          # nombre d'images retournées par itération
EPSILON = 1e-4      # constante epsilon de la formule Type-1 (papier, III-C)
STD_MIN = 1e-8      # en dessous, une dimension est considérée constante

# Garde-fous sur les poids Type-1 : choix d'implémentation, PAS du papier.
# Les poids bruts sont écrêtés dans [WEIGHT_MIN, WEIGHT_MAX], puis divisés
# par leur moyenne pour obtenir une moyenne de 1.
WEIGHT_MIN = 0.1
WEIGHT_MAX = 10.0

# --- Sessions de recherche --------------------------------------------------
MAX_ITERATIONS = 7   # le papier présente 7 itérations (tableau 2)
MAX_SCOPE = 100      # scope maximal accepté par requête (le défaut reste SCOPE)
MAX_SESSIONS = 1000  # sessions gardées en mémoire (les plus anciennes sont évincées)

# Deux façons de calculer le retour de pertinence au fil des itérations :
#   "per_iteration" : fidèle au papier, seule l'itération courante est utilisée
#   "cumulative"    : extension, toutes les images affichées/pertinentes de la
#                     session sont utilisées
METHODS = ("per_iteration", "cumulative")
DEFAULT_METHOD = "per_iteration"
