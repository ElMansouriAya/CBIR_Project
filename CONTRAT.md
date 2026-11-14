# Contrat de données B ↔ C et API C ↔ A (version 2)

**Projet** : Galerie photos + CBIR, Groupe 4 (Re-weighting Type-1)
**Statut** : accepté par B le ____ (à remplir). Tout changement est annoncé le jour même et ce fichier est mis à jour.

---

## 1. Fichiers livrés par B (dossier `features/`)

| Fichier | Contenu |
|---|---|
| `descriptors_raw.npy` | Matrice **(N, 172)**, `float32`, **non normalisée**, sans NaN ni inf |
| `image_mapping.json` | Une entrée par ligne de la matrice, **dans le même ordre** |

Exemple d'une entrée de `image_mapping.json` :

```json
{"index": 0, "image_id": 1, "filename": "0.jpg", "category": "Animals", "path": "dataset/Animals/0.jpg"}
```

- `index` = ligne dans la matrice. `image_id` = identifiant en base (A). Ce ne sont pas la même chose.
- Les `image_id` sont uniques.
- B livre les valeurs **brutes**. C normalise (z-score recalculé à chaque chargement, donc jamais périmé).

## 2. Ordre fixe des 172 dimensions

| Bloc | Colonnes | Dim. | Détail |
|---|---|---|---|
| Histogramme HSV | 0 à 127 | 128 | H 8 × S 4 × V 4, somme = 1 |
| Moments de couleur | 128 à 136 | 9 | moyenne, écart-type, asymétrie pour H, puis S, puis V |
| Couleurs dominantes | 137 à 145 | 9 | 3 couleurs HSV (valeurs dans [0,1]), **triées par taille de cluster décroissante** |
| GLCM | 146 à 161 | 16 | 4 propriétés × 4 angles (0°, 45°, 90°, 135°), distance 1 |
| LBP uniforme | 162 à 171 | 10 | P = 8, R = 1, histogramme normalisé (somme = 1) |

Ordre des propriétés GLCM : contraste, corrélation, énergie, homogénéité.
Ces valeurs sont centralisées dans `backend/cbir_config.py` (`BLOCKS`, `N_DIMS`) : si B change un paramètre, c'est le seul fichier à modifier côté C.

## 3. Règles communes

- Prétraitement identique pour toutes les images, requête comprise (même redimensionnement, même conversion RGB vers HSV).
- Le même code d'extraction sert pour la base et pour les images uploadées.
- Si une dimension n'est pas calculable (par exemple corrélation GLCM sur une image uniforme), elle vaut 0, jamais NaN.
- Si un paramètre change (bins, dimension totale), le tableau de la section 2 est mis à jour le jour même.

## 4. API (C ↔ A)

Blueprint Flask-RESTful autonome. A le monte avec le préfixe de son choix (ci-dessous `/api`). Toutes les requêtes et réponses sont en JSON.

### `POST /api/search`

Requête :

```json
{"query_image_id": 1, "scope": 25}
```

| Champ | Type | Obligatoire | Détail |
|---|---|---|---|
| `query_image_id` | entier | oui | image de la base, ou image uploadée (voir section 7) |
| `scope` | entier | non | de 1 à 100, défaut 25 |
| `method` | chaîne | non | `"per_iteration"` (défaut, fidèle au papier) ou `"cumulative"` |
| `exclude_query` | booléen | non | `true` retire l'image requête des résultats (évaluation) |

Réponse 200 :

```json
{"session_id": "f5490a9a...", "iteration": 1, "method": "per_iteration",
 "results": [{"image_id": 12, "distance": 0.31, "category": "Animals"}],
 "block_weights": {"hsv_hist": 1.0, "color_moments": 1.0, "dominant_colors": 1.0, "glcm": 1.0, "lbp": 1.0}}
```

`block_weights` donne le poids moyen par famille de descripteurs (pour l'affichage). Tous valent 1 à l'itération 1.

### `POST /api/feedback`

Requête :

```json
{"session_id": "f5490a9a...", "relevant_ids": [12, 40, 77]}
```

- `relevant_ids` : identifiants parmi les images **affichées à l'itération courante** (liste vide autorisée, 100 éléments au maximum).

Réponse 200 : même format que `/search`, avec `iteration` incrémenté et un champ en plus :

```json
{"feedback": {"mode": "full", "n_relevant": 3}}
```

`mode` : `"no_feedback"` (aucune image pertinente), `"query_only"` (une seule : requête mise à jour, poids conservés) ou `"full"` (requête et poids recalculés).

### Codes d'erreur

Corps : `{"error": "message"}`.

| Code | Cas |
|---|---|
| 400 | JSON invalide, champ manquant ou de mauvais type, `scope` hors limites, image non affichée à cette itération |
| 404 | `query_image_id` inconnu, session inconnue ou expirée |
| 409 | nombre maximal d'itérations atteint (7) |

Les sessions sont gardées en mémoire (1000 au maximum, les plus anciennes sont supprimées) et perdues au redémarrage du serveur.

## 5. Porte de validation (données de B)

Le chargeur applique ces contrôles et refuse les fichiers non conformes :

```python
from backend.services.loader import load_engine

engine, normalizer = load_engine("features")   # lève ValueError / FileNotFoundError
```

Contrôles : dimension 172, pas de NaN ni inf, mapping de la même taille que la matrice, `index` dans l'ordre, champs `index`, `image_id`, `category` présents.

## 6. Points que B doit confirmer ou modifier

- [ ] Dimension totale = 172 et ordre des blocs (section 2)
- [ ] Histogramme HSV : 8 × 4 × 4 = 128 bins
- [ ] Couleurs dominantes : K-means avec k = 3, triées par taille de cluster
- [ ] GLCM : 16 niveaux de gris, distance 1, 4 angles, 4 propriétés
- [ ] LBP : P = 8, R = 1, méthode uniforme (10 bins)
- [ ] Taille de redimensionnement avant extraction (proposition : 128 px, à préciser)
- [ ] Format de sortie : `.npy` + `image_mapping.json`, valeurs brutes
- [ ] Date de livraison de la première matrice (même partielle, par exemple 500 images)

## 7. Intégration dans l'application de A

```python
from backend.services.loader import load_engine
from backend.routes.search_routes import create_search_blueprint

engine, normalizer = load_engine("features")
bp = create_search_blueprint(
    engine,
    normalizer=normalizer,                  # obligatoire si vector_provider
    vector_provider=get_raw_vector,         # optionnel : images uploadées
    auth_decorators=[login_required],       # décorateurs d'authentification de A
)
app.register_blueprint(bp, url_prefix="/api")
```

- **Authentification** : `auth_decorators` s'applique aux deux endpoints (par exemple `flask_login.login_required`). Sans lui, les endpoints sont publics.
- **Images uploadées** : elles ne sont pas dans la matrice. `get_raw_vector(image_id)` doit retourner le vecteur **brut** de 172 valeurs (même extraction que B, non normalisé) ou `None` si l'image est inconnue. Le blueprint le normalise avec les paramètres de la base.
- **Isolation par utilisateur** : le blueprint ne vérifie pas qu'une image appartient à l'utilisateur connecté. Si nécessaire, A le contrôle dans un décorateur.
- **Test sans l'application de A** : `python scripts/run_demo_api.py` lance une API de démonstration sur le port 5001.
- **Dépendances** : `requirements.txt` (à la racine du dépôt).
