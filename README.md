# Galerie photos et recherche d'images par contenu (CBIR)

Projet du Master IASD, **Groupe 4**. Application web de gestion d'une galerie de photos, avec recherche d'images par similarité visuelle : recherche simple, ou recherche avec **retour de pertinence** par la méthode **Re-weighting Type-1**.

Base d'images : [GHIM-10K](https://www.kaggle.com/datasets/mcpython/ghim-10k) (10 000 images, 20 catégories de 500 images).

## Équipe et état d'avancement

- **EL MANSOURI AYA**
- **EL RHORBA AYA**
- **IBN IBRAHYM MERIEM**

## Fonctionnalités

**Tâche 1 : galerie et descripteurs **
- Authentification, upload, téléchargement et suppression d'images (une ou plusieurs).
- Hiérarchie de catégories, création d'images par transformation (recadrage, changement d'échelle, etc.).
- API REST (Flask, Flask-RESTful) qui calcule les descripteurs de couleur (histogramme, moments, couleurs dominantes) et de texture (matrice de co-occurrence GLCM, histogramme LBP), et affichage de ces descripteurs.

**Tâche 2 : recherche d'images similaires **
- Recherche simple : distance euclidienne entre le vecteur de l'image requête et ceux de la base.
- Recherche avec retour de pertinence : l'utilisateur marque les images pertinentes, le système met à jour les poids des descripteurs et la requête, puis relance la recherche (Re-weighting Type-1).

## Architecture

```
Navigateur
   |
Application web Flask  ---- blueprint /search et /feedback 
   |                                    |
API de descripteurs           moteur de recherche en mémoire
(Flask-RESTful)                  (matrice des descripteurs, sessions)
   |                                    |
   +------------- features/ ------------+
                  descriptors_raw.npy + image_mapping.json
```

## Structure du dépôt

```
CBIR_Project/
├── backend/
│   ├── cbir_config.py            # constantes du module de recherche (blocs de descripteurs, scope...)
│   ├── services/
│   │   ├── normalization.py      # normalisation z-score
│   │   ├── search.py             # distance euclidienne simple et pondérée
│   │   ├── relevance_feedback.py # Re-weighting Type-1
│   │   ├── session.py            # sessions de recherche itérative
│   │   ├── loader.py             # lecture et validation des fichiers de B
│   │   └── evaluation.py         # métriques et boucle d'évaluation
│   └── routes/
│       └── search_routes.py      # blueprint Flask-RESTful : POST /search, POST /feedback
├── frontend/                     # templates et fichiers statiques (A, B)
├── image_storage/                # dataset, uploads, images transformées (non versionné)
├── features/                     # descripteurs générés (non versionné)
├── scripts/
│   ├── make_synthetic_data.py    # jeu de test synthétique
│   ├── run_demo_api.py           # API de démonstration du module C
│   └── evaluate_retrieval.py     # évaluation, figures et tableaux
├── tests/
├── CONTRAT.md                    # contrat de données B <-> C et API C <-> A
├── requirements.txt
└── README.md
```


## Installation

```bash
git clone <url-du-depot>
cd CBIR_Project
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Téléchargez GHIM-10K depuis Kaggle et placez les images dans `image_storage/dataset/` (dossier non versionné).

## Utilisation

### Module C : recherche, retour de pertinence, évaluation

**Tests**
```bash
pytest -q
```

**Données de test synthétiques** (aucune dépendance envers B)
```bash
python scripts/make_synthetic_data.py --per-category 100 --noise 0.6
```

**API de démonstration** (blueprint monté sur `/api`, port 5001)
```bash
python scripts/run_demo_api.py
```
```bash
curl -X POST localhost:5001/api/search -H "Content-Type: application/json" \
     -d '{"query_image_id": 1, "scope": 25}'
```

**Évaluation** (Precision@25, mAP, courbes précision-rappel)
```bash
python scripts/evaluate_retrieval.py --features features --out results
```
Produit dans `results/` : tableaux Markdown prêts pour le rapport, deux figures PNG, des fichiers CSV et un résumé JSON des paramètres. Sur données synthétiques, ajoutez `--features features/synthetic --out results_synthetic` : ces chiffres servent uniquement à vérifier le fonctionnement et ne doivent pas figurer dans le rapport.

**Pour A : monter la recherche dans l'application**
```python
from backend.services.loader import load_engine
from backend.routes.search_routes import create_search_blueprint

engine, normalizer = load_engine("features")
bp = create_search_blueprint(engine, normalizer=normalizer,
                             auth_decorators=[login_required])
app.register_blueprint(bp, url_prefix="/api")
```
Détails (images uploadées, codes d'erreur, format des réponses) : section 7 de [CONTRAT.md](CONTRAT.md).

**Pour B : livrer les descripteurs**

Déposer `features/descriptors_raw.npy` (matrice N x 172, valeurs brutes) et `features/image_mapping.json` selon [CONTRAT.md](CONTRAT.md), puis vérifier :
```bash
python -c "from backend.services.loader import load_engine; load_engine('features'); print('Contrat respecté')"
```

### Module A et module B

**À compléter** : commandes de lancement de l'application web, de l'API de descripteurs et du script d'extraction.

## Vecteur de descripteurs

Chaque image est représentée par un vecteur de 172 dimensions (détail et ordre fixe dans [CONTRAT.md](CONTRAT.md)) :

| Bloc | Dimensions |
|---|---|
| Histogramme HSV (8 x 4 x 4) | 128 |
| Moments de couleur (moyenne, écart-type, asymétrie sur H, S, V) | 9 |
| Couleurs dominantes (K-means, k = 3) | 9 |
| GLCM (4 propriétés x 4 angles) | 16 |
| Histogramme LBP uniforme (P = 8, R = 1) | 10 |

## Méthode : Re-weighting Type-1

Référence : Renuka Devi S. M., *Reweighing Strategies for Relevance Feedback in CBIR*, IJARCCE, vol. 5, no 11, novembre 2016, section III-C.

À chaque itération de retour de pertinence, le poids de chaque composante *i* est mis à jour (équation 5 du papier) :

$$w_i^{k+1} = \frac{\epsilon + \sigma_{N_r,i}^{k}}{\epsilon + \sigma_{rel,i}^{k}}, \qquad \epsilon = 0.0001$$

où $\sigma_{N_r,i}$ est l'écart-type de la composante *i* sur les images retournées, et $\sigma_{rel,i}$ son écart-type sur les images jugées pertinentes. Une composante peu dispersée parmi les images pertinentes reçoit un poids élevé.

La recherche utilise la distance euclidienne pondérée (équation 3) :

$$\mathrm{Dist}(X,Y) = \sqrt{\frac{\sum_{i=1}^{d} w_i\,(x_i - y_i)^2}{\sum_{i=1}^{d} w_i}}$$

La requête de l'itération suivante est la moyenne des vecteurs des images pertinentes. À l'itération 1, tous les poids valent 1 (recherche simple).

**Choix d'implémentation qui ne viennent pas du papier**
- Les descripteurs sont normalisés (z-score par dimension) avant tout calcul de distance.
- Les poids bruts sont écrêtés entre 0,1 et 10, puis ramenés à une moyenne de 1, pour éviter qu'un écart-type nul n'en fasse exploser certains.
- Sans image pertinente, la requête et les poids sont conservés. Avec une seule, seule la requête est mise à jour (un écart-type sur un seul échantillon n'a pas de sens).
- Deux modes de calcul : `per_iteration` (fidèle au papier : l'itération courante seulement) et `cumulative` (extension : toutes les images vues et pertinentes depuis le début de la session).

## Évaluation

- **Requêtes** : images de la base, tirées de façon stratifiée par catégorie (graine fixe).
- **Vérité terrain** : une image est pertinente si elle a la même catégorie que la requête. La requête est exclue des résultats.
- **Feedback simulé (« oracle »)** : parmi les images affichées, celles de la bonne catégorie sont déclarées pertinentes. C'est un retour idéal, plus favorable que celui d'un vrai utilisateur.
- **Mesures** : Precision@25 par itération (7 itérations), mAP et courbes précision-rappel sur le classement complet, avec intervalles de confiance à 95 %.
- **Méthodes comparées** : recherche simple (itération 1), requête moyennée sans poids, Type-1 fidèle au papier, Type-1 cumulatif.

## Tests

`pytest -q` lance les tests unitaires et d'intégration du module C : normalisation, recherche, Type-1 (cas calculés à la main), sessions, API Flask, chargeur de données, évaluation. Un test vérifie que la boucle d'évaluation affiche exactement les mêmes images que le moteur de sessions utilisé par l'application. Un test est ignoré tant que `scripts/make_synthetic_data.py` n'a pas été lancé.

## Références

- Renuka Devi S. M., *Reweighing Strategies for Relevance Feedback in CBIR*, International Journal of Advanced Research in Computer and Communication Engineering, vol. 5, no 11, novembre 2016. DOI 10.17148/IJARCCE.2016.511118.
- GHIM-10K, jeu de données d'images en 20 catégories (Kaggle, `mcpython/ghim-10k`).
