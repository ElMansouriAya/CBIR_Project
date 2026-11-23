# VisualVault

Application web de gestion de galerie et de recherche d’images par le contenu (CBIR). VisualVault compare les caractéristiques visuelles des images et permet d’affiner le classement par retour de pertinence avec la méthode **Re-weighting Type-1**.

Projet du **Master IASD, groupe 4**. Réalisé par **Meriem IBN IBRAHYM**, **Aya EL RHORBA** et **Aya EL MANSOURI**.

## Fonctionnalités

- Créer un compte et accéder à une galerie personnelle.
- Importer, télécharger et supprimer des images, puis les organiser en catégories hiérarchiques.
- Produire de nouvelles images par recadrage, redimensionnement, rotation ou symétrie.
- Parcourir la bibliothèque GHIM-10K, filtrer ses catégories et consulter les descripteurs.
- Chercher des images similaires et affiner les résultats en sélectionnant ceux qui sont pertinents.

## Installation sous Windows

Les commandes suivantes sont à exécuter dans **CMD**. Pour une installation neuve, partir d’un dossier parent où le dépôt n’existe pas encore.

~~~cmd
git clone https://github.com/ElMansouriAya/CBIR_Project.git
cd CBIR_Project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Si le dépôt est déjà installé, ouvrir CMD dans sa racine. Toutes les commandes suivantes s’exécutent depuis ce dossier.

## Préparer GHIM-10K

Télécharger [GHIM-10K sur Kaggle](https://www.kaggle.com/datasets/mcpython/ghim-10k). Le corpus complet contient **10 000 images**, réparties en **20 catégories de 500 images**. Récupérer auprès de l’équipe la matrice et le mapping issus de la même extraction. L’exemple ci-dessous reprend l’organisation locale du lot final.

~~~text
features/
└── ghim_final/
    ├── descriptors_raw.npy
    ├── image_mapping.json
    └── dataset/
        └── GHIM-10K/
            ├── aircraft/
            ├── cars/
            └── ...
~~~

Configurer les deux chemins. DATASET_FOLDER désigne le dossier qui contient directement les catégories.

~~~cmd
set "FEATURES_FOLDER=%CD%\features\ghim_final"
set "DATASET_FOLDER=%FEATURES_FOLDER%\dataset\GHIM-10K"
dir /ad /b "%DATASET_FOLDER%"
.\.venv\Scripts\python.exe scripts\check_features.py --features "%FEATURES_FOLDER%"
~~~

Le mapping doit contenir un identifiant et un indice de ligne pour chaque image. La vérification suivante contrôle les chemins au format **dataset/catégorie/fichier**, conformément au [contrat de données](CONTRAT.md).

~~~cmd
.\.venv\Scripts\python.exe -c "import json,os; from pathlib import Path,PurePosixPath; from collections import Counter; m=json.loads((Path(os.environ['FEATURES_FOLDER'])/'image_mapping.json').read_text(encoding='utf-8')); root=Path(os.environ['DATASET_FOLDER']); counts=Counter(e['category'] for e in m); missing=[e['path'] for e in m if not (root.joinpath(*PurePosixPath(e['path']).parts[1:])).is_file()]; print('Images :',len(m),'Categories :',len(counts),'Chemins manquants :',len(missing)); print(missing[:5]); raise SystemExit(len(m)!=10000 or len(counts)!=20 or set(counts.values())!={500} or bool(missing))"
~~~

## Importer et démarrer VisualVault

Pour une validation isolée, utiliser une nouvelle base SQLite. Les commandes suivantes définissent un chemin distinct dans le dossier temporaire de Windows.

~~~cmd
set "VV_TEST_DB=%TEMP%\visualvault-test-%RANDOM%-%RANDOM%.db"
set "DATABASE_URL=sqlite:///%VV_TEST_DB:\=/%"
.\.venv\Scripts\python.exe -m scripts.import_dataset --help
~~~

Pour un corpus JPEG, lancer l’import standard.

~~~cmd
.\.venv\Scripts\python.exe -m scripts.import_dataset
~~~

Si certains fichiers nommés .jpg contiennent des PNG, utiliser la commande suivante lorsque **--allow-png** figure dans l’aide du script. Si elle est absente et que l’import refuse ces images, utiliser la version de l’importeur qui prend en charge ce lot.

~~~cmd
.\.venv\Scripts\python.exe -m scripts.import_dataset --allow-png
~~~

Lancer Flask après un import réussi. Si le port 5009 est occupé, choisir un port libre.

~~~cmd
.\.venv\Scripts\python.exe -m flask --app backend.app run --port 5009
~~~

Ouvrir [VisualVault](http://127.0.0.1:5009/), créer un compte de test et parcourir la [bibliothèque GHIM](http://127.0.0.1:5009/dataset/). Redémarrer Flask après le remplacement de la matrice ou du mapping.

## Architecture et traitement

| Composant | Rôle |
| --- | --- |
| backend/ | Application et routes Flask, services d’extraction et moteur de recherche. |
| frontend/ | Gabarits Jinja, styles et interactions JavaScript. |
| scripts/ | Import, contrôles des données, démonstration et évaluation. |
| tests/ | Tests unitaires et d’intégration. |
| Stockage local | SQLite pour les comptes et les données applicatives, fichiers image, matrice NumPy et mapping JSON. |

Chaque image GHIM est représentée par un vecteur brut de **172 dimensions**.

| Famille | Dimensions |
| --- | ---: |
| Histogramme HSV (8 × 4 × 4) | 128 |
| Moments de couleur (moyenne, écart-type et asymétrie sur H, S et V) | 9 |
| Couleurs dominantes (K-means, k = 3) | 9 |
| GLCM (quatre propriétés et quatre angles) | 16 |
| Histogramme LBP uniforme (P = 8, R = 1) | 10 |

Le moteur standardise les composantes par z-score et classe les images par distance euclidienne. Lors du retour de pertinence, la requête est recentrée sur la moyenne des vecteurs sélectionnés. Le **Re-weighting Type-1** ajuste les poids en comparant la dispersion de chaque composante dans les résultats affichés et dans les résultats jugés pertinents. Avec une seule image sélectionnée, seule la requête est actualisée. Avec une sélection vide, la requête et les poids sont conservés. Les formats et les modes de recherche sont décrits dans le [contrat](CONTRAT.md).

## Tests et validation

Depuis un CMD ouvert à la racine du projet, lancer la suite avec les réglages utilisés pour la validation sous Windows.

~~~cmd
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
.\.venv\Scripts\python.exe -m pytest -q -rs
~~~

L’option **-rs** affiche les motifs des tests ignorés. Certains tests d’intégration attendent des données générées ou des chemins fixes. Le test du mapping peut notamment attendre **image_storage/dataset** et **features/image_mapping.json**. Vérifier les chemins définis dans **tests/test_image_mapping_dataset_integration.py** avant d’organiser les données de test.

Pour identifier précisément le lot utilisé, relever le commit, la version Python, l’empreinte de la matrice et les premiers octets d’une image de référence. Exécuter ce bloc dans la fenêtre où les chemins GHIM sont configurés.

~~~cmd
git log -1 --oneline
git status --short
.\.venv\Scripts\python.exe --version
certutil -hashfile "%FEATURES_FOLDER%\descriptors_raw.npy" SHA256
.\.venv\Scripts\python.exe -c "import os; from pathlib import Path; p=Path(os.environ['DATASET_FOLDER'])/'cars'/'3_1501.jpg'; print(p.read_bytes()[:8].hex(' ').upper())"
~~~

**FF D8 FF** correspond à un JPEG. **89 50 4E 47** correspond à un PNG. Cette signature documente le format du fichier contrôlé.

### Parcours de bout en bout

1. Contrôler les 10 000 entrées du mapping et les 20 catégories, puis importer le corpus dans une base neuve.
2. Vérifier l’affichage de plusieurs catégories, dont **cars**, ainsi que le filtre et la pagination.
3. Ouvrir la fiche de l’image 1 et comparer les cinq familles aux valeurs de la ligne indiquée par le mapping.
4. Lancer une recherche sur cinq images de catégories différentes. Relever le nombre de résultats, leurs identifiants dans l’ordre, les répétitions éventuelles et l’exclusion de la requête selon les paramètres utilisés.
5. Tester le retour de pertinence avec une sélection vide, unique et complète. Vérifier aussi la réinitialisation et la limite d’itérations. Capturer les itérations 1 et 4 sur la même requête.
6. Vérifier le compteur après la recherche initiale et le feedback, le refus d’une recherche sans connexion et la réponse à une ancienne session de recherche après redémarrage.
7. Si la recherche sur une image personnelle est disponible, téléverser une copie d’une image du corpus et vérifier le rang et la distance de l’original en tenant compte des éventuels ex æquo.

Consigner le résultat de chaque étape et relever les erreurs observées dans les journaux Flask.

### Comparaison avec l’API autonome

Dans un CMD dédié, lancer l’API de démonstration. Vérifier qu’elle charge le même lot que l’interface. Si le script utilise le dossier fixe **features**, préparer une copie de validation contenant la matrice et le mapping à cet emplacement.

~~~cmd
.\.venv\Scripts\python.exe scripts\run_demo_api.py
~~~

Depuis un autre CMD, interroger l’API avec les mêmes options que l’interface. Cet exemple exclut l’image requête.

~~~cmd
curl.exe -X POST "http://127.0.0.1:5001/api/search" -H "Content-Type: application/json" -d "{\"query_image_id\":1,\"scope\":25,\"exclude_query\":true}"
~~~

Comparer les identifiants et leur ordre. Utiliser un serveur de démonstration local pour cette comparaison.

### Évaluation du classement

~~~cmd
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --features "%FEATURES_FOLDER%" --out results
~~~

La **Precision@25** mesure la proportion d’images pertinentes parmi les 25 premiers résultats. Comparer les mesures avant et après retour de pertinence avec les mêmes requêtes. Préciser la règle de pertinence par catégorie, la profondeur du classement et le traitement des doublons de contenu. Les données de **scripts/make_synthetic_data.py** servent aux essais techniques. Les mesures à présenter pour GHIM-10K doivent être calculées sur le corpus réel.

## Références

- S. M. Renuka Devi, *Reweighing Strategies for Relevance Feedback in CBIR*, IJARCCE, vol. 5, n° 11, 2016, section III-C. [DOI 10.17148/IJARCCE.2016.511118](https://doi.org/10.17148/IJARCCE.2016.511118)
- [GHIM-10K sur Kaggle](https://www.kaggle.com/datasets/mcpython/ghim-10k)
