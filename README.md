# marvel_view

Visualisation 3-D de segmentations volumétriques (TIFF multi-pages) à partir de
maillages de surface générés par marching cubes.  
Stack : **tifffile · scikit-image · scipy · VTK · vedo**.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia 
---


export MARVEL_DATA_DIR="/home/rfernandez/Data/Arize/Hollow_test"

##  Quick-start — film VR360 (Meta Quest 3 Pro / YouTube VR)

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate

# 0. (Une fois) Construire le maillage et le cache des flèches (gradient
#    de la carte de distance géodésique) pour des launches instantanés.
marvel-water-conductance-build-meshes     # → ./marvel_output/water_conductance/
                                          #     cortex.vtk + geoddist_arrows.npz

# 1. Ouvrir le viewer (chargera les caches si présents), naviguer,
#    cliquer plusieurs fois sur "Save pos"
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia marvel-water-conductance
# → écrit ./positions/positions_<stamp>.json

# 2. Rendre le film VR360 stéréo SBS, taggé Spherical-Video-V2
#    Toutes les options haute qualité sont DÉJÀ les valeurs par défaut :
#      résolution 8192×2048 SBS, h265 (HEVC) CRF 20 preset slow,
#      cubemap 2048 px/face, IPD 5 %, metadata VR injectée.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia marvel-water-movie --vr vr360   # → ./mp4/positions_<stamp>_vr360.mp4
```

Debug : burner l'index de keyframe (flottant, relatif aux clicks « Save
pos ») en bas de chaque frame d'un rendu **flat** — pratique pour pointer
un instant précis au pair (« déclenche tel comportement à l'index 4.2 ») :

```bash
marvel-water-movie --debugdisplayindexframes        # mono uniquement
```

Pour aller plus vite pendant les essais (4 premières secondes seulement) :

```bash
marvel-water-movie --vr vr360 --stresstest
```

Pour pousser la qualité encore plus loin (très lent, ~visuellement lossless) :

```bash
marvel-water-movie --vr vr360 \
    --vr-cube-resolution 4096 \   # cubemap 4K par face
    --crf 16 --preset veryslow
```

**Test local sur Meta Quest 3 / Pro** — pas besoin de YouTube pour valider :

1. Brancher le casque en USB (autoriser l'accès aux fichiers depuis le casque).
2. Copier `./mp4/positions_<stamp>_vr360.mp4` dans `Quest 3/Internal shared
   storage/Movies/` (ou `Oculus/Movies/`).
3. Dans le casque : **Files → Movies** (ou installer **Meta Quest TV** /
   **Skybox VR** / **DeoVR**) → le film s'ouvre en 360° stéréo automatiquement
   grâce aux metadata.

**Upload YouTube** : YouTube Studio → Upload → le tag spherical-v2 est détecté
automatiquement, la vidéo apparaît avec le badge **« 360° »** + **3D**.
Aucun réglage manuel à faire dans Studio.

**VR180** : `marvel-water-movie --vr vr180` (4096×2048 SBS, hémisphère avant
uniquement). Compatible Quest mais YouTube VR180 utilise un format de tag
légèrement différent — le SBS standard tagué equirect+crop fonctionne bien
sur Quest mais YouTube peut le classer comme « 360° » plutôt que « VR180 ».
Pour de la stéréo immersive sans souci de classement, **préférer `vr360`**.

---

## Aide-mémoire — commandes disponibles

Activer d'abord le venv :

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
```

| Commande                    | Rôle                                                                       |
|-----------------------------|----------------------------------------------------------------------------|
| `marvel-preprocess`         | Pipeline MARVEL générique → `marvel_output/`                               |
| `marvel-view`               | Viewer 3-D pour la scène MARVEL                                            |
| `marvel-export-html`        | Export HTML (vtk.js) de la scène MARVEL                                    |
| `marvel-roots-preprocess`   | Pipeline « roots » → `roots_output/`                                       |
| `marvel-roots-view`         | Viewer 3-D pour les racines                                                |
| `marvel-aerench-preprocess` | Pipeline « aerench » (gaz / méats / eau) → `aerench_output/`               |
| `marvel-aerench-view`       | Viewer 3-D pour la scène aerench                                           |
| `marvel-water-conductance-build-meshes` | Pré-construit `cortex.vtk` + `geoddist_arrows.npz` (caches mesh & vecteurs gradient) |
| `marvel-water-conductance`  | Viewer Wat_Norm_Cortex (mesh, 4 boutons : Shading / Move / Save pos / View Mesh↔Arrows) |
| `marvel-water-movie`        | Rendu MP4 (mono ou VR180/360) depuis un `positions_*.json`                |

Options utiles communes : `--resume`, `--workers N`, `--labels 1 2 3`,
`--max-components N`, `-v` (verbose).

### Aerench — preprocessing (entrées 32-bit + masque Outer)

Le pipeline lit désormais **trois TIFFs 32-bit** dans le même dossier
(« norm » : positif = dans la classe, négatif = hors classe) ainsi qu'un
masque **`Outer.tif`** 8-bit (`255` = extérieur du tube, à exclure).

```
Aerench_norm.tif   → label 1 (gas)
Meat_norm.tif      → label 2 (meats)
Wat_norm.tif       → label 3 (water)
Outer.tif          → exclusion (voxels à 255)
```

Marching cubes tourne directement à `level = 0` sur le champ float, ce
qui donne une iso-surface sub-voxel sans flou gaussien.

```bash
# Tout relancer depuis zéro (lit DEFAULT_INPUT_DIR + Outer.tif) :
marvel-aerench-preprocess

# Reprise sans refaire les labels déjà marqués _done :
marvel-aerench-preprocess --resume

# Ne traiter que le gaz (label 1) :
rm -rf aerench_output/label1_gas
marvel-aerench-preprocess --labels 1

# Pointer vers un autre dossier d'entrée / autre Outer.tif :
marvel-aerench-preprocess \
    --input-dir "/media/.../Unrolled/Extract2" \
    --outer     "/media/.../Unrolled/Extract2/Outer.tif"

# Désactiver l'exclusion par Outer.tif (debug) :
marvel-aerench-preprocess --no-outer

# Changer l'iso-level (par défaut 0) :
marvel-aerench-preprocess --level 0.1
```

Chemins par défaut configurés dans `marvel_view/aerench_config.py`
(`DEFAULT_INPUT_DIR`, `DEFAULT_OUTER_PATH`, `LABEL_CONFIG[*]["input_file"]`).

---

## 1. Créer et activer le venv

```bash
python3 -m venv /home/rfernandez/Dev/venvs/vtk_venv
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
```

## 2. Installer les dépendances et le package

```bash
cd /home/rfernandez/Dev/Python/sunrice
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Preprocessing — générer les maillages VTK

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
marvel-preprocess
```

Par défaut, lit l'image configurée dans `marvel_view/config.py` et écrit les
maillages dans `./marvel_output/`.

Options utiles :

| Option | Description | Défaut |
|--------|-------------|--------|
| `-i PATH` | Chemin vers le TIFF | voir `config.py` |
| `-o DIR` | Dossier de sortie VTK | `./marvel_output` |
| `--spacing Z Y X` | Taille de voxel physique | `1 1 1` |
| `--smooth-iter N` | Itérations de lissage Laplacien (≈20 % avant la décimation, reste après) | `80` |
| `--decimate F` | Fraction de faces à conserver — utilisée seulement si `--face-budget` ≤ 0 | `1.0` |
| `--face-budget N` | Budget global de triangles réparti par label (∝ voxels) puis par composante (∝ voxels^(2/3)) | `10_000_000` |
| `--step-size N` | Pas du marching cubes. `0` = auto (réglé à partir du budget) ; `1` = pleine résolution ; `>1` = plus grossier/rapide | `0` |
| `--workers N` | Processus parallèles pour le meshing des composantes `all_cc` (0 = cpu_count, 1 = série) | `0` |
| `--labels 1 3 5` | Ne traiter que certains labels | tous |
| `--max-components N` | Limiter aux *N* plus grosses composantes (`all_cc`) | — |
| `--resume` | Skip labels/composantes déjà traités (markers `_done`, `_ckpt_mask.npy`, VTK existants) | off |
| `-v` | Logging DEBUG | — |

Exemple avec paramètres personnalisés :

```bash
marvel-preprocess -i "/chemin/vers/image.tiff" -o ./out --spacing 2.5 1.0 1.0 --smooth-iter 120 --labels 1 3 5
```

## 4. Visualisation interactive

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
marvel-view
```

Options utiles :

| Option | Description | Défaut |
|--------|-------------|--------|
| `-d DIR` | Dossier contenant les sous-dossiers VTK | `./marvel_output` |
| `--labels 1 3 5` | N'afficher que certains labels | tous |
| `--bg COLOR` | Couleur de fond (`black`, `white`, `#1a1a2e` …) | `black` |
| `-s FILE.png` | Sauvegarder un screenshot PNG et quitter | — |
| `-v` | Logging DEBUG | — |

Exemple — n'afficher que l'iode et les membranes :

```bash
marvel-view --labels 3 5
```

## 5. Personnalisation

Toute la configuration (chemin image, couleurs, opacités, stratégies par label,
taille voxel, lissage) est centralisée dans **`marvel_view/config.py`**.

### Stratégies de traitement par label

| Valeur | Comportement |
|--------|-------------|
| `full_mask` | Maille le label entier en une seule surface |
| `interior_cc` | Composantes connexes, conserve uniquement celles qui **ne touchent pas** les bords du volume (cavités fermées) |
| `all_cc` | Un maillage indépendant par composante connexe |

### Labels configurés

| Label | Nom | Stratégie | Couleur | Opacité |
|-------|-----|-----------|---------|---------|
| 1 | cavités background | `interior_cc` | bleu sombre | 0.35 |
| 2 | tissus aqueux | `full_mask` + enveloppe floutée | blanc/gris | 0.06 |
| 3 | iode | `all_cc` | cyan-vert | 0.75 |
| 4 | tuteur | `full_mask` fantomatique | gris éteint | 0.08 |
| 5 | membranes | `all_cc` | orange | 0.55 |

## 6. Exécution rapide sans `pip install -e .`

Les scripts peuvent aussi être lancés directement :

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
python marvel_view/scripts/preprocess_all.py
python marvel_view/scripts/visualize_scene.py
```

---

## 7. Extension « roots » (jeu de données *small_roots-2*)

Pipeline parallèle, indépendant du pipeline SunRice principal. Il utilise sa
propre configuration (`marvel_view/roots_config.py`), ses propres scripts, et
son propre dossier de sortie (`./roots_output`).

Différences clés :

* **Sceau des bordures** : juste après le chargement de la segmentation, les
  voxels des 6 faces du volume sont peints en label `2` (= fond). Les autres
  labels sont alors strictement intérieurs et leurs surfaces se ferment
  proprement aux bords (pas de trous, pas de singularités).
* **Cibles de triangles par label** :

  | Label | Nom | Couleur | Opacité | `face_budget` |
  |-------|-----|---------|---------|---------------|
  | 1 | iode | turquoise clair | 1.0 | 5 M |
  | 3 | lamelles | orange | 0.55 | 10 M |
  | 4 | stèle | marron | 0.20 | 2 M |
  | 5 | milieu environnant | gris clair | 1.0 | 20 M |

* **Viewer enrichi** : un bouton `Mode` cycle entre :
  * `mesh`   → maillages seuls,
  * `volume` → rendu volumique du TIFF source 8 bits avec deux sliders de
    seuil (`thr_low` / `thr_high`) + un slider d'opacité,
  * `split`  → fenêtre splittée en deux, maillages à gauche / volume à
    droite, **caméra synchronisée**.
* **Éclairage type lampe-torche** : une seule lumière attachée à la caméra,
  légèrement décalée, pour le feeling « tunnels explorés à la lampe ».
* **Sauvegarde / Reset** : boutons `Save` (écrit un JSON à côté des données,
  par défaut `small_roots-2_marvel_settings.json`) et `Reset` (revient à
  l'état initial). Le fichier est aussi relu automatiquement au démarrage.

### Préparation des maillages

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
cd /home/rfernandez/Dev/Python/sunrice
pip install -e .            # une seule fois, pour exposer les nouveaux scripts
marvel-roots-preprocess     # lit le TIFF de segmentation et écrit ./roots_output/
```

Options principales (mêmes sémantiques que `marvel-preprocess`) :

| Option | Description | Défaut |
|--------|-------------|--------|
| `-i PATH` | TIFF de segmentation | voir `roots_config.py` |
| `-o DIR` | Dossier de sortie VTK | `./roots_output` |
| `--spacing Z Y X` | Taille de voxel physique | `1 1 1` |
| `--smooth-iter N` | Itérations de lissage Laplacien | `80` |
| `--workers N` | Processus parallèles `all_cc` | cpu_count |
| `--labels 1 3 5` | Restreindre à certains labels | tous |
| `--max-components N` | Limiter aux N plus grosses CC | — |
| `--no-seal-borders` | Désactive le scellage des bordures | off |
| `--resume` | Reprendre où on en était | off |

### Visualisation interactive

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
marvel-roots-view
```

Options :

| Option | Description | Défaut |
|--------|-------------|--------|
| `-d DIR` | Dossier des sous-dossiers VTK | `./roots_output` |
| `-s PATH` | TIFF source 8 bits (rendu volumique) | voir `roots_config.py` |
| `--settings PATH` | Fichier JSON de réglages | à côté des données |

Boutons en haut à droite : `Mode` (mesh/volume/split), `Save`, `Reset`.
Boutons en haut à gauche : visibilité par label.
Sliders en bas : `opacity / ambient / diffuse / specular` du label
sélectionné par le bouton `Tune`. En modes `volume` et `split`, trois
sliders supplémentaires (`vol.thr_low`, `vol.thr_high`, `vol.opacity`)
contrôlent le rendu volumique.

### Lancement direct sans `pip install -e .`

```bash
source /home/rfernandez/Dev/venvs/vtk_venv/bin/activate
python marvel_view/scripts/roots_preprocess.py
python marvel_view/scripts/roots_visualize.py
```

---

## Structure du package

```
marvel_view/
├── config.py                    # couleurs, opacités, stratégies par label
├── preprocessing/
│   ├── loader.py                # chargement TIFF
│   ├── connected_components.py  # analyse CC + filtre bordures
│   └── meshing.py               # marching cubes → lissage → décimation
├── visualization/
│   ├── actors.py                # style (couleur, opacité, éclairage Phong)
│   └── scene.py                 # assemblage et rendu vedo
└── scripts/
    ├── preprocess_all.py        # CLI preprocessing
    └── visualize_scene.py       # CLI visualisation
```
