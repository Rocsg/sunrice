# Pipeline Data — Fichiers d'entrée / sortie

## Constantes de chemin

| Constante | Valeur |
|---|---|
| `DEFAULT_INPUT_DIR` (`aerench_config.py`) | `/home/rfernandez/Data/Arize/Hollow_test/1_Intermediate_computed_images/` |
| `DEFAULT_VTK_OUTPUT_DIR` (`water_conductance.py`) | `/home/rfernandez/Data/Arize/Hollow_test/2_Vtk_files/` |

---

## Étape 0 — Pré-traitement aerench (`marvel-aerench-preprocess`)

### Entrées

| Fichier | Description |
|---|---|
| `DEFAULT_INPUT_DIR/Wat_Norm_Cortex.tif` | Image de conductance hydrique normalisée (volume principal) |
| `DEFAULT_INPUT_DIR/Raw.tif` | Volume brut (intensité, pour l'affichage en niveaux de gris) |
| `DEFAULT_INPUT_DIR/Outer.tif` | Masque extérieur 8 bits (255 = exclu de tous les labels) |
| `aerench_output/label*/component_*.vtk` | Composantes connexes par label (issues de aerench) |

### Sorties écrites dans `DEFAULT_INPUT_DIR`

| Fichier | Description |
|---|---|
| `dilatation_scalar_field.tiff` | Champ scalaire de dilatation calculé par aerench |
| `aerench_settings.json` | Paramètres de session aerench (via `DEFAULT_SETTINGS_PATH`) |

---

## Étape 1 — Construction des maillages (`marvel-water-conductance-build-meshes`)

Script : `marvel_view/scripts/water_conductance_build_meshes.py`

### Entrées

| Constante | Fichier | Description |
|---|---|---|
| `DEFAULT_INPUT_PATH` | `1_Intermediate_computed_images/Wat_Norm_Cortex.tif` | Volume de conductance (marching cubes cortex + all) |
| `DEFAULT_RAW_PATH` | `1_Intermediate_computed_images/Raw.tif` | Volume brut (gradient overlay) |
| `DEFAULT_GEODDIST_PATH` | `1_Intermediate_computed_images/Wat_norm-geoddist.tif` | Distance géodésique (flèches arrows-grid) |
| `DEFAULT_MEMBRANES_BG_DIST_PATH` | `1_Intermediate_computed_images/Source_Target_Possible_Paths-dist.tif` | Distance source/cible pour les membranes |
| `DEFAULT_DILATATION_TIFF_PATH` | `1_Intermediate_computed_images/dilatation_scalar_field.tiff` | Champ scalaire de dilatation (produit de l'étape 0) |

### Sorties — `DEFAULT_VTK_OUTPUT_DIR` (`2_Vtk_files/`)

| Constante | Fichier | Description |
|---|---|---|
| `DEFAULT_MESH_CACHE_PATH` | `cortex.vtk` | Maillage iso-surface du cortex |
| `DEFAULT_ALL_MESH_CACHE_PATH` | `all.vtk` | Maillage de tout le volume |
| `DEFAULT_PILLARS_CACHE_PATH` | `pillars_iso.vtk` | Iso-surface des piliers |
| `DEFAULT_MEMBRANES_VTP_CACHE` | `membranes.vtp` | Géométrie des membranes (PolyData) |
| `DEFAULT_MEMBRANES_META_CACHE` | `membranes_meta.json` | Métadonnées des membranes (voxel spacing, etc.) |
| `DEFAULT_MEMBRANES_LABELS_CACHE` | `membranes_labels.tif` | Volume de labels des membranes |
| `DEFAULT_CROWN_TRACKS_CACHE` | `crown_tracks.npz` | Chemins Dijkstra (NumPy) |
| `DEFAULT_CROWN_TRACKS_VTP_CACHE` | `crown_tracks.vtp` | Chemins Dijkstra (vtkPolyLine, lecture directe VTK) |
| `DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE` | `crown_tracks_arrows.vtp` | Flèches pré-calculées pour les tracks |
| `DEFAULT_ARROWS_CACHE_PATH` | `geoddist_arrows.npz` | Flèches geoddist (positions + tangentes) |
| `DEFAULT_OVERLAY_CACHE_PATH` | `cortex_gradient_iso.vtk` | Iso-surface gradient du cortex (overlay) |
| `DEFAULT_DENSITY_BRIDGES_CACHE` | `density_facets_bridges.npy` | Densité facettes bridges |
| `DEFAULT_DENSITY_ALL_CACHE` | `density_facets_all.npy` | Densité facettes all |

---

## Étape 2 — Visualiseur interactif (`marvel-water-conductance`)

Script : `marvel_view/scripts/water_conductance.py`

**Entrées** : tous les fichiers produits à l'étape 1 (lus depuis `DEFAULT_VTK_OUTPUT_DIR`) + les volumes dans `DEFAULT_INPUT_DIR`.

**Sorties** :
- `positions/positions_<timestamp>.json` — positions de caméra sauvegardées via le bouton "Save pos"

---

## Étape 3 — Rendu film (`marvel-water-movie`)

Script : `marvel_view/scripts/water_movie.py`

**Entrées** : mêmes caches que l'étape 2 (importés depuis `water_conductance.py`).

**Sorties** :
- `mp4/<nom>.mp4` — film rendu frame par frame

---

## Note

Les fichiers `dilatation_scalar_field.tiff` et `aerench_settings.json` sont **écrits dans `DEFAULT_INPUT_DIR`**
(`1_Intermediate_computed_images/`), car ils sont des données intermédiaires calculées à partir des entrées
brutes. C'est voulu : ils font partie du même espace de données que les TIFF sources.
