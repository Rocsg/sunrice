# Gitflow — sunrice

Dépôt hébergé sur GitHub/GitLab.  
Code sur deux machines ; données **jamais** dans git (trop grosses, synchro via rsync).

---

## Chemins de référence

| | Ordi local | Serveur |
|---|---|---|
| **Code** | `/home/rfernandez/Dev/Python/sunrice/` | `/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/sunrice/` |
| **Données entrées** | `/home/rfernandez/Data/Arize/Hollow_test/1_Intermediate_computed_images/` | `/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/Data/1_Intermediate_computed_images/` |
| **Sorties VTK** | `/home/rfernandez/Data/Arize/Hollow_test/2_Vtk_files/` | `/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/Data/2_Vtk_files/` |
| **Venv** | `/home/rfernandez/Dev/venvs/vtk_venv/` | `/mnt/.../Dev/vtk_venv/` |

Les chemins sont pilotés par deux variables d'environnement (voir section *Config*).

---

## Initialisation (une seule fois)

### Sur l'ordi local

```bash
cd /home/rfernandez/Dev/Python/sunrice
git init
git add .
git commit -m "init"
git remote add origin git@github.com:TOI/sunrice.git
git push -u origin main
```

### Sur le serveur

```bash
SERVER_CODE=/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev
git clone git@github.com:TOI/sunrice.git $SERVER_CODE/sunrice
cd $SERVER_CODE/sunrice

python3 -m venv $SERVER_CODE/vtk_venv
source $SERVER_CODE/vtk_venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Copier les données source une seule fois (TIFFs) :**
```bash
rsync -av --progress \
    /home/rfernandez/Data/Arize/Hollow_test/1_Intermediate_computed_images/ \
    serveur:$SERVER_CODE/Data/1_Intermediate_computed_images/
```

---

## Config — variables d'environnement

À ajouter dans le `~/.bashrc` **du serveur** :

```bash
export MARVEL_INPUT_DIR="/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/Data/1_Intermediate_computed_images"
export MARVEL_VTK_OUTPUT_DIR="/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/Data/2_Vtk_files"
```

Sur l'ordi local ces variables ne sont **pas** nécessaires (les defaults dans le code pointent déjà vers les bons chemins locaux).

Pour un job non-interactif (cron, sbatch…) passer les vars directement :
```bash
MARVEL_INPUT_DIR="..." MARVEL_VTK_OUTPUT_DIR="..." marvel-water-conductance-build-meshes ...
```

---

## Workflow quotidien

```
┌──────────────────────────────────────────────────────────────────┐
│  LOCAL                          SERVEUR                          │
├──────────────────────────────────────────────────────────────────┤
│  1. Modifier le code                                             │
│  2. git add -p && git commit -m "..."                            │
│  3. git push                                                     │
│                                 4. git pull                      │
│                                 5. source vtk_venv/bin/activate  │
│                                 6. marvel-water-conductance-     │
│                                    build-meshes                  │
│                                    --skip-mesh --skip-all-mesh   │
│                                    --skip-pillars --skip-overlay │
│                                    --skip-arrows --skip-tracks   │
│                                    --skip-dilatation             │
│                                    --skip-density                │
│                                    (→ recalcule les membranes)   │
│  7. rsync résultats ←────────────────────────────────────────── │
│  8. marvel-water-conductance (viewer local)                      │
└──────────────────────────────────────────────────────────────────┘
```

### Étape 7 — rsync des résultats vers l'ordi local

```bash
SERVER_VTK="/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/Data/2_Vtk_files"
LOCAL_VTK="/home/rfernandez/Data/Arize/Hollow_test/2_Vtk_files"

rsync -av --progress serveur:"$SERVER_VTK/membranes.vtp" \
                     serveur:"$SERVER_VTK/membranes_meta.json" \
                     serveur:"$SERVER_VTK/membranes_labels.tif" \
                     "$LOCAL_VTK/"
```

Pour tout rapatrier (premiers builds) :
```bash
rsync -av --progress serveur:"$SERVER_VTK/" "$LOCAL_VTK/"
```

---

## Cas particulier — cache des iso-surfaces

Le calcul des iso-surfaces (marching cubes par niveau) est l'étape la plus longue.
Elle est **mise en cache** dans `$MARVEL_VTK_OUTPUT_DIR/membranes_iso_cache/`.

Si ce cache existe déjà, `build-meshes` saute le marching cubes et repart
directement de là pour recalculer les colonnes / délais / seuil de fond.

**Forcer le recalcul complet du cache :**
```bash
marvel-water-conductance-build-meshes \
    --skip-mesh --skip-all-mesh --skip-pillars --skip-overlay \
    --skip-arrows --skip-tracks --skip-dilatation --skip-density \
    --rebuild-iso-cache
```

**Ne recalculer que les colonnes/délais (cache déjà présent) :**
```bash
marvel-water-conductance-build-meshes \
    --skip-mesh --skip-all-mesh --skip-pillars --skip-overlay \
    --skip-arrows --skip-tracks --skip-dilatation --skip-density
    # (pas de --rebuild-iso-cache → repart du cache)
```

---

## Branches recommandées

| Branche | Usage |
|---|---|
| `main` | Code stable, tourné et validé |
| `dev` | Expérimentations en cours |

```bash
# Créer une branche de dev
git checkout -b dev

# Merger dans main quand c'est bon
git checkout main && git merge dev
```
