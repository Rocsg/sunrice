# How to install on your Linux machine

## Prerequisites

- Git
- Python 3.10+
- A Linux machine with a graphical environment (GNOME recommended)

---

## 1. Clone the repository

```bash
mkdir ~/sunrice
git clone https://github.com/Rocsg/sunrice.git ~/sunrice
cd ~/sunrice
```

---

## 2. Create and populate the virtual environment

```bash
python3 -m venv ~/vtk_venv
source ~/vtk_venv/bin/activate
pip install -e .
```

This installs all entry points (`marvel-aerenquest`, `marvel-water-conductance`, etc.) into the venv.

---

## 3. Set up environment variables

Add these two lines to your `~/.bashrc` (adapt paths to your machine):

```bash
export MARVEL_VENV="$HOME/vtk_venv"
export MARVEL_DATA_DIR="$HOME/Data/marvel"   # path to the map data folder
```

Then reload:
```bash
source ~/.bashrc
```

> **Map data** — The source data files are not included in the repository.  
> Contact **Romain Fernandez** to obtain them.

---

## 4. Create the launcher script

Create a file `~/launch_aerenquest.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MARVEL_VENV:?Set MARVEL_VENV in ~/.bashrc}"
: "${MARVEL_DATA_DIR:?Set MARVEL_DATA_DIR in ~/.bashrc}"

source "${MARVEL_VENV}/bin/activate"
exec marvel-aerenquest "$@"
```

Make it executable:
```bash
chmod +x ~/launch_aerenquest.sh
```

---

## 5. Install the desktop launcher (GNOME)

Create `/usr/share/applications/launcher_aerenquest.desktop`:

```ini
[Desktop Entry]
Version=1.0
Type=Application
Name=Aerenquest
Comment=3D exploration of rice root anatomy
Exec=/home/YOUR_USER/launch_aerenquest.sh
Icon=/home/YOUR_USER/sunrice/images/icon.png
Terminal=false
Categories=Science;
StartupNotify=true
```

Replace `YOUR_USER` with your username, then register it:

```bash
sudo update-desktop-database /usr/share/applications/
```

To add it to the GNOME dock: press `Super`, search "Aerenquest", right-click the icon → **Add to Favorites**.

---

## 6. Score submission key

The leaderboard/score submission feature requires a personal API key.  
Contact **Romain Fernandez** to obtain yours.

---

## Keeping the code up to date

```bash
cd ~/sunrice
git pull
pip install -e .
```
