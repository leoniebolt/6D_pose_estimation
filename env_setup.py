import os
import shutil
from pathlib import Path

def copy_rename(src_path, goal_path, new_name):
    try:
        src_path = Path(src_path)
        goal_path = Path(goal_path)

        if not src_path.is_file():
            print(f"Fehler: Quelldatei '{src_path}' existiert nicht oder ist keine Datei.")
            return

        # Zielordner erstellen, falls nicht vorhanden
        goal_path.mkdir(parents=True, exist_ok=True)

        dest_file = goal_path / new_name

        # Datei kopieren
        shutil.copy2(src_path, dest_file)

        print(f"Datei kopiert von '{src_path}' nach '{dest_file}'")
    except Exception as e:
        print(f"Fehler beim Kopieren der Datei '{src_path}' nach '{goal_path}': {e}")

def copy_folder(base_folder, goal_folder):
    try:
        base_folder = Path(base_folder)
        goal_folder = Path(goal_folder)

        if not base_folder.is_dir():
            print(f"Fehler: Quellordner '{base_folder}' existiert nicht oder ist kein Verzeichnis.")
            return

        # Zielordner löschen, falls er existiert
        if goal_folder.exists():
            shutil.rmtree(goal_folder)
            print(f"Existierender Ordner '{goal_folder}' wurde gelöscht.")

        # Ordner kopieren
        shutil.copytree(base_folder, goal_folder)

        print(f"Ordnerinhalt von '{base_folder}' wurde nach '{goal_folder}' kopiert.")
    except Exception as e:
        print(f"Fehler beim Kopieren des Ordners '{base_folder}' nach '{goal_folder}': {e}")

if __name__ == "__main__":
    # Dateien kopieren mit neuem Namen
    copy_rename("utility_scripts/environment_full.yaml", "megapose6d/conda", "environment_full.yaml")
    copy_rename("utility_scripts/run_inference_on_example.py", "megapose6d/src/megapose/scripts", "run_inference_on_example.py")

    # Ordner komplett kopieren (ersetzen)
    copy_folder("data/morobot", "megapose6d/local_data/examples/morobot")
