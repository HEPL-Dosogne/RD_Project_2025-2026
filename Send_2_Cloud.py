#!/usr/bin/env python3
import os
import time
from mega import Mega

# Connexion MEGA
mega = Mega()
m = mega.login("guilllaume.dosogne@student.hepl.be", "isil2025")  # identifiants MEGA

CAPTURE_DIR = "/home/gdos/Projet_RD/captures"
SYNC_INTERVAL = 10  # secondes
DEL_AFTER = True  # passe à True si tu veux supprimer après upload

def sync_loop():
    already_sent = set()  # mémorise les fichiers déjà envoyés (session courante)

    while True:
        try:
            files = [
                f for f in os.listdir(CAPTURE_DIR)
                if f.lower().endswith(".jpg")
            ]
        except FileNotFoundError:
            print(f"Dossier introuvable: {CAPTURE_DIR}")
            time.sleep(SYNC_INTERVAL)
            continue

        for f in sorted(files):
            full_path = os.path.join(CAPTURE_DIR, f)

            # ignore si déjà envoyé pendant cette session
            if full_path in already_sent:
                continue

            # vérifie que c'est bien un fichier
            if not os.path.isfile(full_path):
                continue

            try:
                # upload du fichier actuel (pas un nom hardcodé)
                m.upload(full_path)
                already_sent.add(full_path)
                print(f"Synchronisé: {f}")

                if DEL_AFTER:
                    os.remove(full_path)
                    print(f"Supprimé localement: {f}")

            except Exception as e:
                print(f"Erreur upload {f}: {e}")

        # time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    print(f"Synchronisation toutes les {SYNC_INTERVAL} secondes depuis {CAPTURE_DIR}...")
    sync_loop()

