"""
This script provides an automated synchronization service that uploads image files
from a local directory to a MEGA cloud storage account.

The program periodically scans the capture directory for new .jpg files, uploads
each file to MEGA, and optionally deletes the local copy after a successful transfer.
A session‑level cache prevents re‑uploading files that were already synchronized
during the current execution.
"""


#!/usr/bin/env python3
import os
import time
from mega import Mega

# Connection to MEGA
mega = Mega()
m = mega.login("email address", "password")  # ID of the MEGA's account used

CAPTURE_DIR = "/home/gdos/Projet_RD/captures"
SYNC_INTERVAL = 10    # seconds
DEL_AFTER = True      # True to delete captures from the jetson after upload

def sync_loop():
    already_sent = set()  # remembers files already uploaded (current session)

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

            # ignore already sent
            if full_path in already_sent:
                continue

            if not os.path.isfile(full_path):
                continue

            try:
                # upload
                m.upload(full_path)
                already_sent.add(full_path)
                print(f"Synchronisé: {f}")

                if DEL_AFTER:
                    os.remove(full_path)
                    print(f"Supprimé localement: {f}")

            except Exception as e:
                print(f"Erreur upload {f}: {e}")

        time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    print(f"Synchronisation toutes les {SYNC_INTERVAL} secondes depuis {CAPTURE_DIR}...")
    sync_loop()

