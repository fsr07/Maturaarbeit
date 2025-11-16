# Käsekästchen KI Maturaarbeit Projekt

Dieses Projekt implementiert eine Künstliche Intelligenz für das Käsekästchen-Spiel unter Verwendung der Gymnasium-Umgebung und Stable-Baselines3. Das Ziel war, verschiedene Belohnungsfunktionen zu untersuchen und einen Agenten zu trainieren, der effektiv gegen einen zufälligen oder strategischen Spieler spielt.

## Installation

Um dieses Programm nutzen zu können, benötigen Sie Git. Nach der Installation können Sie das Repository mit folgendem Befehl klonen:

```PowerShell
git clone https://github.com/fsr07/Maturaarbeit.git
```

Es ist empfehlenswert, das Repository an einen Ort zu klonen, an dem der gesamte Pfad nicht allzu viele Zeichen enthält, da es sonst aufgrund der maximalen Pfadlänge von Windows zu Problemen beim Ausführen des Programms kommen kann.

Um das Projekt auszuführen, stellen Sie sicher, dass Sie Python (3.8-3.11; Achtung: Python 3.12 und neuere Versionen werden NICHT unterstützt) installiert haben. Alle benötigten Bibliotheken sind in der Datei requirements.txt aufgeführt und können mit einem einzigen Befehl installiert werden:

```PowerShell
cd Maturaarbeit
pip install -r requirements.txt
```

## Verwendung von `Steuerung.py`

Das `Steuerung.py`-Skript ist das Hauptprogramm zur Interaktion mit den Käsekästchen-KI-Umgebungen. Wird es gestartet, bietet folgende Optionen:

1.  **Neues Model trainieren:** Startet ein neues Training für einen DQN-Agenten. Sie können die Belohnungsfunktion (B1, B2, B3) und optional Hyperparameter wie Lernrate, Diskontierungsfaktor und Erkundungsanteil anpassen.
2.  **Vorhandenes Model weitertrainieren:** Lädt ein bestehendes Modell und setzt das Training für eine angegebene Anzahl von Zeitschitten fort.
3.  **Model evaluieren lassen:** Lädt ein trainiertes Modell und bewertet dessen Leistung über eine bestimmte Anzahl von Spielen, wobei die durchschnittliche Belohnung, Standardabweichung und Gewinnrate ausgegeben werden.
4.  **Selber Gegen ein trainiertes Model spielen:** Ermöglicht es Ihnen, interaktiv gegen ein geladenes KI-Modell zu spielen.

Starten Sie das Skript und folgen Sie den Anweisungen in der Konsole:

```PowerShell
# Zuerst in den Ordner mit dem Code wechseln
cd Code
# Dann den Code ausführen
python Steuerung.py
```

## Projektstruktur

Das Projekt ist folgendermassen aufgebaut:

```
.
├── README.md                              # Diese Datei
├── requirements.txt                       # Python-Abhängigkeiten
├── LICENSE.txt                            # MIT Lizenz
├── .gitignore                             # Git-Konfiguration
│
└── Code/                                  # Implementierung
    ├── Steuerung.py                       # Hauptprogramm
    ├── Belohnungsfunktion_1.py            # Umgebung mit reaktiver Bestrafung
    ├── Belohnungsfunktion_2.py            # Umgebung mit vorausschauender Bestrafung
    ├── Belohnungsfunktion_3.py            # Umgebung mit kombinierter Bestrafung
    ├── Evaluierungsumgebung.py            # Evaluierungsumgebung mit Evaluierungsgegner
    ├── [Modellname].zip                   # Trainierte Modelle
    │
    └── logs/
        ├── tmp/                           # Trainings-Logs
        │   └── [Modellname]/
        │       ├── events.out.tfevents.*  # TensorBoard Dateien
        │       └── progress.csv           # Trainingsdaten
        │
        └── Datenauswertung/               # Ausgewertete Daten und Diagramme
            └── *.xlsx                     # Excel-Auswertungen
```

## Übersicht der Modelle

Die folgenden Modelle wurden im Rahmen der Maturaarbeit trainiert und analysiert:

| Datei (.zip) | Bezeichnungen | Experiment | Belohnungsfunktion (Umgebung) | Lernrate | Diskontierungsfaktor | Explorationsphasenanteil |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | (B1), (T2), Referenzmodell | 1 und 2 | B1 | 0,0001 | 0,99 | 0,1 |
| B2 | (B2) | 1 | B2 | 0,0001 | 0,99 | 0,1 |
| B3 | (B3) | 1 | B3 | 0,0001 | 0,99 | 0,1 |
| T1a | (T1a) | 2 | B1 | 0,00005 | 0,99 | 0,1 |
| T3a | (T3a) | 2 | B1 | 0,001 | 0,99 | 0,1 |
| T1b | (T1b) | 2 | B1 | 0,0001 | 0,9 | 0,1 |
| T3b | (T3b) | 2 | B1 | 0,0001 | 0,999 | 0,1 |
| T1c | (T1c) | 2 | B1 | 0,0001 | 0,99 | 0,05 |
| T3c | (T3c) | 2 | B1 | 0,0001 | 0,99 | 0,25 |

## TensorBoard Statistiken

Während des Trainings werden Metriken und Logs in `"Code/logs/tmp/"` gespeichert. Sie können diese Statistiken mit TensorBoard visualisieren. Öffnen Sie ein Terminal im Projektstammverzeichnis und führen Sie den folgenden Befehl aus:

```PowerShell
tensorboard --logdir "Code/logs/tmp/" --reload_interval 30
```

Anschliessend können Sie TensorBoard in Ihrem Webbrowser unter der angezeigten Adresse (normalerweise `http://localhost:6006/`) aufrufen.