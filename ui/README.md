# Dokumentenbasiertes QA-System UI

Dieses Verzeichnis enthält die Benutzeroberfläche für das dokumentenbasierte Frage-Antwort-System. Die UI ermöglicht das Hochladen und Verwalten von Dokumenten, das Training von KI-Modellen mit diesen Dokumenten, das Stellen von Fragen und das Durchführen von Churn-Prediction.

## Struktur

```
ui/
├── public/               # Statische Dateien
├── src/                  # Quellcode
│   ├── components/       # React-Komponenten
│   │   ├── common/       # Wiederverwendbare Komponenten
│   │   ├── dashboard/    # Dashboard-Komponenten
│   │   ├── documents/    # Dokumentenverwaltung
│   │   ├── training/     # Training-Interface
│   │   ├── qa/           # QA-Interface
│   │   ├── models/       # Modellverwaltung
│   │   └── churn/        # Churn-Prediction
│   ├── hooks/            # Custom React Hooks
│   ├── services/         # API-Dienste
│   ├── store/            # State Management
│   ├── utils/            # Hilfsfunktionen
│   ├── pages/            # Hauptseiten
│   └── styles/           # CSS-Styles
└── package.json          # Projektabhängigkeiten
```

## Installation

```bash
# Navigiere zum UI-Verzeichnis
cd ui

# Installiere Abhängigkeiten
npm install

# Starte die Entwicklungsumgebung
npm start
```

## Integration mit dem Backend

Die UI kommuniziert mit dem Python-Backend über eine REST-API und WebSockets für Echtzeit-Updates während des Trainings. Die API-Endpunkte sind in der Datei `services/api.ts` definiert.

## Entwicklung

Um die UI weiterzuentwickeln:

1. Starten Sie das Backend mit `python src/api.py`
2. Starten Sie die UI mit `npm start` im `ui`-Verzeichnis
3. Die Anwendung ist unter http://localhost:3000 verfügbar

## Erstellen für die Produktion

```bash
npm run build
```

Die erstellen Dateien befinden sich im `build`-Verzeichnis und können auf einem beliebigen Webserver bereitgestellt werden.
