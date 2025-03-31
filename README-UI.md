# Web-UI für das Dokumentenbasierte Frage-Antwort-System

Diese Web-Benutzeroberfläche bietet eine intuitive und benutzerfreundliche Möglichkeit, mit dem dokumentenbasierten Frage-Antwort-System zu interagieren, Dokumente zu verwalten und Churn-Prediction-Analysen durchzuführen.

## Funktionen

Die Web-UI bietet folgende Hauptfunktionen:

1. **Frage & Antwort**:
   - Stellen Sie Fragen zu Churn-Prediction und erhalten Sie Antworten mit Quellenangaben
   - Sehen Sie, welche Dokumente für die Antwort verwendet wurden
   - Anpassen Sie die Anzahl der zu durchsuchenden Dokumente und andere Parameter

2. **Churn-Prediction**:
   - Analysieren Sie Kundendaten, um Abwanderungsrisiken zu identifizieren
   - Erhalten Sie dokumentenbasierte Handlungsempfehlungen
   - Sehen Sie detaillierte Informationen zu Risikofaktoren und Maßnahmen

3. **Dokumentenverwaltung**:
   - Laden Sie neue Dokumente hoch (TXT, PDF, DOCX, MD, HTML)
   - Sehen Sie vorhandene Dokumente und ihre Verarbeitung
   - Passen Sie die Dokumentverarbeitung an (Chunk-Größe, Überlappung usw.)

## Installation und Start

### Voraussetzungen

- Python 3.8 oder höher
- Alle benötigten Pakete aus `requirements-ui.txt`

### Installation

1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Installieren Sie die Abhängigkeiten:
   ```bash
   pip install -r requirements-ui.txt
   ```

3. Starten Sie die Web-UI:
   ```bash
   python app.py
   ```

4. Öffnen Sie einen Browser und navigieren Sie zu:
   ```
   http://localhost:5000
   ```

## Nutzung der Web-UI

### Erste Schritte

1. **Systeminitialisierung**: Beim ersten Start müssen Sie das System initialisieren, indem Sie auf der Startseite auf "System initialisieren" klicken. Dies lädt die Modelle und verarbeitet die Beispieldokumente.

2. **Dokumente hinzufügen**: Gehen Sie zur Dokumentenverwaltung, um weitere Dokumente hochzuladen. Unterstützte Formate sind TXT, PDF, DOCX, MD und HTML.

### Frage & Antwort

1. Geben Sie Ihre Frage in das Textfeld ein und klicken Sie auf "Frage stellen".
2. Das System sucht in den Dokumenten nach relevanten Informationen und generiert eine Antwort.
3. Die Antwort wird mit Quellenangaben angezeigt, die zeigen, aus welchen Dokumenten die Informationen stammen.
4. Verwenden Sie die erweiterten Optionen, um anzupassen, wie das System arbeitet:
   - Aktivieren/Deaktivieren der Antwortgenerierung
   - Anpassen der Anzahl der zu durchsuchenden Dokumente

### Churn-Prediction

1. Wählen Sie zwischen Beispieldaten oder dem Hochladen eigener Kundendaten (CSV).
2. Klicken Sie auf "Churn-Vorhersage starten", um die Analyse durchzuführen.
3. Die Ergebnisse zeigen eine Zusammenfassung der Risikokategorien und eine detaillierte Tabelle mit Kunden.
4. Klicken Sie auf "Details", um mehr über einen bestimmten Kunden zu erfahren, einschließlich:
   - Risikofaktoren
   - Empfohlene Maßnahmen
   - Relevante Dokumentreferenzen

### Dokumentenverwaltung

1. Sehen Sie eine Liste aller verfügbaren Dokumente.
2. Laden Sie neue Dokumente hoch über das Formular.
3. Klicken Sie auf "Anzeigen", um Dokumentdetails zu sehen:
   - Abschnitte und Chunks
   - Extrahierte Schlüsselwörter
   - Metadaten
4. Passen Sie die Dokumentverarbeitung an:
   - Ändern Sie die Chunk-Größe und Überlappung
   - Klicken Sie auf "Dokumente neu verarbeiten", um Änderungen anzuwenden

## Anpassung der Web-UI

### Anpassung des Erscheinungsbilds

Die UI verwendet Bootstrap 5 und kann leicht angepasst werden:

- Ändern Sie das Design in `static/css/style.css`
- Verwenden Sie andere Bootstrap-Themes oder -Komponenten

### Erweiterung der Funktionalität

Die modulare Struktur ermöglicht einfache Erweiterungen:

1. Fügen Sie neue Routen in `app.py` hinzu
2. Erstellen Sie neue Templates in `templates/`
3. Fügen Sie JavaScript-Funktionalität in `static/js/` hinzu

## Tipps und Troubleshooting

### Leistungsoptimierung

- Für größere Dokumentensammlungen verwenden Sie einen leistungsstärkeren Server
- Aktivieren Sie die Vorverarbeitung von Dokumenten im Voraus, um die Startzeit zu verkürzen

### Bekannte Probleme

- Bei sehr großen Dokumenten kann die Verarbeitung länger dauern
- Einige PDF-Dokumente mit komplexem Layout könnten nicht optimal extrahiert werden

### Logging

Die Anwendung protokolliert Aktivitäten und Fehler. Überprüfen Sie die Logs für detaillierte Informationen.

## Beitragen und Weiterentwicklung

Beiträge zur Verbesserung der Web-UI sind willkommen:

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch (`git checkout -b feature/new-feature`)
3. Committen Sie Ihre Änderungen (`git commit -am 'Add new feature'`)
4. Pushen Sie zum Branch (`git push origin feature/new-feature`)
5. Erstellen Sie einen Pull Request
