# Dokumentenbasiertes Frage-Antwort-System mit SCODi 4P Design

Ein Machine Learning Modell für dokumentenbasierte Frage-Antwort-Systeme mit Churn-Prediction als Anwendungsbeispiel, designt nach SCODi 4P Standards.

Das System nutzt Natural Language Processing (NLP), um auf Basis von Dokumenten Fragen zu beantworten und enthält ein Modul zur Kundenfluktuation-Vorhersage (Churn-Prediction).

## Funktionen

Das System bietet folgende Hauptfunktionen:

1. **Dokumentenbasiertes Frage-Antwort-System**:
   - Dokumente in verschiedenen Formaten einlesen und verarbeiten (PDF, DOCX, TXT, MD, HTML)
   - Dokumente in Abschnitte und Chunks unterteilen
   - Semantische Suche nach relevanten Passagen
   - Generieren von Antworten auf Fragen basierend auf den Dokumentinhalten

2. **Churn-Prediction**:
   - Machine Learning Modell zur Vorhersage von Kundenabwanderung
   - Identifikation von Risikofaktoren
   - Vorschläge für Maßnahmen basierend auf den Dokumenten
   - Visualisierung der Ergebnisse

3. **Web-Benutzeroberfläche**:
   - Intuitive Benutzeroberfläche im SCODi 4P Design
   - Datei-Upload und -Management
   - Visualisierung der Ergebnisse
   - Anpassbare Einstellungen

## Design-System

Das Projekt verwendet das SCODi 4P Design-System mit folgenden Komponenten:

- Farbschema basierend auf SCODi Corporate Identity
- Responsive Layout und Komponenten
- Konsistente Typografie und Ikonografie
- Angepasste Navigation und Footer

### Farben

- **Primärfarbe**: #007f78 (Dunkelgrün/Türkis)
- **Sekundärfarbe**: #4b5864 (Dunkelgrau)
- **Akzentfarbe**: #f7f7f7 (Hellgrau für Hintergründe)

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Manager)

### Installation der Abhängigkeiten

1. Klone das Repository:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt  # Für die grundlegenden Funktionen
   pip install -r requirements-ui.txt  # Für die Web-UI
   ```

### Erste Schritte

1. Starte die Anwendung:
   ```bash
   python app.py
   ```

2. Öffne einen Browser und navigiere zu:
   ```
   http://localhost:5000
   ```

3. Folge den Anweisungen auf der Startseite, um das System zu initialisieren und Dokumente hochzuladen.

## Nutzung

### Dokumentenbasiertes Frage-Antwort-System

1. Navigiere zum Bereich "Frage & Antwort"
2. Gib eine Frage ein, die basierend auf deinen Dokumenten beantwortet werden soll
3. Das System sucht nach relevanten Passagen und generiert eine Antwort mit Quellenangaben

### Churn-Prediction

1. Navigiere zum Bereich "Churn-Prediction"
2. Wähle zwischen Beispieldaten oder dem Hochladen eigener Kundendaten (CSV)
3. Starte die Vorhersage, um Kunden mit Abwanderungsrisiko zu identifizieren
4. Für jeden Kunden werden Risikofaktoren und empfohlene Maßnahmen angezeigt

### Dokumente verwalten

1. Navigiere zum Bereich "Dokumente"
2. Lade neue Dokumente hoch (PDF, DOCX, TXT, MD, HTML)
3. Passe Einstellungen für die Dokumentverarbeitung an

## Architektur

Das System basiert auf einer modularen Architektur mit folgenden Komponenten:

1. **Dokumentenverarbeitung** (`DocumentProcessor`):
   - Extraktion von Text aus verschiedenen Dokumentformaten
   - Unterteilung in Chunks
   - Speicherung und Verwaltung der verarbeiteten Dokumente

2. **Frage-Antwort-System** (`DocumentQA`):
   - Suche nach relevanten Passagen
   - Generierung von Antworten basierend auf den Dokumenten

3. **Churn-Vorhersage** (`ChurnModel`):
   - Machine Learning Modell für die Vorhersage von Kundenabwanderung
   - Identifikation von Risikofaktoren und Empfehlungen

4. **Web-Oberfläche**:
   - Flask-basiertes Web-Interface im SCODi 4P Design
   - Responsive Design
   - Interaktive Komponenten mit JavaScript

## SCODi 4P Anpassungen

Diese Version enthält SCODi 4P Design-Anpassungen, die folgende Dateien umfassen:

- `static/css/scodi-4p.css`: Stylesheet für SCODi 4P Design
- `static/js/scodi-4p.js`: JavaScript-Funktionen für Interaktionen
- `templates/layout.html`: Basis-Layout im SCODi 4P Design

### Design-Implementierung

Um das SCODi 4P Design zu verwenden:

1. Kopiere die CSS- und JS-Dateien in die entsprechenden Verzeichnisse
2. Ersetze das bestehende Layout durch das neue SCODi 4P Layout
3. Passe die Anwendung an, um Designvariablen zu übergeben:
   ```python
   return render_template('index.html', design=SCODI_DESIGN)
   ```

## Konfiguration

Die Konfigurationsparameter können in der Datei `app.py` angepasst werden.

## Beitrag

Beiträge zum Projekt sind willkommen! Bitte folge diesen Schritten:

1. Forke das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/new-feature`)
3. Committe deine Änderungen (`git commit -am 'Add new feature'`)
4. Pushe zum Branch (`git push origin feature/new-feature`)
5. Erstelle einen Pull Request

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht - siehe LICENSE-Datei für Details.
