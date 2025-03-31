# Dokumentenbasiertes Frage-Antwort-System mit Churn-Prediction

Ein Machine Learning Modell für dokumentenbasierte Frage-Antwort-Systeme mit Churn-Prediction als Anwendungsbeispiel. 

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
   - Intuitive Benutzeroberfläche für alle Funktionen
   - Datei-Upload und -Management
   - Visualisierung der Ergebnisse
   - Anpassbare Einstellungen

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
   - Flask-basiertes Web-Interface
   - Responsive Design mit Bootstrap
   - Interaktive Komponenten mit JavaScript

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
