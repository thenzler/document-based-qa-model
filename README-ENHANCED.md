# Verbessertes Dokumentenbasiertes QA-System mit OpenAI/Claude Integration

Eine Weiterentwicklung des dokumentenbasierten QA-Systems mit verbesserter LLM-Funktionalität und einheitlicher Benutzeroberfläche.

## Hauptverbesserungen

1. **Erweiterte LLM-Funktionalität**:
   - Integration mit OpenAI GPT-4/GPT-4o für präzisere Antworten
   - Anthropic Claude API als Fallback-Option
   - Verbesserte RAG-Architektur mit Cross-Encoder Reranking
   - Semantische Chunking-Strategien für bessere Kontexterstellung
   - Multi-Query-Ansatz für robustere Retrievals

2. **Einheitliche Benutzeroberfläche**:
   - Alles auf einer Hauptseite mit Tab-basierter Navigation
   - Moderne UI mit SCODi 4P Design-Vorgaben
   - Live-Systemstatus mit API-Konfigurationsmöglichkeiten
   - Verbesserte Dokumentenverwaltung
   - Antworten mit detaillierten Quellennachweisen

3. **Verbesserte Entwicklererfahrung**:
   - Modularere Codestruktur
   - Verbesserte Logging und Fehlerbehandlung
   - Einfache API-Schlüssel-Konfiguration über die UI oder Umgebungsvariablen
   - Optimierte Abhängigkeiten

## Installation

1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Installieren Sie die Abhängigkeiten:
   ```bash
   pip install -r requirements-enhanced.txt
   ```

3. Konfigurieren Sie die API-Schlüssel (optional):
   - Option 1: Setzen Sie Umgebungsvariablen:
     ```bash
     export OPENAI_API_KEY="Ihr-OpenAI-API-Schlüssel"
     export ANTHROPIC_API_KEY="Ihr-Anthropic-API-Schlüssel"
     ```
   - Option 2: Konfigurieren Sie die Schlüssel später über die Weboberfläche

4. Starten Sie die verbesserte Anwendung:
   ```bash
   python enhanced_app.py
   ```

5. Öffnen Sie die Anwendung in Ihrem Browser:
   ```
   http://localhost:5000
   ```

## Verwendung

### Dokumente hochladen

1. Navigieren Sie zum "Dokumente"-Tab
2. Wählen Sie eine Datei aus (PDF, DOCX, TXT, MD, HTML, usw.)
3. Klicken Sie auf "Hochladen"
4. Die verarbeiteten Dokumente erscheinen in der Dokumentenliste

### Fragen stellen

1. Navigieren Sie zum "Frage & Antwort"-Tab
2. Geben Sie Ihre Frage in das Textfeld ein
3. Aktivieren Sie "Websearch für unbekannte Fragen" (optional)
4. Klicken Sie auf "Frage stellen"
5. Die Antwort wird zusammen mit den relevanten Quellen angezeigt

### API-Konfiguration

1. Klicken Sie auf "Einstellungen" in der Navigationsleiste
2. Geben Sie Ihre API-Schlüssel ein
3. Klicken Sie auf "Speichern"

## Konfiguration

Sie können verschiedene Aspekte der Anwendung konfigurieren:

- **LLM-Auswahl**: Die Anwendung verwendet standardmäßig OpenAI, wenn verfügbar, mit Claude als Fallback
- **Embedding-Modell**: Das Standard-Embedding-Modell ist `all-mpnet-base-v2`
- **GPU-Beschleunigung**: Aktiviert automatisch, wenn verfügbar
- **Websearch**: Kann für Fragen aktiviert werden, die nicht durch lokale Dokumente beantwortet werden können

## Features im Detail

### RAG-Architektur

Die verbesserte Implementierung verwendet eine fortschrittliche RAG-Architektur:

1. **Dokumentensegmentierung**: Intelligentes Chunking mit semantischen Grenzen
2. **Embedding-Generierung**: Hochwertige Einbettungen durch SentenceTransformers
3. **Retrieval**: Effiziente Vektorsuche mit FAISS
4. **Reranking**: Cross-Encoder-Modelle zur Präzisionsverbesserung
5. **Antwortgenerierung**: Kontextbasierte Antworten durch OpenAI/Claude
6. **Quellennachweis**: Automatische Verfolgung der Informationsquellen

### UI-Komponenten

- **Frage-Antwort-Interface**: Hauptbereich zum Stellen von Fragen
- **Dokumentenverwaltung**: Upload und Übersicht über alle Dokumente
- **System-Status**: Live-Anzeige des Systemzustands
- **Einstellungen**: Konfiguration der LLM-APIs

## Fehlerbehebung

- **API-Fehler**: Stellen Sie sicher, dass Ihre API-Schlüssel korrekt sind und das entsprechende Guthaben vorhanden ist
- **Dokumentenverarbeitung**: Überprüfen Sie die unterstützten Formate und die Dateigröße
- **Antwortqualität**: Verwenden Sie präzisere Fragen und laden Sie zusätzliche Dokumente hoch, wenn Antworten ungenau sind

## Technologie-Stack

- **Backend**: Flask, Python 3.8+
- **LLM-APIs**: OpenAI API, Anthropic Claude API
- **Embedding-Modelle**: SentenceTransformers
- **Vektordatenbank**: FAISS
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Dokumentenverarbeitung**: PyPDF, python-docx, BeautifulSoup

## Beitragende

Entwickelt auf Basis des ursprünglichen document-based-qa-model mit signifikanten Verbesserungen und Erweiterungen.

## Kontakt

Bei Fragen oder Anmerkungen erstellen Sie bitte ein Issue im GitHub-Repository.
