# SCODi 4P - Dokumentenbasiertes QA-System

Ein dokumentenbasiertes Frage-Antwort-System mit modernem SCODi 4P Design für langfristige, nachhaltige Entwicklung.

## Moderne Design-Variante

Dieses Repository enthält zwei Design-Varianten des Systems:

1. **Klassisches Design**: Die ursprüngliche Implementierung mit angepasstem SCODi 4P Design
2. **Modernes Design**: Eine minimalistische, zukunftssichere Version mit reduziertem Design

### Vorteile des modernen Designs

- **Konsistenz**: Einheitliche Designsprache in allen Komponenten
- **Wartbarkeit**: Vereinfachte CSS-Struktur ohne Abhängigkeit von umfangreichen Frameworks
- **Erweiterbarkeit**: Modularer Aufbau, der einfache Anpassungen ermöglicht
- **Performance**: Reduzierte Abhängigkeiten und optimierte Ladezeiten

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

Das moderne SCODi 4P Design-System ist auf langfristige Entwicklung und Wartbarkeit ausgelegt:

- **Farbpalette**: Konsistente Farbvariablen für alle UI-Elemente
- **Komponenten**: Wiederverwendbare UI-Komponenten mit einheitlichem Styling
- **Typografie**: Klare Hierarchie und Lesbarkeit durch definierte Schriftgrößen
- **Responsive Design**: Optimierung für alle Bildschirmgrößen

### Verwendete Technologien

- **CSS-Variables**: Für konsistente Designwerte ohne Präprozessoren
- **SVG-Icons**: Skalierbare Vektorgrafiken für gestochen scharfe Darstellung
- **Modernes Layout**: Flexbox und CSS Grid für robuste Layouts
- **Minimale Abhängigkeiten**: Reduzierter Einsatz von externen Bibliotheken

## Installation und Verwendung

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
   pip install -r requirements.txt
   ```

### Starten der Anwendung

#### Modernes Design

```bash
python scodi_modern_app.py
```

#### Klassisches Design

```bash
python app.py
```

Öffne dann einen Browser und navigiere zu:
```
http://localhost:5000
```

## Vergleich der Design-Varianten

### Modernes Design

- **Dateien**: 
  - CSS: `static/css/scodi-4p-modern.css`
  - Layout: `templates/modern_layout.html`
  - App: `scodi_modern_app.py`

- **Merkmale**:
  - Horizontale Navigation mit Icons
  - Minimalistisches, cleanes Design
  - Responsive ohne umfangreiche Frameworks
  - Zentrale SVG-Logo-Darstellung

### Klassisches Design

- **Dateien**:
  - CSS: `static/css/scodi-4p.css`
  - Layout: `templates/layout.html`
  - App: `app.py`

- **Merkmale**:
  - Vertikale Navigation
  - Mehr Bootstrap-basierte Komponenten
  - Komplexere Gestaltung mit mehr visuellen Elementen

## Weiterentwicklung

Für die langfristige Entwicklung empfehlen wir die Nutzung des modernen Designs, da es bessere Voraussetzungen für Wartbarkeit, Erweiterbarkeit und Zukunftssicherheit bietet. Die folgenden Schritte können bei der Weiterentwicklung nützlich sein:

1. **Komponenten-Bibliothek**: Erweitern Sie die UI-Komponenten basierend auf dem bestehenden Design-System
2. **Template-Überarbeitung**: Wandeln Sie schrittweise alle Templates in das moderne Design um
3. **Modularisierung**: Teilen Sie die Anwendung in kleinere, spezialisierte Module auf
4. **Tests**: Fügen Sie automatisierte Tests für UI-Komponenten und Backend-Funktionalität hinzu

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht - siehe LICENSE-Datei für Details.