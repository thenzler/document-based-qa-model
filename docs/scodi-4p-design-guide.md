# SCODi 4P Design-Guide

Diese Anleitung beschreibt, wie das SCODi 4P Design im dokumentenbasierten QA-System implementiert und verwendet wird.

## Übersicht

Das SCODi 4P Design ist ein konsistentes Design-System, das auf der Corporate Identity von SCODi basiert. Es verwendet:

- Eine auf Türkis/Dunkelgrün (#007f78) basierende Farbpalette als Primärfarbe
- Dunkelgrau (#4b5864) als Sekundärfarbe
- Ein modernes, responsives Layout
- Konsistente Typografie und Komponenten

## Dateien im Design-System

Das Design-System besteht aus folgenden Hauptkomponenten:

1. **CSS**: `static/css/scodi-4p.css`
   - Enthält alle Styles für das SCODi 4P Design
   - Definiert Variablen für Farben, Typografie, Abstände und mehr
   - Styles für alle Komponenten (Buttons, Karten, Tabellen, etc.)

2. **JavaScript**: `static/js/scodi-4p.js`
   - Bietet Hilfsfunktionen für interaktive Elemente
   - Initialisiert Tooltips, Dropzones und andere spezifische Komponenten
   - Enthält Utility-Funktionen für Formatierungen und Benachrichtigungen

3. **Layout-Template**: `templates/layout.html`
   - Basis-Layout für alle Seiten
   - Enthält Navigation, Footer und gemeinsame Struktur
   - Einbindung der CSS- und JavaScript-Dateien

## Verwendung des Designs

### 1. In Templates

Um das SCODi 4P Design in einem Template zu verwenden, erweitern Sie das Layout-Template:

```html
{% extends 'layout.html' %}

{% block content %}
<!-- Hier kommt der Seiteninhalt -->
{% endblock %}
```

Die wichtigsten Blöcke, die Sie überschreiben können:

- `head_extra`: Für zusätzliche Styles oder Meta-Tags
- `content`: Für den Hauptinhalt der Seite
- `scripts`: Für zusätzliche JavaScript-Dateien oder -Code

### 2. In der Anwendung

In der Flask-Anwendung müssen Sie das Design-Objekt an die Templates übergeben:

```python
return render_template('template.html', 
                      design=SCODI_DESIGN,
                      page_title="Titel der Seite")
```

Das Design-Objekt enthält alle Konfigurationsvariablen für das Design, wie z.B. Farben, Flags für Feature-Anzeige usw.

### 3. CSS-Klassen verwenden

Die wichtigsten CSS-Klassen sind:

- `.scodi-4p`: Grundklasse für das Design (wird automatisch auf den Body-Tag angewendet)
- `.btn-primary`, `.btn-secondary`: Für Buttons im SCODi-Stil
- `.card`: Für Karten/Panels
- `.navbar`: Für die Navigation

Beispiel für einen Button:

```html
<button class="btn btn-primary">Aktion ausführen</button>
```

## Farbpalette

Die Farbvariablen sind als CSS-Variablen definiert:

- `--scodi-primary`: #007f78 (Hauptfarbe)
- `--scodi-primary-light`: #3a9e99 (Hellere Variante)
- `--scodi-primary-dark`: #005752 (Dunklere Variante)
- `--scodi-secondary`: #4b5864 (Sekundärfarbe)
- `--scodi-accent`: #f7f7f7 (Akzentfarbe für Hintergründe)

Statusfarben:

- `--scodi-success`: #32a852 (Erfolg/Grün)
- `--scodi-warning`: #ffc107 (Warnung/Gelb)
- `--scodi-error`: #dc3545 (Fehler/Rot)
- `--scodi-info`: #17a2b8 (Info/Blau)

## Komponenten

### Buttons

```html
<button class="btn btn-primary">Primärer Button</button>
<button class="btn btn-secondary">Sekundärer Button</button>
<button class="btn btn-outline-primary">Outline Button</button>
```

### Karten/Cards

```html
<div class="card">
    <div class="card-header">
        <h3>Kartentitel</h3>
    </div>
    <div class="card-body">
        Karteninhalt
    </div>
    <div class="card-footer">
        Kartenfooter
    </div>
</div>
```

### Formular-Elemente

```html
<div class="form-group">
    <label class="form-label">Bezeichnung</label>
    <input type="text" class="form-control" placeholder="Eingabe">
</div>
```

### Benachrichtigungen

Toast-Benachrichtigungen können mit der JavaScript-Funktion `scodiNotify()` erstellt werden:

```javascript
scodiNotify('Aktion erfolgreich ausgeführt!', 'success');
scodiNotify('Fehler bei der Verarbeitung', 'error');
```

## Responsive Design

Das SCODi 4P Design ist vollständig responsive und passt sich verschiedenen Bildschirmgrößen an. Verwenden Sie die Bootstrap-Klassen für das Grid-System:

```html
<div class="row">
    <div class="col-md-6 col-lg-4">Spalte 1</div>
    <div class="col-md-6 col-lg-4">Spalte 2</div>
    <div class="col-md-6 col-lg-4">Spalte 3</div>
</div>
```

## Anpassung bestehender Templates

Um bestehende Templates auf das SCODi 4P Design umzustellen:

1. Ändern Sie den Template-Header so, dass er `layout.html` erweitert
2. Verschieben Sie den Hauptinhalt in den `content`-Block
3. Verschieben Sie benutzerdefinierte Skripte in den `scripts`-Block
4. Entfernen Sie doppelte Elemente wie Navigation und Footer

## JavaScript-Hilfsfunktionen

Das SCODi 4P Design bietet verschiedene Hilfsfunktionen:

- `scodiNotify(message, type, duration)`: Zeigt eine Toast-Benachrichtigung an
- `formatEuro(amount)`: Formatiert einen Betrag als Euro
- `formatDate(date)`: Formatiert ein Datum ins deutsche Format
- `truncateText(text, maxLength)`: Kürzt einen Text auf eine bestimmte Länge

## Bekannte Probleme und Lösungen

- **Farbpräferenz-Konflikte**: Bei Konflikten mit den Standard-Bootstrap-Farben müssen die SCODi-Farben mit `!important` überschrieben werden
- **JavaScript-Konflikte**: Wenn bereits ein `document.ready`-Event vorhanden ist, müssen die SCODi-Initialisierungsfunktionen manuell aufgerufen werden
