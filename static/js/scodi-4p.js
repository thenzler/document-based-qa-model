/**
 * SCODi 4P JavaScript Utilities
 * Ergänzende JavaScript-Funktionen für das SCODi 4P Design
 */

// Initialisierung wenn das DOM geladen ist
document.addEventListener('DOMContentLoaded', function() {
    initScodiDesignSystem();
});

/**
 * Initialisiert die SCODi 4P Design-Komponenten
 */
function initScodiDesignSystem() {
    // Aktiviere Tooltips
    if (typeof $ !== 'undefined' && typeof $.fn.tooltip !== 'undefined') {
        $('[data-toggle="tooltip"]').tooltip();
    }
    
    // Aktiviere Popovers
    if (typeof $ !== 'undefined' && typeof $.fn.popover !== 'undefined') {
        $('[data-toggle="popover"]').popover();
    }
    
    // Initialisiere SCODi-spezifische Komponenten
    initDropzones();
    initAnswerAnimations();
    initProcessComponents();
    updateDynamicElements();
}

/**
 * Initialisiert File-Upload-Dropzones
 */
function initDropzones() {
    const dropzones = document.querySelectorAll('.file-upload-area');
    
    dropzones.forEach(dropzone => {
        dropzone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        dropzone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        dropzone.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            // Falls ein File-Input Element vorhanden ist
            const fileInput = this.querySelector('input[type="file"]');
            if (fileInput && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
                
                // Update display
                const fileNameDisplay = this.querySelector('.file-name-display');
                if (fileNameDisplay) {
                    const fileName = e.dataTransfer.files[0].name;
                    fileNameDisplay.textContent = fileName;
                    fileNameDisplay.classList.remove('d-none');
                }
            }
        });
    });
}

/**
 * Initialisiert Animationen für die Antwortgenerierung
 */
function initAnswerAnimations() {
    const answerForms = document.querySelectorAll('.question-form');
    
    answerForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Falls ein Antwortbereich bereits existiert
            const existingAnswerBox = document.querySelector('.answer-box');
            if (existingAnswerBox) {
                // Füge Loading-Animation hinzu
                existingAnswerBox.innerHTML = '<div class="text-center p-4"><div class="spinner-border text-primary" role="status"><span class="sr-only">Generiere Antwort...</span></div><div class="mt-3">Generiere Antwort...</div></div>';
            } else {
                // Erstelle neuen Antwortbereich mit Loading-Animation
                const answerContainer = document.querySelector('#answer-container');
                if (answerContainer) {
                    answerContainer.innerHTML = '<div class="answer-box"><div class="text-center p-4"><div class="spinner-border text-primary" role="status"><span class="sr-only">Generiere Antwort...</span></div><div class="mt-3">Generiere Antwort...</div></div></div>';
                }
            }
        });
    });
}

/**
 * Initialisiert Prozess-bezogene Komponenten
 */
function initProcessComponents() {
    // Prozess-Diagramme mit Klick-Interaktionen
    const processItems = document.querySelectorAll('.process-item');
    
    processItems.forEach(item => {
        item.addEventListener('click', function() {
            const processId = this.getAttribute('data-process-id');
            if (processId) {
                window.location.href = `/processes/${processId}`;
            }
        });
    });
    
    // ISO-Badge Tooltips
    const isoBadges = document.querySelectorAll('.iso-badge');
    
    isoBadges.forEach(badge => {
        // Füge Tooltip-Informationen hinzu
        const isoNumber = badge.getAttribute('data-iso');
        let tooltipText = '';
        
        switch (isoNumber) {
            case '9001':
                tooltipText = 'ISO 9001: Qualitätsmanagement';
                break;
            case '14001':
                tooltipText = 'ISO 14001: Umweltmanagement';
                break;
            case '45001':
                tooltipText = 'ISO 45001: Arbeitsschutzmanagement';
                break;
            case '27001':
                tooltipText = 'ISO 27001: Informationssicherheit';
                break;
            default:
                tooltipText = `ISO ${isoNumber}`;
        }
        
        badge.setAttribute('title', tooltipText);
        badge.setAttribute('data-toggle', 'tooltip');
    });
}

/**
 * Aktualisiert dynamische UI-Elemente
 */
function updateDynamicElements() {
    // Füge aktuelle Jahreszahl in Copyright ein
    const copyrightElements = document.querySelectorAll('.copyright-year');
    const currentYear = new Date().getFullYear();
    
    copyrightElements.forEach(element => {
        element.textContent = currentYear;
    });
}

/**
 * SCODi Notify - Toast-Benachrichtigungen
 * @param {string} message - Anzuzeigende Nachricht
 * @param {string} type - Typ der Nachricht (success, error, warning, info)
 * @param {number} duration - Dauer in ms, wie lange die Nachricht angezeigt wird
 */
function scodiNotify(message, type = 'info', duration = 3000) {
    // Erstelle Toast-Container falls noch nicht vorhanden
    let toastContainer = document.querySelector('.toast-container');
    
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1050';
        document.body.appendChild(toastContainer);
    }
    
    // Definiere Farben je nach Typ
    let bgColor, iconClass;
    
    switch (type) {
        case 'success':
            bgColor = 'var(--scodi-success)';
            iconClass = 'fas fa-check-circle';
            break;
        case 'error':
            bgColor = 'var(--scodi-error)';
            iconClass = 'fas fa-exclamation-circle';
            break;
        case 'warning':
            bgColor = 'var(--scodi-warning)';
            iconClass = 'fas fa-exclamation-triangle';
            break;
        default:
            bgColor = 'var(--scodi-info)';
            iconClass = 'fas fa-info-circle';
    }
    
    // Erstelle Toast-Element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = 'toast show';
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="toast-header" style="background-color: ${bgColor}; color: white;">
            <i class="${iconClass} me-2"></i>
            <strong class="me-auto">SCODi 4P</strong>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Schließen" onclick="document.getElementById('${toastId}').remove()"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Füge Toast zum Container hinzu
    toastContainer.appendChild(toast);
    
    // Entferne Toast nach Zeitablauf
    setTimeout(() => {
        toast.remove();
    }, duration);
}

/**
 * Formatiert eine Zahl als Euro-Betrag
 * @param {number} amount - Zu formatierender Betrag
 * @returns {string} Formatierter Euro-Betrag
 */
function formatEuro(amount) {
    return new Intl.NumberFormat('de-DE', { 
        style: 'currency', 
        currency: 'EUR'
    }).format(amount);
}

/**
 * Formatiert ein Datum ins deutsche Format
 * @param {string|Date} date - Zu formatierendes Datum
 * @returns {string} Formatiertes Datum
 */
function formatDate(date) {
    const dateObj = date instanceof Date ? date : new Date(date);
    return dateObj.toLocaleDateString('de-DE', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });
}

/**
 * Kürzt einen Text auf eine bestimmte Länge
 * @param {string} text - Zu kürzender Text
 * @param {number} maxLength - Maximale Länge
 * @returns {string} Gekürzter Text
 */
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) {
        return text;
    }
    
    return text.substring(0, maxLength) + '...';
}