/* 
 * Main JavaScript for Document-based QA and Churn Prediction UI
 * Contains shared functionality used across all pages
 */

// Global variables
let systemStatus = {
    qaSystemReady: false,
    docProcessorReady: false,
    churnModelReady: false,
    isProcessing: false,
    status: 'idle',
    message: ''
};

// DOM elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const statusSpinner = document.getElementById('status-spinner');

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Check system status on page load
    checkSystemStatus();
    
    // Set up polling for status updates
    setInterval(checkSystemStatus, 5000);
});

/**
 * Check the current system status
 */
function checkSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
            updateStatusUI();
            
            // If on the home page, update the system components
            if (document.getElementById('system-components')) {
                updateSystemComponents(data);
            }
            
            // If document stats are available, update them
            if (data.document_stats && document.getElementById('document-stats')) {
                updateDocumentStats(data.document_stats);
            }
        })
        .catch(error => {
            console.error('Error checking system status:', error);
            showError('Fehler beim Abrufen des Systemstatus. Bitte aktualisieren Sie die Seite.');
        });
}

/**
 * Update the system status with new data
 */
function updateSystemStatus(data) {
    systemStatus = {
        qaSystemReady: data.qa_system_ready || false,
        docProcessorReady: data.doc_processor_ready || false,
        churnModelReady: data.churn_model_ready || false,
        isProcessing: data.is_processing || false,
        status: data.status || 'idle',
        message: data.message || ''
    };
}

/**
 * Update the status UI based on current system status
 */
function updateStatusUI() {
    // Update status text
    statusText.textContent = systemStatus.message || 'Bereit';
    
    // Show/hide spinner based on processing status
    if (systemStatus.isProcessing || systemStatus.status === 'initializing' || 
        systemStatus.status === 'processing' || systemStatus.status === 'loading') {
        statusSpinner.classList.remove('d-none');
    } else {
        statusSpinner.classList.add('d-none');
    }
    
    // Update status indicator color
    statusIndicator.className = 'status-indicator';
    if (systemStatus.status === 'error') {
        statusIndicator.classList.add('text-danger');
    } else if (systemStatus.status === 'ready') {
        statusIndicator.classList.add('text-success');
    } else if (systemStatus.status === 'processing' || systemStatus.status === 'initializing' || 
               systemStatus.status === 'loading' || systemStatus.status === 'uploading') {
        statusIndicator.classList.add('text-warning');
    }
    
    // Update initialize button visibility
    const initializeControls = document.getElementById('initialize-controls');
    if (initializeControls) {
        if (!systemStatus.qaSystemReady || !systemStatus.docProcessorReady || !systemStatus.churnModelReady) {
            initializeControls.classList.remove('d-none');
        } else {
            initializeControls.classList.add('d-none');
        }
    }
}

/**
 * Update system components on the home page
 */
function updateSystemComponents(data) {
    // QA System
    const qaStatus = document.getElementById('qa-status');
    const qaProgress = document.getElementById('qa-progress');
    
    if (data.qa_system_ready) {
        qaStatus.textContent = 'Bereit';
        qaProgress.style.width = '100%';
        qaProgress.classList.remove('progress-bar-animated');
        qaProgress.classList.add('bg-success');
    } else if (data.status === 'initializing' && data.message.includes('QA')) {
        qaStatus.textContent = 'Wird initialisiert...';
        qaProgress.style.width = '50%';
        qaProgress.classList.add('progress-bar-animated');
        qaProgress.classList.add('bg-warning');
    } else {
        qaStatus.textContent = 'Nicht initialisiert';
        qaProgress.style.width = '0%';
    }
    
    // Document Processor
    const docStatus = document.getElementById('doc-status');
    const docProgress = document.getElementById('doc-progress');
    
    if (data.doc_processor_ready) {
        docStatus.textContent = 'Bereit';
        docProgress.style.width = '100%';
        docProgress.classList.remove('progress-bar-animated');
        docProgress.classList.add('bg-success');
    } else if (data.is_processing || (data.status === 'processing' && data.message.includes('Dokument'))) {
        docStatus.textContent = 'Verarbeite Dokumente...';
        docProgress.style.width = '50%';
        docProgress.classList.add('progress-bar-animated');
        docProgress.classList.add('bg-warning');
    } else {
        docStatus.textContent = 'Keine Dokumente verarbeitet';
        docProgress.style.width = '0%';
    }
    
    // Churn Model
    const modelStatus = document.getElementById('model-status');
    const modelProgress = document.getElementById('model-progress');
    
    if (data.churn_model_ready) {
        modelStatus.textContent = 'Bereit';
        modelProgress.style.width = '100%';
        modelProgress.classList.remove('progress-bar-animated');
        modelProgress.classList.add('bg-success');
    } else if (data.status === 'training' || data.status === 'loading' && data.message.includes('churn')) {
        modelStatus.textContent = 'Wird geladen/trainiert...';
        modelProgress.style.width = '50%';
        modelProgress.classList.add('progress-bar-animated');
        modelProgress.classList.add('bg-warning');
    } else {
        modelStatus.textContent = 'Nicht geladen';
        modelProgress.style.width = '0%';
    }
}

/**
 * Update document statistics on the home page
 */
function updateDocumentStats(stats) {
    const statsContainer = document.getElementById('document-stats');
    if (!statsContainer) return;
    
    let html = '';
    
    if (stats.total_documents === 0) {
        html = `
            <div class="text-center py-4">
                <i class="bi bi-file-earmark-x text-muted" style="font-size: 3rem;"></i>
                <p class="mt-2">Keine Dokumente verfügbar. Gehen Sie zur Dokumentenverwaltung, um Dokumente hinzuzufügen.</p>
            </div>
        `;
    } else {
        html = `
            <div class="row text-center">
                <div class="col-md-4">
                    <div class="card bg-light mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Dokumente</h5>
                            <p class="display-4">${stats.total_documents}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Chunks</h5>
                            <p class="display-4">${stats.chunks}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Ø Chunk-Größe</h5>
                            <p class="display-4">${stats.avg_chunk_size}</p>
                            <p class="mb-0">Zeichen</p>
                        </div>
                    </div>
                </div>
            </div>
            <h5 class="mt-3 mb-2">Dokumentquellen:</h5>
            <div class="row">
        `;
        
        // Add document sources
        stats.sources.forEach(source => {
            html += `
                <div class="col-md-6 mb-2">
                    <div class="d-flex justify-content-between align-items-center p-2 bg-light rounded">
                        <span><i class="bi bi-file-text me-2"></i>${source.name}</span>
                        <span class="badge bg-primary">${source.chunks} Chunks</span>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    statsContainer.innerHTML = html;
}

/**
 * Initialize the system components
 */
function initializeSystem() {
    // Show initializing state
    statusText.textContent = 'Initialisierung...';
    statusSpinner.classList.remove('d-none');
    
    // Disable initialize button during initialization
    const initializeBtn = document.getElementById('initialize-btn');
    if (initializeBtn) {
        initializeBtn.disabled = true;
        initializeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Initialisiere...';
    }
    
    // Call API to initialize system
    fetch('/api/initialize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            use_generation: true,
            force_reprocess: false
        })
    })
    .then(response => response.json())
    .then(data => {
        // System initialization started, status will be updated by polling
        console.log('System initialization started:', data);
    })
    .catch(error => {
        console.error('Error initializing system:', error);
        showError('Fehler bei der Systeminitialisierung. Bitte aktualisieren Sie die Seite und versuchen Sie es erneut.');
        
        // Reset initialize button
        if (initializeBtn) {
            initializeBtn.disabled = false;
            initializeBtn.innerHTML = '<i class="bi bi-play-fill"></i> System initialisieren';
        }
    });
}

/**
 * Show an error message
 */
function showError(message) {
    // Check if error container exists, create if not
    let errorContainer = document.getElementById('error-container');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'error-container';
        errorContainer.className = 'alert alert-danger alert-dismissible fade show';
        errorContainer.setAttribute('role', 'alert');
        
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close';
        closeButton.setAttribute('data-bs-dismiss', 'alert');
        closeButton.setAttribute('aria-label', 'Schließen');
        
        errorContainer.appendChild(closeButton);
        
        // Add to main content
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorContainer, main.firstChild);
        }
    }
    
    // Set error message
    errorContainer.innerHTML = `
        <i class="bi bi-exclamation-triangle-fill me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Schließen"></button>
    `;
}

/**
 * Format file size to human-readable string
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Helper function to sanitize HTML to prevent XSS
 */
function sanitizeHTML(text) {
    const temp = document.createElement('div');
    temp.textContent = text;
    return temp.innerHTML;
}

/**
 * Helper function to create a risk badge
 */
function createRiskBadge(riskCategory) {
    let badgeClass = '';
    
    if (riskCategory.includes('Hoh')) {
        badgeClass = 'risk-high';
    } else if (riskCategory.includes('Mittel')) {
        badgeClass = 'risk-medium';
    } else {
        badgeClass = 'risk-low';
    }
    
    return `<span class="risk-badge ${badgeClass}">${riskCategory}</span>`;
}
