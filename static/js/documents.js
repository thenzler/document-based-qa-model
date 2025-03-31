/* 
 * JavaScript for the Documents management page
 */

// DOM elements
const documentsLoading = document.getElementById('documents-loading');
const documentsTableContainer = document.getElementById('documents-table-container');
const documentsTableBody = document.getElementById('documents-table-body');
const noDocuments = document.getElementById('no-documents');
const documentUploadForm = document.getElementById('document-upload-form');
const documentUpload = document.getElementById('document-upload');
const uploadDocumentBtn = document.getElementById('upload-document-btn');
const uploadStatus = document.getElementById('upload-status');
const refreshDocsBtn = document.getElementById('refresh-docs-btn');
const documentDetailsContainer = document.getElementById('document-details-container');
const documentTitle = document.getElementById('document-title');
const documentChunks = document.getElementById('document-chunks');
const documentMetadata = document.getElementById('document-metadata');
const documentChunksAccordion = document.getElementById('document-chunks-accordion');
const documentKeywords = document.getElementById('document-keywords');
const closeDocumentDetails = document.getElementById('close-document-details');

// Document processing options
const chunkSizeSlider = document.getElementById('chunk-size');
const chunkSizeValue = document.getElementById('chunk-size-value');
const chunkOverlapSlider = document.getElementById('chunk-overlap');
const chunkOverlapValue = document.getElementById('chunk-overlap-value');
const reprocessBtn = document.getElementById('reprocess-btn');

// Initialize the Documents page
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    documentUploadForm.addEventListener('submit', handleDocumentUpload);
    refreshDocsBtn.addEventListener('click', loadDocuments);
    closeDocumentDetails.addEventListener('click', hideDocumentDetails);
    reprocessBtn.addEventListener('click', reprocessDocuments);
    
    // Set up sliders
    chunkSizeSlider.addEventListener('input', function() {
        chunkSizeValue.textContent = this.value;
    });
    
    chunkOverlapSlider.addEventListener('input', function() {
        chunkOverlapValue.textContent = this.value;
    });
    
    // Load documents list
    loadDocuments();
    
    // Check if system is ready
    checkSystemStatus();
});

/**
 * Load the list of documents
 */
function loadDocuments() {
    // Show loading state
    documentsLoading.classList.remove('d-none');
    documentsTableContainer.classList.add('d-none');
    noDocuments.classList.add('d-none');
    
    // Get documents list
    fetch('/api/document-list')
        .then(response => response.json())
        .then(documents => {
            // Update documents table
            updateDocumentsTable(documents);
        })
        .catch(error => {
            console.error('Error loading documents:', error);
            showError('Fehler beim Laden der Dokumente. Bitte aktualisieren Sie die Seite.');
            
            // Hide loading
            documentsLoading.classList.add('d-none');
        });
}

/**
 * Update the documents table
 */
function updateDocumentsTable(documents) {
    // Hide loading
    documentsLoading.classList.add('d-none');
    
    // No documents
    if (!documents || documents.length === 0) {
        noDocuments.classList.remove('d-none');
        return;
    }
    
    // Show table
    documentsTableContainer.classList.remove('d-none');
    
    // Build table rows
    let tableHtml = '';
    
    documents.forEach(doc => {
        tableHtml += `
            <tr>
                <td>
                    <span class="d-flex align-items-center">
                        <i class="bi bi-file-text me-2"></i>
                        ${sanitizeHTML(doc.filename)}
                    </span>
                </td>
                <td>${formatFileSize(doc.size)}</td>
                <td>
                    <span class="badge bg-primary">${doc.chunks}</span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-info view-doc-btn" data-filename="${sanitizeHTML(doc.filename)}">
                        <i class="bi bi-eye"></i> Anzeigen
                    </button>
                </td>
            </tr>
        `;
    });
    
    // Update table
    documentsTableBody.innerHTML = tableHtml;
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-doc-btn').forEach(button => {
        button.addEventListener('click', function() {
            const filename = this.dataset.filename;
            viewDocument(filename);
        });
    });
}

/**
 * Handle document upload
 */
function handleDocumentUpload(event) {
    event.preventDefault();
    
    // Check if file is selected
    if (!documentUpload.files || documentUpload.files.length === 0) {
        showError('Bitte wählen Sie ein Dokument aus.');
        return;
    }
    
    const file = documentUpload.files[0];
    
    // Check file type
    const allowedTypes = ['.txt', '.pdf', '.docx', '.md', '.html'];
    const isValidType = allowedTypes.some(type => file.name.toLowerCase().endsWith(type));
    
    if (!isValidType) {
        showError(`Nicht unterstützter Dateityp. Erlaubte Typen: ${allowedTypes.join(', ')}`);
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show upload status
    uploadStatus.classList.remove('d-none');
    uploadStatus.classList.remove('alert-danger');
    uploadStatus.classList.add('alert-info');
    uploadStatus.innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-2" role="status"></div>
            <div>${file.name} wird hochgeladen und verarbeitet...</div>
        </div>
    `;
    
    // Disable upload button
    uploadDocumentBtn.disabled = true;
    
    // Upload document
    fetch('/api/upload-document', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Fehler beim Hochladen des Dokuments');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Show success message
        uploadStatus.classList.remove('alert-info');
        uploadStatus.classList.add('alert-success');
        uploadStatus.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-check-circle-fill me-2"></i>
                <div>${file.name} erfolgreich hochgeladen. Verarbeitung läuft im Hintergrund...</div>
            </div>
        `;
        
        // Clear file input
        documentUpload.value = '';
        
        // Reload documents list after a short delay
        setTimeout(() => {
            loadDocuments();
            
            // Hide success message after 5 seconds
            setTimeout(() => {
                uploadStatus.classList.add('d-none');
            }, 5000);
        }, 2000);
    })
    .catch(error => {
        console.error('Error uploading document:', error);
        
        // Show error message
        uploadStatus.classList.remove('alert-info');
        uploadStatus.classList.add('alert-danger');
        uploadStatus.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                <div>Fehler beim Hochladen des Dokuments: ${error.message}</div>
            </div>
        `;
    })
    .finally(() => {
        // Re-enable upload button
        uploadDocumentBtn.disabled = false;
    });
}

/**
 * View a document's details
 */
function viewDocument(filename) {
    // Show loading in document title
    documentTitle.innerHTML = `
        <span class="spinner-border spinner-border-sm text-light me-2" role="status"></span>
        Lade Dokumentdetails...
    `;
    
    // Show document details container
    documentDetailsContainer.classList.remove('d-none');
    
    // Scroll to details container
    documentDetailsContainer.scrollIntoView({ behavior: 'smooth' });
    
    // Get document content
    fetch(`/api/document-content/${filename}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Dokument nicht gefunden');
            }
            return response.json();
        })
        .then(chunks => {
            displayDocumentDetails(filename, chunks);
        })
        .catch(error => {
            console.error('Error loading document content:', error);
            showError(`Fehler beim Laden des Dokuments: ${error.message}`);
            
            // Hide details container
            documentDetailsContainer.classList.add('d-none');
        });
}

/**
 * Display document details
 */
function displayDocumentDetails(filename, chunks) {
    // Set document title
    documentTitle.textContent = filename;
    
    // Set document chunks count
    documentChunks.textContent = `${chunks.length} Chunks`;
    
    // Set document metadata
    documentMetadata.innerHTML = '';
    
    if (chunks.length > 0) {
        // First chunk usually has the document metadata
        const firstChunk = chunks[0];
        
        // Create metadata rows
        let metadataHtml = '';
        
        // File type icon
        let fileTypeIcon = 'bi-file-text';
        if (filename.endsWith('.pdf')) {
            fileTypeIcon = 'bi-file-pdf';
        } else if (filename.endsWith('.docx')) {
            fileTypeIcon = 'bi-file-word';
        } else if (filename.endsWith('.md')) {
            fileTypeIcon = 'bi-file-code';
        } else if (filename.endsWith('.html')) {
            fileTypeIcon = 'bi-file-code';
        }
        
        metadataHtml += `
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi ${fileTypeIcon} mb-3" style="font-size: 2rem;"></i>
                        <h5 class="card-title">${filename}</h5>
                        <p class="card-text text-muted">${getFileExtension(filename).toUpperCase()} Dokument</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-puzzle mb-3" style="font-size: 2rem;"></i>
                        <h5 class="card-title">${chunks.length}</h5>
                        <p class="card-text text-muted">Dokumentchunks</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="bi bi-tags mb-3" style="font-size: 2rem;"></i>
                        <h5 class="card-title">${getUniqueKeywordCount(chunks)}</h5>
                        <p class="card-text text-muted">Eindeutige Schlüsselwörter</p>
                    </div>
                </div>
            </div>
        `;
        
        documentMetadata.innerHTML = metadataHtml;
    }
    
    // Set document chunks
    documentChunksAccordion.innerHTML = '';
    
    if (chunks.length === 0) {
        documentChunksAccordion.innerHTML = '<p class="text-muted">Keine Chunks verfügbar</p>';
    } else {
        chunks.forEach((chunk, index) => {
            const chunkId = `chunk-${index}`;
            const section = chunk.section || 'Kein Abschnittstitel';
            
            documentChunksAccordion.innerHTML += `
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${chunkId}">
                            <span class="me-2 badge bg-secondary">#${index + 1}</span>
                            ${section}
                        </button>
                    </h2>
                    <div id="${chunkId}" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <div class="section-content">${sanitizeHTML(chunk.text)}</div>
                        </div>
                    </div>
                </div>
            `;
        });
    }
    
    // Set document keywords
    documentKeywords.innerHTML = '';
    
    const allKeywords = getAllKeywords(chunks);
    
    if (allKeywords.length === 0) {
        documentKeywords.innerHTML = '<p class="text-muted">Keine Schlüsselwörter verfügbar</p>';
    } else {
        // Sort keywords by frequency
        const keywordCounts = {};
        allKeywords.forEach(keyword => {
            keywordCounts[keyword] = (keywordCounts[keyword] || 0) + 1;
        });
        
        const sortedKeywords = Object.entries(keywordCounts)
            .sort((a, b) => b[1] - a[1])
            .map(entry => entry[0]);
        
        sortedKeywords.forEach(keyword => {
            const count = keywordCounts[keyword];
            documentKeywords.innerHTML += `
                <span class="keyword-badge" title="Kommt ${count}x vor">
                    ${sanitizeHTML(keyword)}
                    <span class="badge bg-secondary ms-1">${count}</span>
                </span>
            `;
        });
    }
}

/**
 * Hide document details container
 */
function hideDocumentDetails() {
    documentDetailsContainer.classList.add('d-none');
}

/**
 * Reprocess documents with new settings
 */
function reprocessDocuments() {
    // Confirm reprocessing
    if (!confirm('Möchten Sie alle Dokumente mit den neuen Einstellungen neu verarbeiten? Dies kann einige Zeit dauern.')) {
        return;
    }
    
    // Get chunk settings
    const chunkSize = parseInt(chunkSizeSlider.value);
    const chunkOverlap = parseInt(chunkOverlapSlider.value);
    
    // Show processing status
    showError(
        `Dokumente werden mit Chunk-Größe ${chunkSize} und Überlappung ${chunkOverlap} neu verarbeitet. ` +
        `Dies kann einige Zeit dauern. Die Seite wird aktualisiert, wenn der Vorgang abgeschlossen ist.`,
        'info'
    );
    
    // Disable reprocess button
    reprocessBtn.disabled = true;
    reprocessBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Verarbeite...';
    
    // Initialize system with new settings
    fetch('/api/initialize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            force_reprocess: true,
            chunk_size: chunkSize,
            chunk_overlap: chunkOverlap
        })
    })
    .then(response => response.json())
    .then(data => {
        // Check status in a loop
        const checkStatus = setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(statusData => {
                    if (statusData.status === 'ready' && !statusData.is_processing) {
                        // Processing completed
                        clearInterval(checkStatus);
                        
                        // Reload documents
                        loadDocuments();
                        
                        // Re-enable reprocess button
                        reprocessBtn.disabled = false;
                        reprocessBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Dokumente neu verarbeiten';
                        
                        // Show success message
                        showError('Dokumente wurden erfolgreich neu verarbeitet.', 'success');
                    } else if (statusData.status === 'error') {
                        // Error occurred
                        clearInterval(checkStatus);
                        
                        // Re-enable reprocess button
                        reprocessBtn.disabled = false;
                        reprocessBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Dokumente neu verarbeiten';
                        
                        // Show error message
                        showError(`Fehler bei der Verarbeitung: ${statusData.message}`);
                    }
                });
        }, 2000);
    })
    .catch(error => {
        console.error('Error reprocessing documents:', error);
        showError('Fehler beim Starten der Neuverarbeitung. Bitte versuchen Sie es später erneut.');
        
        // Re-enable reprocess button
        reprocessBtn.disabled = false;
        reprocessBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Dokumente neu verarbeiten';
    });
}

/**
 * Get all keywords from document chunks
 */
function getAllKeywords(chunks) {
    const keywords = [];
    
    chunks.forEach(chunk => {
        if (chunk.keywords && Array.isArray(chunk.keywords)) {
            keywords.push(...chunk.keywords);
        }
    });
    
    return keywords;
}

/**
 * Get the number of unique keywords
 */
function getUniqueKeywordCount(chunks) {
    const uniqueKeywords = new Set(getAllKeywords(chunks));
    return uniqueKeywords.size;
}

/**
 * Get file extension
 */
function getFileExtension(filename) {
    return filename.split('.').pop();
}

/**
 * Show an error or info message
 */
function showError(message, type = 'danger') {
    // Check if error container exists, create if not
    let errorContainer = document.getElementById('error-container');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'error-container';
        errorContainer.className = `alert alert-${type} alert-dismissible fade show`;
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
    } else {
        // Update class for type
        errorContainer.className = `alert alert-${type} alert-dismissible fade show`;
    }
    
    // Set icon based on type
    let icon = 'exclamation-triangle-fill';
    if (type === 'info') {
        icon = 'info-circle-fill';
    } else if (type === 'success') {
        icon = 'check-circle-fill';
    }
    
    // Set error message
    errorContainer.innerHTML = `
        <i class="bi bi-${icon} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Schließen"></button>
    `;
}
