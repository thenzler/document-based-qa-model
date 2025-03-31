/* 
 * JavaScript for the Churn Prediction page
 */

// DOM elements
const sampleDataRadio = document.getElementById('sample-data');
const customDataRadio = document.getElementById('custom-data');
const uploadContainer = document.getElementById('upload-container');
const csvUploadInput = document.getElementById('csv-upload');
const uploadBtn = document.getElementById('upload-btn');
const predictBtn = document.getElementById('predict-btn');
const dataPreviewContainer = document.getElementById('data-preview-container');
const dataPreviewHeader = document.getElementById('data-preview-header');
const dataPreviewBody = document.getElementById('data-preview-body');
const resultsContainer = document.getElementById('results-container');
const predictionLoading = document.getElementById('prediction-loading');
const resultsSummary = document.getElementById('results-summary');
const detailedResults = document.getElementById('detailed-results');
const resultsTableBody = document.getElementById('results-table-body');
const highRiskCount = document.getElementById('high-risk-count');
const mediumRiskCount = document.getElementById('medium-risk-count');
const lowRiskCount = document.getElementById('low-risk-count');

// Customer modal elements
const customerModal = new bootstrap.Modal(document.getElementById('customer-details-modal'));
const customerModalLabel = document.getElementById('customerDetailsModalLabel');
const customerInfo = document.getElementById('customer-info');
const customerRiskBar = document.getElementById('customer-risk-bar');
const riskDetails = document.getElementById('risk-details');
const riskFactorsList = document.getElementById('risk-factors-list');
const interventions = document.getElementById('interventions');
const documentReferences = document.getElementById('document-references');

// Customer data
let customerData = null;
let predictionResults = null;

// Initialize the Churn page
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    sampleDataRadio.addEventListener('change', toggleDataSource);
    customDataRadio.addEventListener('change', toggleDataSource);
    uploadBtn.addEventListener('click', handleFileUpload);
    predictBtn.addEventListener('click', predictChurn);
    
    // Load sample data initially
    loadSampleData();
    
    // Check if system is ready
    checkSystemStatus();
});

/**
 * Toggle between sample and custom data sources
 */
function toggleDataSource() {
    if (customDataRadio.checked) {
        uploadContainer.classList.remove('d-none');
    } else {
        uploadContainer.classList.add('d-none');
        loadSampleData();
    }
}

/**
 * Load sample customer data
 */
function loadSampleData() {
    // Show loading state in data preview
    dataPreviewBody.innerHTML = `
        <tr>
            <td colspan="10" class="text-center">
                <div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div>
                Lade Beispieldaten...
            </td>
        </tr>
    `;
    
    // Simulated sample data (in real app, this would be an API call)
    setTimeout(() => {
        // Sample data fields
        const fields = [
            'customer_id', 'alter', 'vertragsdauer', 'nutzungsfrequenz', 
            'support_anfragen', 'zahlungsverzoegerungen', 'upgrades', 
            'preiserhohungen', 'fehlermeldungen', 'nps_score'
        ];
        
        // Generate 5 sample customers
        customerData = [];
        for (let i = 1; i <= 5; i++) {
            const customer = {
                customer_id: `CUST-${(1000 + i).toString().padStart(4, '0')}`,
                alter: Math.floor(Math.random() * 50) + 20,
                vertragsdauer: Math.floor(Math.random() * 48) + 1,
                nutzungsfrequenz: Math.floor(Math.random() * 30) + 1,
                support_anfragen: Math.floor(Math.random() * 10),
                zahlungsverzoegerungen: Math.floor(Math.random() * 5),
                upgrades: Math.floor(Math.random() * 3),
                preiserhohungen: Math.floor(Math.random() * 2),
                fehlermeldungen: Math.floor(Math.random() * 15),
                nps_score: Math.floor(Math.random() * 10) + 1
            };
            customerData.push(customer);
        }
        
        // Update data preview
        updateDataPreview(customerData, fields);
    }, 500);
}

/**
 * Handle file upload for custom customer data
 */
function handleFileUpload() {
    // Check if file is selected
    if (!csvUploadInput.files || csvUploadInput.files.length === 0) {
        showError('Bitte wählen Sie eine CSV-Datei aus.');
        return;
    }
    
    const file = csvUploadInput.files[0];
    
    // Check file type
    if (!file.name.endsWith('.csv')) {
        showError('Nur CSV-Dateien werden unterstützt.');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    dataPreviewBody.innerHTML = `
        <tr>
            <td colspan="10" class="text-center">
                <div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div>
                Lade Kundendaten...
            </td>
        </tr>
    `;
    
    // Upload file
    fetch('/api/upload-customer-data', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Fehler beim Hochladen der Datei');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Set customer data
        customerData = data.data;
        
        // Get fields from first item
        const fields = Object.keys(customerData[0]);
        
        // Update data preview
        updateDataPreview(customerData, fields);
    })
    .catch(error => {
        console.error('Error uploading customer data:', error);
        showError('Fehler beim Hochladen der Kundendaten. Bitte versuchen Sie es später erneut.');
    });
}

/**
 * Update the data preview table
 */
function updateDataPreview(data, fields) {
    if (!data || data.length === 0) {
        dataPreviewBody.innerHTML = `
            <tr>
                <td colspan="${fields.length}" class="text-center">Keine Daten verfügbar</td>
            </tr>
        `;
        return;
    }
    
    // Create header
    let headerHtml = '<tr>';
    fields.forEach(field => {
        headerHtml += `<th>${field}</th>`;
    });
    headerHtml += '</tr>';
    dataPreviewHeader.innerHTML = headerHtml;
    
    // Create body (show up to 5 customers)
    let bodyHtml = '';
    const displayCount = Math.min(data.length, 5);
    
    for (let i = 0; i < displayCount; i++) {
        bodyHtml += '<tr>';
        fields.forEach(field => {
            bodyHtml += `<td>${data[i][field]}</td>`;
        });
        bodyHtml += '</tr>';
    }
    
    // Add indicator if more data available
    if (data.length > 5) {
        bodyHtml += `
            <tr>
                <td colspan="${fields.length}" class="text-center text-muted">
                    ... und ${data.length - 5} weitere Einträge
                </td>
            </tr>
        `;
    }
    
    dataPreviewBody.innerHTML = bodyHtml;
}

/**
 * Predict churn for customer data
 */
function predictChurn() {
    // Check if customer data is available
    if (!customerData || customerData.length === 0) {
        showError('Keine Kundendaten verfügbar. Bitte laden Sie zuerst Daten.');
        return;
    }
    
    // Show results container and loading state
    resultsContainer.classList.remove('d-none');
    predictionLoading.classList.remove('d-none');
    resultsSummary.classList.add('d-none');
    detailedResults.classList.add('d-none');
    
    // Disable predict button
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analysiere...';
    
    // Make prediction request
    fetch('/api/predict-churn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            data: customerData
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Fehler bei der Vorhersage');
        }
        return response.json();
    })
    .then(data => {
        // Store prediction results
        predictionResults = data;
        
        // Display results
        displayPredictionResults(data);
    })
    .catch(error => {
        console.error('Error predicting churn:', error);
        showError('Fehler bei der Churn-Vorhersage. Bitte versuchen Sie es später erneut.');
        
        // Hide loading
        predictionLoading.classList.add('d-none');
    })
    .finally(() => {
        // Re-enable predict button
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bi bi-play-fill"></i> Churn-Vorhersage starten';
    });
}

/**
 * Display prediction results
 */
function displayPredictionResults(results) {
    // Hide loading
    predictionLoading.classList.add('d-none');
    
    // Show results components
    resultsSummary.classList.remove('d-none');
    detailedResults.classList.remove('d-none');
    
    // Count risk categories
    let highCount = 0;
    let mediumCount = 0;
    let lowCount = 0;
    
    results.forEach(result => {
        if (result.risk_category.includes('Hoh')) {
            highCount++;
        } else if (result.risk_category.includes('Mittel')) {
            mediumCount++;
        } else {
            lowCount++;
        }
    });
    
    // Update risk counts
    highRiskCount.textContent = highCount;
    mediumRiskCount.textContent = mediumCount;
    lowRiskCount.textContent = lowCount;
    
    // Create results table
    let tableHtml = '';
    
    results.forEach((result, index) => {
        // Determine risk class for highlighting
        let rowClass = '';
        if (result.risk_category.includes('Hoh')) {
            rowClass = 'table-danger';
        } else if (result.risk_category.includes('Mittel')) {
            rowClass = 'table-warning';
        } else {
            rowClass = 'table-success';
        }
        
        // Format risk factors
        const riskFactors = result.top_risk_factors && result.top_risk_factors.length > 0
            ? result.top_risk_factors.join(', ')
            : 'Keine spezifischen Risikofaktoren';
        
        // Format interventions
        const interventions = result.recommended_interventions && result.recommended_interventions.length > 0
            ? result.recommended_interventions.join(', ')
            : 'Standard-Betreuung';
        
        // Create table row
        tableHtml += `
            <tr class="${rowClass}">
                <td>${result.customer_id}</td>
                <td>${createRiskBadge(result.risk_category)}</td>
                <td>${(result.churn_probability * 100).toFixed(1)}%</td>
                <td>${riskFactors}</td>
                <td>${interventions}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-details-btn" data-index="${index}">
                        <i class="bi bi-info-circle"></i> Details
                    </button>
                </td>
            </tr>
        `;
    });
    
    // Update table
    resultsTableBody.innerHTML = tableHtml;
    
    // Add event listeners to detail buttons
    document.querySelectorAll('.view-details-btn').forEach(button => {
        button.addEventListener('click', function() {
            const index = parseInt(this.dataset.index);
            showCustomerDetails(index);
        });
    });
}

/**
 * Show detailed customer information in modal
 */
function showCustomerDetails(index) {
    // Get customer result
    const result = predictionResults[index];
    const customer = customerData[index];
    
    // Set modal title
    customerModalLabel.textContent = `Kundendetails: ${result.customer_id}`;
    
    // Customer info
    let infoHtml = '<dl class="row mb-0">';
    Object.entries(customer).forEach(([key, value]) => {
        // Skip customer_id as it's in the title
        if (key === 'customer_id') return;
        
        infoHtml += `
            <dt class="col-sm-6">${key}</dt>
            <dd class="col-sm-6">${value}</dd>
        `;
    });
    infoHtml += '</dl>';
    customerInfo.innerHTML = infoHtml;
    
    // Risk bar
    const riskPercentage = result.churn_probability * 100;
    customerRiskBar.style.width = `${riskPercentage}%`;
    customerRiskBar.textContent = `${riskPercentage.toFixed(1)}%`;
    
    // Set bar color
    customerRiskBar.className = 'progress-bar';
    if (riskPercentage >= 70) {
        customerRiskBar.classList.add('bg-danger');
    } else if (riskPercentage >= 30) {
        customerRiskBar.classList.add('bg-warning');
    } else {
        customerRiskBar.classList.add('bg-success');
    }
    
    // Risk details
    riskDetails.innerHTML = `
        <p><strong>Risikokategorie:</strong> ${createRiskBadge(result.risk_category)}</p>
        <p><strong>Abwanderungswahrscheinlichkeit:</strong> ${riskPercentage.toFixed(1)}%</p>
    `;
    
    // Risk factors
    let factorsHtml = '';
    if (result.top_risk_factors && result.top_risk_factors.length > 0) {
        result.top_risk_factors.forEach(factor => {
            factorsHtml += `
                <li class="list-group-item">
                    <div class="risk-factor">
                        <i class="bi bi-exclamation-triangle-fill"></i>
                        <span>${factor}</span>
                    </div>
                </li>
            `;
        });
    } else {
        factorsHtml = `
            <li class="list-group-item text-muted">
                Keine spezifischen Risikofaktoren identifiziert
            </li>
        `;
    }
    riskFactorsList.innerHTML = factorsHtml;
    
    // Interventions
    let interventionsHtml = '';
    if (result.recommended_interventions && result.recommended_interventions.length > 0) {
        result.recommended_interventions.forEach(intervention => {
            interventionsHtml += `
                <div class="intervention">
                    <i class="bi bi-check-circle-fill"></i>
                    <div>${intervention}</div>
                </div>
            `;
        });
    } else {
        interventionsHtml = '<p class="text-muted">Standardmäßige Kundenbetreuung empfohlen</p>';
    }
    interventions.innerHTML = interventionsHtml;
    
    // Document references
    let referencesHtml = '';
    if (result.document_references && result.document_references.length > 0) {
        referencesHtml = '<ul class="list-group">';
        result.document_references.forEach(reference => {
            referencesHtml += `
                <li class="list-group-item">
                    <i class="bi bi-file-text me-2"></i>
                    <strong>${reference.filename}</strong>
                    ${reference.section ? `<span class="text-muted"> - Abschnitt: ${reference.section}</span>` : ''}
                </li>
            `;
        });
        referencesHtml += '</ul>';
    } else {
        referencesHtml = '<p class="text-muted">Keine Dokumentreferenzen verfügbar</p>';
    }
    documentReferences.innerHTML = referencesHtml;
    
    // Show modal
    customerModal.show();
}
