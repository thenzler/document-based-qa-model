/* 
 * JavaScript for the Question-Answering page
 */

// DOM elements
const questionInput = document.getElementById('question-input');
const askButton = document.getElementById('ask-button');
const historyList = document.getElementById('history-list');
const resultsContainer = document.getElementById('results-container');
const resultQuestion = document.getElementById('result-question');
const answerLoading = document.getElementById('answer-loading');
const answerContainer = document.getElementById('answer-container');
const answerText = document.getElementById('answer-text');
const sourcesListContainer = document.getElementById('sources-list');
const processingTimeElement = document.getElementById('processing-time');
const queryVariationsContainer = document.getElementById('query-variations-container');
const queryVariations = document.getElementById('query-variations');

// Advanced options
const useGenerationCheckbox = document.getElementById('use-generation');
const topKSlider = document.getElementById('top-k');
const topKValue = document.getElementById('top-k-value');

// Question history
let questionHistory = [];

// Initialize the QA page
document.addEventListener('DOMContentLoaded', function() {
    // Load question history from local storage
    loadQuestionHistory();
    
    // Set up event listeners
    askButton.addEventListener('click', handleAskQuestion);
    questionInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            handleAskQuestion();
        }
    });
    
    // Set up advanced options
    topKSlider.addEventListener('input', function() {
        topKValue.textContent = this.value;
    });
    
    // Check if system is ready
    checkSystemStatus();
});

/**
 * Handle asking a question
 */
function handleAskQuestion() {
    const question = questionInput.value.trim();
    
    // Validate question
    if (!question) {
        // Highlight input
        questionInput.classList.add('is-invalid');
        setTimeout(() => questionInput.classList.remove('is-invalid'), 2000);
        return;
    }
    
    // Get options
    const useGeneration = useGenerationCheckbox.checked;
    const topK = parseInt(topKSlider.value);
    
    // Show loading state
    showLoading(question);
    
    // Add to history
    addToHistory(question);
    
    // Clear input
    questionInput.value = '';
    
    // Ask question via API
    askQuestion(question, useGeneration, topK);
}

/**
 * Show loading state
 */
function showLoading(question) {
    // Update question
    resultQuestion.textContent = question;
    
    // Show results container
    resultsContainer.classList.remove('d-none');
    
    // Show loading and hide answer
    answerLoading.classList.remove('d-none');
    answerContainer.classList.add('d-none');
}

/**
 * Ask a question via the API
 */
function askQuestion(question, useGeneration, topK) {
    fetch('/api/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: question,
            use_generation: useGeneration,
            top_k: topK
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Fehler bei der Anfrage');
        }
        return response.json();
    })
    .then(data => {
        displayAnswer(data);
    })
    .catch(error => {
        console.error('Error asking question:', error);
        showError('Fehler beim Stellen der Frage. Bitte versuchen Sie es später erneut.');
        
        // Hide loading
        answerLoading.classList.add('d-none');
    });
}

/**
 * Display the answer and sources
 */
function displayAnswer(data) {
    // Hide loading and show answer
    answerLoading.classList.add('d-none');
    answerContainer.classList.remove('d-none');
    
    // Set answer text
    answerText.innerHTML = formatText(data.answer);
    
    // Set processing time if available
    if (data.processing_time) {
        processingTimeElement.textContent = `${data.processing_time.toFixed(2)}s`;
    } else {
        processingTimeElement.textContent = '';
    }
    
    // Display sources
    displaySources(data.sources);
    
    // Display query variations if available
    if (data.query_variations && data.query_variations.length > 0) {
        displayQueryVariations(data.query_variations);
    } else {
        queryVariationsContainer.classList.add('d-none');
    }
}

/**
 * Display the sources used for the answer
 */
function displaySources(sources) {
    // Clear sources list
    sourcesListContainer.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        sourcesListContainer.innerHTML = '<p>Keine Quellen verfügbar.</p>';
        return;
    }
    
    // Create source elements
    sources.forEach(source => {
        const sourceElement = document.createElement('div');
        sourceElement.className = 'source-item';
        
        // Source header with filename and score
        const sourceHeader = document.createElement('div');
        sourceHeader.className = 'source-header';
        
        // Filename and section
        let sourceTitle = `<strong>[${source.id}] ${sanitizeHTML(source.filename)}</strong>`;
        if (source.section) {
            sourceTitle += ` - <span class="text-muted">Abschnitt: ${sanitizeHTML(source.section)}</span>`;
        }
        
        // Score badge
        const scoreBadge = `<span class="badge bg-primary">Relevanz: ${source.score}</span>`;
        
        sourceHeader.innerHTML = `<div>${sourceTitle}</div><div>${scoreBadge}</div>`;
        sourceElement.appendChild(sourceHeader);
        
        // Add evidence/matching sentences if available
        if (source.evidence && source.evidence.length > 0) {
            const evidenceContainer = document.createElement('div');
            evidenceContainer.className = 'source-evidence';
            
            source.evidence.forEach(sentence => {
                const sentenceElement = document.createElement('p');
                sentenceElement.className = 'mb-0';
                sentenceElement.textContent = sentence;
                evidenceContainer.appendChild(sentenceElement);
            });
            
            sourceElement.appendChild(evidenceContainer);
        }
        
        sourcesListContainer.appendChild(sourceElement);
    });
}

/**
 * Display query variations used for retrieving answers
 */
function displayQueryVariations(variations) {
    // Clear variations
    queryVariations.innerHTML = '';
    
    // Show container
    queryVariationsContainer.classList.remove('d-none');
    
    // Add variations
    variations.forEach(variation => {
        const variationElement = document.createElement('span');
        variationElement.className = 'query-variation';
        variationElement.textContent = variation;
        queryVariations.appendChild(variationElement);
    });
}

/**
 * Add a question to history
 */
function addToHistory(question) {
    // Check if question is already in history
    const existingIndex = questionHistory.findIndex(q => q.toLowerCase() === question.toLowerCase());
    
    if (existingIndex !== -1) {
        // Remove from current position
        questionHistory.splice(existingIndex, 1);
    }
    
    // Add to beginning of history
    questionHistory.unshift(question);
    
    // Limit history to 10 questions
    if (questionHistory.length > 10) {
        questionHistory.pop();
    }
    
    // Save to local storage
    saveQuestionHistory();
    
    // Update history display
    updateHistoryDisplay();
}

/**
 * Update the history display
 */
function updateHistoryDisplay() {
    // Clear history list
    historyList.innerHTML = '';
    
    // No history
    if (questionHistory.length === 0) {
        const noHistoryItem = document.createElement('li');
        noHistoryItem.className = 'list-group-item text-muted';
        noHistoryItem.textContent = 'Keine Fragen im Verlauf';
        historyList.appendChild(noHistoryItem);
        return;
    }
    
    // Create history items
    questionHistory.forEach((question, index) => {
        const historyItem = document.createElement('li');
        historyItem.className = 'list-group-item d-flex justify-content-between align-items-center';
        historyItem.innerHTML = `
            <span>${sanitizeHTML(question)}</span>
            <button class="btn btn-sm btn-outline-secondary history-btn" data-question="${sanitizeHTML(question)}">
                <i class="bi bi-arrow-clockwise"></i>
            </button>
        `;
        historyList.appendChild(historyItem);
        
        // Add event listener to re-ask this question
        historyItem.querySelector('.history-btn').addEventListener('click', function() {
            questionInput.value = this.dataset.question;
            handleAskQuestion();
        });
    });
}

/**
 * Save question history to local storage
 */
function saveQuestionHistory() {
    localStorage.setItem('qaQuestionHistory', JSON.stringify(questionHistory));
}

/**
 * Load question history from local storage
 */
function loadQuestionHistory() {
    const savedHistory = localStorage.getItem('qaQuestionHistory');
    
    if (savedHistory) {
        try {
            questionHistory = JSON.parse(savedHistory);
            updateHistoryDisplay();
        } catch (error) {
            console.error('Error loading question history:', error);
            // Reset history if error
            questionHistory = [];
        }
    } else {
        questionHistory = [];
    }
}

/**
 * Format text for display (convert URLs, etc.)
 */
function formatText(text) {
    if (!text) return '';
    
    // Convert URLs to links
    text = text.replace(
        /(https?:\/\/[^\s]+)/g, 
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    
    // Convert line breaks to <br>
    text = text.replace(/\n/g, '<br>');
    
    return text;
}
