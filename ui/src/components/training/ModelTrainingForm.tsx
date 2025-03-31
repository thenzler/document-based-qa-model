import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Checkbox, 
  FormControlLabel, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon,
  Divider,
  CircularProgress,
  Paper,
  Alert,
  Snackbar
} from '@mui/material';
import { Article, Check, School } from '@mui/icons-material';
import { DocumentAPI, TrainingAPI } from '../../services/api';
import { useNavigate } from 'react-router-dom';

interface Document {
  id: string;
  filename: string;
  category: string;
  status: string;
}

interface TrainingParameters {
  epochs: number;
  batchSize: number;
  learningRate: number;
  extractQAPairs: boolean;
}

const ModelTrainingForm: React.FC = () => {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [baseModel, setBaseModel] = useState('deepset/gbert-base');
  const [parameters, setParameters] = useState<TrainingParameters>({
    epochs: 3,
    batchSize: 8,
    learningRate: 0.0001,
    extractQAPairs: true
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Verfügbare Modelle
  const availableModels = [
    { id: 'deepset/gbert-base', name: 'German BERT (Deepset)' },
    { id: 'google/flan-t5-base', name: 'Flan-T5 (Google)' },
    { id: 'deutsche-telekom/gbert-large-paraphrase-cosine', name: 'German BERT Large (Telekom)' }
  ];
  
  useEffect(() => {
    // Lade indexierte Dokumente
    const fetchDocuments = async () => {
      setIsLoading(true);
      try {
        const data = await DocumentAPI.getDocuments({ status: 'indexiert' });
        setDocuments(data);
      } catch (err) {
        console.error('Error fetching documents:', err);
        setError('Fehler beim Laden der Dokumente. Bitte versuchen Sie es später erneut.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchDocuments();
  }, []);
  
  const handleDocumentSelect = (docId: string) => {
    setSelectedDocuments(prev => {
      if (prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      } else {
        return [...prev, docId];
      }
    });
  };
  
  const handleSelectAll = () => {
    if (selectedDocuments.length === documents.length) {
      setSelectedDocuments([]);
    } else {
      setSelectedDocuments(documents.map(doc => doc.id));
    }
  };
  
  const handleParameterChange = (param: keyof TrainingParameters, value: any) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  };
  
  const handleCloseSnackbar = () => {
    setError(null);
    setSuccess(null);
  };
  
  const handleStartTraining = async () => {
    if (selectedDocuments.length === 0) {
      setError('Bitte wählen Sie mindestens ein Dokument für das Training aus.');
      return;
    }
    
    setIsStarting(true);
    setError(null);
    
    // Vorbereitung der Trainingsdaten
    const trainingData = {
      baseModel,
      documentIds: selectedDocuments,
      parameters
    };
    
    try {
      // API-Aufruf zum Starten des Trainings
      const response = await TrainingAPI.startTraining(trainingData);
      setSuccess('Training erfolgreich gestartet!');
      
      // Weiterleitung zur Trainingsfortschrittsanzeige
      setTimeout(() => {
        navigate(`/training/${response.training_id}`);
      }, 1500);
    } catch (err) {
      console.error('Error starting training:', err);
      setError('Fehler beim Starten des Trainings. Bitte versuchen Sie es später erneut.');
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="training-form">
      <Card className="model-selection">
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Basismodell
          </Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>Modell auswählen</InputLabel>
            <Select
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value as string)}
              disabled={isStarting}
            >
              {availableModels.map((model) => (
                <MenuItem key={model.id} value={model.id}>{model.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </CardContent>
      </Card>
      
      <Card className="document-selection">
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Dokumente für Training
          </Typography>
          
          {isLoading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
              <CircularProgress />
            </div>
          ) : documents.length === 0 ? (
            <Paper elevation={0} sx={{ p: 2, bgcolor: '#f5f7f9' }}>
              <Typography align="center">
                Keine indexierten Dokumente verfügbar. Bitte laden Sie zuerst Dokumente hoch.
              </Typography>
            </Paper>
          ) : (
            <>
              <div className="document-selection-header">
                <FormControlLabel
                  control={
                    <Checkbox 
                      checked={selectedDocuments.length === documents.length && documents.length > 0} 
                      indeterminate={selectedDocuments.length > 0 && selectedDocuments.length < documents.length} 
                      onChange={handleSelectAll}
                      disabled={isStarting}
                    />
                  }
                  label="Alle auswählen"
                />
                <Typography variant="body2">
                  {selectedDocuments.length} von {documents.length} ausgewählt
                </Typography>
              </div>
              
              <Divider sx={{ my: 1 }} />
              
              <List className="document-list" sx={{ maxHeight: '300px', overflow: 'auto' }}>
                {documents.map((doc) => (
                  <ListItem 
                    key={doc.id}
                    button 
                    onClick={() => handleDocumentSelect(doc.id)}
                    selected={selectedDocuments.includes(doc.id)}
                    disabled={isStarting}
                  >
                    <ListItemIcon>
                      {selectedDocuments.includes(doc.id) ? <Check /> : <Article />}
                    </ListItemIcon>
                    <ListItemText 
                      primary={doc.filename} 
                      secondary={doc.category} 
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
        </CardContent>
      </Card>
      
      <Card className="training-parameters">
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Training-Parameter
          </Typography>
          
          <TextField
            label="Epochen"
            type="number"
            value={parameters.epochs}
            onChange={(e) => handleParameterChange('epochs', parseInt(e.target.value))}
            InputProps={{ inputProps: { min: 1, max: 10 } }}
            margin="normal"
            fullWidth
            disabled={isStarting}
          />
          
          <TextField
            label="Batch-Größe"
            type="number"
            value={parameters.batchSize}
            onChange={(e) => handleParameterChange('batchSize', parseInt(e.target.value))}
            InputProps={{ inputProps: { min: 1, max: 32 } }}
            margin="normal"
            fullWidth
            disabled={isStarting}
          />
          
          <TextField
            label="Lernrate"
            type="number"
            value={parameters.learningRate}
            onChange={(e) => handleParameterChange('learningRate', parseFloat(e.target.value))}
            InputProps={{ inputProps: { min: 0.00001, max: 0.01, step: 0.00001 } }}
            margin="normal"
            fullWidth
            disabled={isStarting}
          />
          
          <FormControlLabel
            control={
              <Checkbox 
                checked={parameters.extractQAPairs} 
                onChange={(e) => handleParameterChange('extractQAPairs', e.target.checked)}
                disabled={isStarting}
              />
            }
            label="QA-Paare automatisch aus Dokumenten extrahieren"
          />
        </CardContent>
      </Card>
      
      <Button
        variant="contained"
        color="primary"
        size="large"
        onClick={handleStartTraining}
        disabled={selectedDocuments.length === 0 || isStarting || isLoading}
        className="start-training-button"
        fullWidth
        startIcon={isStarting ? <CircularProgress size={20} color="inherit" /> : <School />}
        sx={{ mt: 2, py: 1.5 }}
      >
        {isStarting ? 'Training wird gestartet...' : 'Training starten'}
      </Button>
      
      <Snackbar open={!!error || !!success} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={error ? "error" : "success"} 
          sx={{ width: '100%' }}
        >
          {error || success}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default ModelTrainingForm;