import React, { useState } from 'react';
import { 
  Button, 
  Card, 
  CardContent, 
  Typography, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Snackbar, 
  Alert,
  Grid
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import { DocumentAPI } from '../../services/api';

interface DocumentUploadProps {
  onUploadSuccess?: () => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUploadSuccess }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [category, setCategory] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const categories = [
    'Grundlagen', 
    'ML-Modelle', 
    'Kundenanalyse', 
    'Maßnahmen', 
    'Fallstudien'
  ];

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const filesArray = Array.from(event.target.files);
      setSelectedFiles(filesArray);
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || !category) {
      setError('Bitte wählen Sie Dateien und eine Kategorie aus.');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    try {
      await DocumentAPI.uploadDocuments(selectedFiles, category);
      setSuccess(`${selectedFiles.length} Dokument(e) erfolgreich hochgeladen.`);
      setSelectedFiles([]);
      setCategory('');
      
      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } catch (err) {
      setError('Fehler beim Hochladen der Dokumente. Bitte versuchen Sie es erneut.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleCloseSnackbar = () => {
    setError(null);
    setSuccess(null);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Dokumente hochladen
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <div className="upload-container">
              <input
                type="file"
                multiple
                id="document-upload"
                style={{ display: 'none' }}
                onChange={handleFileChange}
                accept=".pdf,.docx,.txt,.md,.html"
              />
              <label htmlFor="document-upload">
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<CloudUpload />}
                  disabled={uploading}
                >
                  Dateien auswählen
                </Button>
              </label>
              
              {selectedFiles.length > 0 && (
                <Typography variant="body2" className="selected-files" sx={{ mt: 1 }}>
                  {selectedFiles.length} Datei(en) ausgewählt:
                  <ul>
                    {selectedFiles.map((file, index) => (
                      <li key={index}>{file.name} ({(file.size / 1024).toFixed(1)} KB)</li>
                    ))}
                  </ul>
                </Typography>
              )}
            </div>
          </Grid>
          
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Kategorie</InputLabel>
              <Select
                value={category}
                onChange={(e) => setCategory(e.target.value as string)}
                disabled={uploading}
              >
                {categories.map((cat) => (
                  <MenuItem key={cat} value={cat}>{cat}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || !category || uploading}
              fullWidth
            >
              {uploading ? 'Wird hochgeladen...' : 'Hochladen und Verarbeiten'}
            </Button>
          </Grid>
        </Grid>
      </CardContent>
      
      <Snackbar open={!!error || !!success} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={error ? "error" : "success"} 
          sx={{ width: '100%' }}
        >
          {error || success}
        </Alert>
      </Snackbar>
    </Card>
  );
};

export default DocumentUpload;
