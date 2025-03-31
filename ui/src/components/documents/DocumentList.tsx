import React, { useEffect, useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper, 
  Chip, 
  IconButton, 
  TextField, 
  InputAdornment,
  Typography,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Button,
  CircularProgress
} from '@mui/material';
import { Search, MoreVert, Delete, Visibility, Refresh } from '@mui/icons-material';
import { DocumentAPI } from '../../services/api';

interface Document {
  id: string;
  filename: string;
  fileType: string;
  category: string;
  size: number;
  uploadDate: string;
  status: 'indexiert' | 'verarbeitung' | 'fehler';
}

const DocumentList: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null);
  
  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await DocumentAPI.getDocuments();
      setDocuments(data);
      setFilteredDocuments(data);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError('Fehler beim Laden der Dokumente. Bitte versuchen Sie es später erneut.');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchDocuments();
  }, []);
  
  useEffect(() => {
    // Filtern der Dokumente basierend auf dem Suchbegriff
    if (!searchTerm.trim()) {
      setFilteredDocuments(documents);
      return;
    }
    
    const searchTermLower = searchTerm.toLowerCase();
    const filtered = documents.filter(doc => 
      doc.filename.toLowerCase().includes(searchTermLower) ||
      doc.category.toLowerCase().includes(searchTermLower) ||
      doc.fileType.toLowerCase().includes(searchTermLower)
    );
    
    setFilteredDocuments(filtered);
  }, [searchTerm, documents]);
  
  const handleDeleteClick = (document: Document) => {
    setDocumentToDelete(document);
    setDeleteDialogOpen(true);
  };
  
  const handleDeleteConfirm = async () => {
    if (!documentToDelete) return;
    
    try {
      await DocumentAPI.deleteDocument(documentToDelete.id);
      setDocuments(prev => prev.filter(doc => doc.id !== documentToDelete.id));
      setDeleteDialogOpen(false);
      setDocumentToDelete(null);
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Fehler beim Löschen des Dokuments.');
    }
  };
  
  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setDocumentToDelete(null);
  };
  
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };
  
  const StatusChip = ({ status }: { status: Document['status'] }) => {
    const statusProps = {
      indexiert: { color: 'success', label: 'Indexiert' },
      verarbeitung: { color: 'primary', label: 'Verarbeitung' },
      fehler: { color: 'error', label: 'Fehler' }
    };
    
    const { color, label } = statusProps[status];
    
    return (
      <Chip 
        label={label} 
        color={color as 'success' | 'primary' | 'error'} 
        size="small"
      />
    );
  };
  
  if (loading && documents.length === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
        <CircularProgress />
      </div>
    );
  }
  
  if (error && documents.length === 0) {
    return (
      <Paper sx={{ p: 3, bgcolor: '#ffebee' }}>
        <Typography color="error">{error}</Typography>
        <Button
          variant="outlined"
          onClick={fetchDocuments}
          startIcon={<Refresh />}
          sx={{ mt: 2 }}
        >
          Erneut versuchen
        </Button>
      </Paper>
    );
  }

  return (
    <div className="document-list">
      <div className="search-bar">
        <TextField
          variant="outlined"
          placeholder="Suchen..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          fullWidth
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            )
          }}
        />
      </div>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Typ</TableCell>
              <TableCell>Kategorie</TableCell>
              <TableCell>Größe</TableCell>
              <TableCell>Hochgeladen am</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Aktionen</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredDocuments.map((doc) => (
              <TableRow key={doc.id}>
                <TableCell>{doc.filename}</TableCell>
                <TableCell>{doc.fileType.toUpperCase()}</TableCell>
                <TableCell>{doc.category}</TableCell>
                <TableCell>{formatFileSize(doc.size)}</TableCell>
                <TableCell>{new Date(doc.uploadDate).toLocaleDateString('de-DE')}</TableCell>
                <TableCell>
                  <StatusChip status={doc.status} />
                </TableCell>
                <TableCell>
                  <IconButton size="small" title="Anzeigen">
                    <Visibility fontSize="small" />
                  </IconButton>
                  <IconButton size="small" title="Löschen" onClick={() => handleDeleteClick(doc)}>
                    <Delete fontSize="small" />
                  </IconButton>
                  <IconButton size="small" title="Mehr">
                    <MoreVert fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            
            {filteredDocuments.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography sx={{ py: 2 }}>
                    {documents.length === 0 
                      ? 'Keine Dokumente vorhanden. Laden Sie Dokumente hoch, um zu beginnen.' 
                      : 'Keine Dokumente gefunden, die dem Suchbegriff entsprechen.'}
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Dokument löschen?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Sind Sie sicher, dass Sie das Dokument "{documentToDelete?.filename}" löschen möchten? 
            Diese Aktion kann nicht rückgängig gemacht werden.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>Abbrechen</Button>
          <Button onClick={handleDeleteConfirm} color="error">
            Löschen
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default DocumentList;
