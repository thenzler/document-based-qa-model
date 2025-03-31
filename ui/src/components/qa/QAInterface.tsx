import React, { useState, useEffect } from 'react';
import { 
  Button, 
  Card, 
  CardContent, 
  TextField, 
  Typography, 
  Chip, 
  CircularProgress,
  FormControlLabel,
  Switch, 
  Paper, 
  Divider,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Box
} from '@mui/material';
import { Search, ThumbUp, ThumbDown, Share, Print } from '@mui/icons-material';
import { QAAPI } from '../../services/api';

interface Source {
  source: string;
  section: string;
  filename: string;
  relevanceScore: number;
  matchingSentences: string[];
}

interface QAResponse {
  answer: string;
  sources: Source[];
  processingTime: number;
}

const QAInterface: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [useGeneration, setUseGeneration] = useState(true);
  const [showSources, setShowSources] = useState(true);
  const [showExplanation, setShowExplanation] = useState(false);
  const [recentQuestions, setRecentQuestions] = useState<string[]>([]);
  
  useEffect(() => {
    // Lade kürzlich gestellte Fragen
    const fetchRecentQuestions = async () => {
      try {
        const data = await QAAPI.getRecentQuestions();
        setRecentQuestions(data);
      } catch (error) {
        console.error('Fehler beim Laden der kürzlichen Fragen:', error);
      }
    };
    
    fetchRecentQuestions();
  }, []);
  
  const handleQuestionSubmit = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setErrorMessage('');
    
    try {
      const data = await QAAPI.answerQuestion(question, useGeneration);
      setResponse(data);
      
      // Füge die Frage zu den kürzlich gestellten Fragen hinzu
      setRecentQuestions(prev => [question, ...prev.slice(0, 4)]);
    } catch (error) {
      console.error('Fehler bei der Anfrage:', error);
      setErrorMessage('Fehler beim Senden der Anfrage.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleRecentQuestionClick = (q: string) => {
    setQuestion(q);
  };
  
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuestionSubmit();
    }
  };

  return (
    <div className="qa-interface">
      <div className="main-qa-area">
        <Card className="question-card">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Stellen Sie eine Frage zu Ihren Dokumenten
            </Typography>
            <TextField
              variant="outlined"
              placeholder="Ihre Frage..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              fullWidth
              multiline
              rows={2}
              onKeyPress={handleKeyPress}
              InputProps={{
                endAdornment: (
                  <Button
                    variant="contained"
                    onClick={handleQuestionSubmit}
                    disabled={loading || !question.trim()}
                    startIcon={<Search />}
                  >
                    Fragen
                  </Button>
                ),
              }}
            />
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useGeneration}
                    onChange={(e) => setUseGeneration(e.target.checked)}
                  />
                }
                label="KI-generierte Antwort"
              />
            </Box>
          </CardContent>
        </Card>
        
        {loading && (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <CircularProgress />
            <Typography variant="body1" style={{ marginTop: 16 }}>
              Suche in Dokumenten und generiere Antwort...
            </Typography>
          </div>
        )}
        
        {errorMessage && (
          <Paper style={{ padding: 16, marginTop: 16, backgroundColor: '#ffebee' }}>
            <Typography color="error">{errorMessage}</Typography>
          </Paper>
        )}
        
        {response && !loading && (
          <Card className="answer-card">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Antwort
              </Typography>
              <Typography variant="body1" paragraph style={{ whiteSpace: 'pre-line' }}>
                {response.answer}
              </Typography>
              
              {showSources && response.sources && response.sources.length > 0 && (
                <div className="sources-section">
                  <Typography variant="subtitle1" gutterBottom>
                    <strong>Quellen:</strong>
                  </Typography>
                  {response.sources.map((source, index) => (
                    <Paper key={index} elevation={1} style={{ padding: 12, marginBottom: 8 }}>
                      <Typography variant="subtitle2">
                        {source.filename}
                        {source.section && ` - Abschnitt: ${source.section}`}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Relevanz: {(source.relevanceScore * 100).toFixed(1)}%
                      </Typography>
                      {source.matchingSentences && source.matchingSentences.length > 0 && (
                        <div style={{ marginTop: 8 }}>
                          <Typography variant="body2" style={{ fontStyle: 'italic' }}>
                            "{source.matchingSentences[0]}"
                          </Typography>
                        </div>
                      )}
                    </Paper>
                  ))}
                </div>
              )}
              
              <div className="answer-footer">
                <div>
                  <Typography variant="body2" color="textSecondary">
                    Verarbeitungszeit: {response.processingTime.toFixed(2)} Sekunden
                  </Typography>
                </div>
                <div>
                  <IconButton size="small" title="Hilfreich">
                    <ThumbUp fontSize="small" />
                  </IconButton>
                  <IconButton size="small" title="Nicht hilfreich">
                    <ThumbDown fontSize="small" />
                  </IconButton>
                  <IconButton size="small" title="Teilen">
                    <Share fontSize="small" />
                  </IconButton>
                  <IconButton size="small" title="Drucken">
                    <Print fontSize="small" />
                  </IconButton>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      
      <div className="qa-sidebar">
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Kürzlich gestellte Fragen
            </Typography>
            <List>
              {recentQuestions.length > 0 ? (
                recentQuestions.map((q, index) => (
                  <ListItem 
                    key={index} 
                    button 
                    onClick={() => handleRecentQuestionClick(q)}
                    style={{ padding: '4px 0' }}
                  >
                    <ListItemText 
                      primary={q} 
                      primaryTypographyProps={{ style: { fontSize: '0.9rem' } }} 
                    />
                  </ListItem>
                ))
              ) : (
                <ListItem>
                  <ListItemText 
                    primary="Keine kürzlichen Fragen" 
                    primaryTypographyProps={{ color: 'textSecondary' }} 
                  />
                </ListItem>
              )}
            </List>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <Typography variant="h6" gutterBottom>
              Einstellungen
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={showSources}
                  onChange={(e) => setShowSources(e.target.checked)}
                />
              }
              label="Quellen anzeigen"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showExplanation}
                  onChange={(e) => setShowExplanation(e.target.checked)}
                />
              }
              label="Erklärung anzeigen"
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default QAInterface;
