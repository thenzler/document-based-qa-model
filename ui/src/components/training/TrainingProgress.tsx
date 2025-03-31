import React, { useEffect, useState } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  LinearProgress, 
  CircularProgress,
  Chip,
  Paper,
  Button,
  Box
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';
import { Warning, CheckCircle, ErrorOutline, Refresh } from '@mui/icons-material';
import { TrainingAPI } from '../../services/api';
import { useParams, useNavigate } from 'react-router-dom';

interface TrainingMetrics {
  epoch: number;
  batch: number;
  loss: number;
  accuracy: number;
  timestamp: string;
}

interface TrainingStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  metrics: TrainingMetrics[];
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  failedAt?: string;
  error?: string;
  modelId?: string;
}

const statusColors = {
  pending: '#f39c12',
  running: '#3498db',
  completed: '#2ecc71',
  failed: '#e74c3c'
};

const statusIcons = {
  pending: null,
  running: null,
  completed: <CheckCircle />,
  failed: <ErrorOutline />
};

const TrainingProgress: React.FC = () => {
  const { trainingId } = useParams<{ trainingId: string }>();
  const navigate = useNavigate();
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentMetric, setCurrentMetric] = useState<TrainingMetrics | null>(null);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  
  // Funktion zum Laden des Trainingsstatus
  const fetchTrainingStatus = async () => {
    if (!trainingId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await TrainingAPI.getTrainingStatus(trainingId);
      setStatus(data);
      
      if (data.metrics && data.metrics.length > 0) {
        setCurrentMetric(data.metrics[data.metrics.length - 1]);
      }
    } catch (err) {
      console.error('Error fetching training status:', err);
      setError('Fehler beim Laden des Trainingsstatus. Bitte versuchen Sie es später erneut.');
    } finally {
      setLoading(false);
    }
  };
  
  // Beim Laden der Komponente den Trainingsstatus abrufen
  useEffect(() => {
    if (!trainingId) {
      navigate('/training');
      return;
    }
    
    fetchTrainingStatus();
    
    // WebSocket-Verbindung für Echtzeit-Updates einrichten
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}/ws/training/${trainingId}`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'status') {
        setStatus(prev => prev ? { ...prev, status: data.status, progress: data.progress } : null);
      } else if (data.type === 'metric') {
        const newMetric = data.metric;
        setCurrentMetric(newMetric);
        setStatus(prev => {
          if (!prev) return null;
          
          const updatedMetrics = [...prev.metrics, newMetric];
          return { ...prev, metrics: updatedMetrics };
        });
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
      
      // Bei geschlossener WebSocket-Verbindung den Status periodisch abrufen
      const intervalId = setInterval(() => {
        if (status?.status === 'completed' || status?.status === 'failed') {
          clearInterval(intervalId);
        } else {
          fetchTrainingStatus();
        }
      }, 5000);
      
      // Cleanup beim Unmount
      return () => clearInterval(intervalId);
    };
    
    setWebsocket(ws);
    
    // Cleanup beim Unmount
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [trainingId]);
  
  // Berechnung der geschätzten Restzeit
  const getEstimatedTimeRemaining = (): string => {
    if (!status || !currentMetric || status.progress >= 100) {
      return '--:--:--';
    }
    
    const now = new Date().getTime();
    const startTime = new Date(status.startedAt || status.createdAt).getTime();
    const elapsedMs = now - startTime;
    
    if (status.progress <= 0 || elapsedMs <= 0) {
      return '--:--:--';
    }
    
    // Berechne die geschätzte Gesamtzeit basierend auf dem aktuellen Fortschritt
    const estimatedTotalMs = (elapsedMs / status.progress) * 100;
    const remainingMs = estimatedTotalMs - elapsedMs;
    
    // Formatiere die Restzeit
    const hours = Math.floor(remainingMs / (1000 * 60 * 60));
    const minutes = Math.floor((remainingMs % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((remainingMs % (1000 * 60)) / 1000);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  if (loading && !status) {
    return (
      <div className="loading-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Lade Trainingsdaten...</Typography>
      </div>
    );
  }
  
  if (error && !status) {
    return (
      <Paper sx={{ p: 3, bgcolor: '#ffebee' }}>
        <Typography color="error">{error}</Typography>
        <Button
          variant="outlined"
          onClick={fetchTrainingStatus}
          startIcon={<Refresh />}
          sx={{ mt: 2 }}
        >
          Erneut versuchen
        </Button>
      </Paper>
    );
  }
  
  if (!status) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography>Kein Trainingsjob gefunden. Bitte starten Sie einen neuen Trainingsprozess.</Typography>
        <Button
          variant="outlined"
          onClick={() => navigate('/training')}
          sx={{ mt: 2 }}
        >
          Zurück zum Training
        </Button>
      </Paper>
    );
  }
  
  return (
    <div className="training-progress">
      <Card className="progress-overview">
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trainingsfortschritt
          </Typography>
          
          <div className="progress-circle" style={{ position: 'relative', textAlign: 'center', margin: '2rem auto' }}>
            <Box sx={{ position: 'relative', display: 'inline-flex' }}>
              <CircularProgress 
                variant="determinate" 
                value={status.progress} 
                size={120} 
                thickness={5}
                sx={{ color: statusColors[status.status] }}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: 'absolute',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography variant="h4" color="text.secondary">
                  {Math.round(status.progress)}%
                </Typography>
              </Box>
            </Box>
          </div>
          
          <div className="training-status">
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Chip 
                label={status.status.charAt(0).toUpperCase() + status.status.slice(1)} 
                color={
                  status.status === 'completed' ? 'success' : 
                  status.status === 'failed' ? 'error' : 
                  status.status === 'running' ? 'primary' : 
                  'warning'
                }
                icon={statusIcons[status.status] || undefined}
              />
            </Box>
            
            {currentMetric && status.status === 'running' && (
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2">
                  Epoch {currentMetric.epoch} / {3}, 
                  Batch {currentMetric.batch}
                </Typography>
                <Typography variant="body2">
                  Aktuelle Loss: {currentMetric.loss.toFixed(4)}
                </Typography>
                <Typography variant="body2">
                  Aktuelle Genauigkeit: {(currentMetric.accuracy * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Geschätzte Restzeit: {getEstimatedTimeRemaining()}
                </Typography>
              </Box>
            )}
            
            {status.error && (
              <Paper sx={{ p: 2, mt: 2, bgcolor: '#ffebee' }}>
                <Typography color="error" variant="body2">
                  Fehler: {status.error}
                </Typography>
              </Paper>
            )}
            
            {status.status === 'completed' && (
              <Box sx={{ textAlign: 'center', mt: 2 }}>
                <Typography variant="body2">
                  Training abgeschlossen am {new Date(status.completedAt || '').toLocaleString('de-DE')}
                </Typography>
                <Button 
                  variant="contained" 
                  color="primary"
                  sx={{ mt: 1 }}
                  onClick={() => navigate(`/models/${status.modelId}`)}
                >
                  Zum Modell
                </Button>
              </Box>
            )}
          </div>
        </CardContent>
      </Card>
      
      {status.metrics && status.metrics.length > 0 && (
        <Card className="metrics-chart">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Trainingsmetriken
            </Typography>
            
            <div className="chart-container">
              <Typography variant="subtitle1">Genauigkeit / Batch</Typography>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={status.metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="batch" 
                      label={{ value: 'Batch', position: 'insideBottom', offset: -5 }} 
                    />
                    <YAxis 
                      domain={[0.6, 1.0]} 
                      label={{ value: 'Genauigkeit', angle: -90, position: 'insideLeft' }} 
                    />
                    <Tooltip 
                      formatter={(value) => [(value as number).toFixed(4), 'Genauigkeit']} 
                      labelFormatter={(label) => `Batch ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#3498db" 
                      dot={false} 
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <Typography variant="subtitle1" sx={{ mt: 4 }}>Loss / Batch</Typography>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={status.metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="batch" 
                      label={{ value: 'Batch', position: 'insideBottom', offset: -5 }} 
                    />
                    <YAxis 
                      label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} 
                    />
                    <Tooltip 
                      formatter={(value) => [(value as number).toFixed(4), 'Loss']} 
                      labelFormatter={(label) => `Batch ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="loss" 
                      stroke="#e74c3c" 
                      dot={false} 
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Button 
          variant="outlined" 
          onClick={() => navigate('/training')}
        >
          Zurück zum Training
        </Button>
      </Box>
    </div>
  );
};

export default TrainingProgress;
