import React from 'react';
import { Typography, Container, Paper } from '@mui/material';
import QAInterface from '../components/qa/QAInterface';

const QAPage: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Dokument-basierte Frage-Antwort-Schnittstelle
        </Typography>
        <Typography variant="body1" color="textSecondary" paragraph>
          Stellen Sie Fragen zu Ihren hochgeladenen Dokumenten und erhalten Sie präzise Antworten basierend auf deren Inhalt.
        </Typography>
      </Paper>
      
      <QAInterface />
    </Container>
  );
};

export default QAPage;
