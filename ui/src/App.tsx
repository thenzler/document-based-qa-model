import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import DocumentsPage from './pages/DocumentsPage';
import TrainingPage from './pages/TrainingPage';
import ModelsPage from './pages/ModelsPage';
import QAPage from './pages/QAPage';
import './styles/main.css';

// Definieren des Theme für Material UI
const theme = createTheme({
  palette: {
    primary: {
      main: '#2c3e50',
    },
    secondary: {
      main: '#3498db',
    },
    error: {
      main: '#e74c3c',
    },
    warning: {
      main: '#f39c12',
    },
    success: {
      main: '#2ecc71',
    },
    background: {
      default: '#f5f7f9',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Segoe UI", Arial, sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 5,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 5,
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
        },
      },
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/training/:trainingId" element={<TrainingPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/qa" element={<QAPage />} />
          </Routes>
        </MainLayout>
      </Router>
    </ThemeProvider>
  );
};

export default App;
