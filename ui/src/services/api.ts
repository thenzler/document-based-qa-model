import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

// Zentrale Axios-Instanz mit Basiskonfiguration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Document API
export const DocumentAPI = {
  getDocuments: async (params = {}) => {
    try {
      const response = await api.get('/api/documents', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching documents:', error);
      throw error;
    }
  },
  
  uploadDocuments: async (files: File[], category: string) => {
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('category', category);
      
      const response = await api.post('/api/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error uploading documents:', error);
      throw error;
    }
  },
  
  deleteDocument: async (documentId: string) => {
    try {
      const response = await api.delete(`/api/documents/${documentId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  },
  
  processDocuments: async (forceReprocess = false) => {
    try {
      const response = await api.post('/api/documents/process', { forceReprocess });
      return response.data;
    } catch (error) {
      console.error('Error processing documents:', error);
      throw error;
    }
  }
};

// Training API
export const TrainingAPI = {
  startTraining: async (trainingData: any) => {
    try {
      const response = await api.post('/api/training/start', trainingData);
      return response.data;
    } catch (error) {
      console.error('Error starting training:', error);
      throw error;
    }
  },
  
  getTrainingStatus: async (trainingId: string) => {
    try {
      const response = await api.get(`/api/training/${trainingId}/status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching training status:', error);
      throw error;
    }
  },
  
  getTrainingHistory: async (limit = 10, offset = 0) => {
    try {
      const response = await api.get('/api/training/history', {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching training history:', error);
      throw error;
    }
  }
};

// QA API
export const QAAPI = {
  answerQuestion: async (question: string, useGeneration = true, topK = 5) => {
    try {
      const response = await api.post('/api/qa/answer', {
        question,
        useGeneration,
        topK
      });
      return response.data;
    } catch (error) {
      console.error('Error getting answer:', error);
      throw error;
    }
  },
  
  getRecentQuestions: async (limit = 5) => {
    try {
      const response = await api.get('/api/qa/recent-questions', {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching recent questions:', error);
      throw error;
    }
  },
  
  createConversation: async () => {
    try {
      const response = await api.post('/api/qa/conversations');
      return response.data;
    } catch (error) {
      console.error('Error creating conversation:', error);
      throw error;
    }
  },
  
  getConversations: async (limit = 10, offset = 0) => {
    try {
      const response = await api.get('/api/qa/conversations', {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching conversations:', error);
      throw error;
    }
  },
  
  addMessageToConversation: async (conversationId: string, text: string) => {
    try {
      const response = await api.post(`/api/qa/conversations/${conversationId}/messages`, {
        text
      });
      return response.data;
    } catch (error) {
      console.error('Error adding message to conversation:', error);
      throw error;
    }
  }
};

export default {
  DocumentAPI,
  TrainingAPI,
  QAAPI
};
