import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import axios from 'axios';
import toast from 'react-hot-toast';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Context
const MedicalContext = createContext();

// Initial state
const initialState = {
  systemInfo: null,
  chatHistory: [],
  currentChat: null,
  isProcessing: false,
  error: null,
  models: null,
  currentModel: null,
  voiceSettings: {
    whisperModel: 'base',
    ttsLanguage: 'en',
    ttsSpeed: 1.0,
  },
  medicalStats: {
    totalChats: 0,
    totalVoiceChats: 0,
    emergencyDetections: 0,
    averageResponseTime: 0,
  },
};

// Action types
const MEDICAL_ACTIONS = {
  SET_SYSTEM_INFO: 'SET_SYSTEM_INFO',
  SET_MODELS: 'SET_MODELS',
  SET_CURRENT_MODEL: 'SET_CURRENT_MODEL',
  ADD_CHAT_MESSAGE: 'ADD_CHAT_MESSAGE',
  SET_CURRENT_CHAT: 'SET_CURRENT_CHAT',
  SET_PROCESSING: 'SET_PROCESSING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  UPDATE_VOICE_SETTINGS: 'UPDATE_VOICE_SETTINGS',
  UPDATE_MEDICAL_STATS: 'UPDATE_MEDICAL_STATS',
  CLEAR_CHAT_HISTORY: 'CLEAR_CHAT_HISTORY',
  LOAD_CHAT_HISTORY: 'LOAD_CHAT_HISTORY',
};

// Reducer
const medicalReducer = (state, action) => {
  switch (action.type) {
    case MEDICAL_ACTIONS.SET_SYSTEM_INFO:
      return {
        ...state,
        systemInfo: action.payload,
      };

    case MEDICAL_ACTIONS.SET_MODELS:
      return {
        ...state,
        models: action.payload,
      };

    case MEDICAL_ACTIONS.SET_CURRENT_MODEL:
      return {
        ...state,
        currentModel: action.payload,
      };

    case MEDICAL_ACTIONS.ADD_CHAT_MESSAGE:
      const updatedHistory = [...state.chatHistory, action.payload];
      return {
        ...state,
        chatHistory: updatedHistory,
        currentChat: action.payload,
      };

    case MEDICAL_ACTIONS.SET_CURRENT_CHAT:
      return {
        ...state,
        currentChat: action.payload,
      };

    case MEDICAL_ACTIONS.SET_PROCESSING:
      return {
        ...state,
        isProcessing: action.payload,
      };

    case MEDICAL_ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        isProcessing: false,
      };

    case MEDICAL_ACTIONS.CLEAR_ERROR:
      return {
        ...state,
        error: null,
      };

    case MEDICAL_ACTIONS.UPDATE_VOICE_SETTINGS:
      return {
        ...state,
        voiceSettings: {
          ...state.voiceSettings,
          ...action.payload,
        },
      };

    case MEDICAL_ACTIONS.UPDATE_MEDICAL_STATS:
      return {
        ...state,
        medicalStats: {
          ...state.medicalStats,
          ...action.payload,
        },
      };

    case MEDICAL_ACTIONS.CLEAR_CHAT_HISTORY:
      return {
        ...state,
        chatHistory: [],
        currentChat: null,
      };

    case MEDICAL_ACTIONS.LOAD_CHAT_HISTORY:
      return {
        ...state,
        chatHistory: action.payload,
      };

    default:
      return state;
  }
};

// Provider component
export const MedicalProvider = ({ children }) => {
  const [state, dispatch] = useReducer(medicalReducer, initialState);
  const queryClient = useQueryClient();

  // Load system info
  const { data: systemInfo } = useQuery(
    'systemInfo',
    async () => {
      const response = await api.get('/models/system-info');
      return response.data;
    },
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      onSuccess: (data) => {
        dispatch({ type: MEDICAL_ACTIONS.SET_SYSTEM_INFO, payload: data });
      },
      onError: (error) => {
        console.error('Failed to load system info:', error);
        dispatch({ 
          type: MEDICAL_ACTIONS.SET_ERROR, 
          payload: 'Failed to connect to medical AI system' 
        });
      },
    }
  );

  // Load available models
  const { data: models } = useQuery(
    'models',
    async () => {
      const response = await api.get('/models/available');
      return response.data;
    },
    {
      onSuccess: (data) => {
        dispatch({ type: MEDICAL_ACTIONS.SET_MODELS, payload: data });
      },
      onError: (error) => {
        console.error('Failed to load models:', error);
      },
    }
  );

  // Send text message mutation
  const sendTextMessage = useMutation(
    async (messageData) => {
      const response = await api.post('/chat', messageData);
      return response.data;
    },
    {
      onMutate: () => {
        dispatch({ type: MEDICAL_ACTIONS.SET_PROCESSING, payload: true });
      },
      onSuccess: (data, variables) => {
        const userMessage = {
          id: Date.now(),
          type: 'user',
          content: variables.message,
          timestamp: new Date(),
        };

        const aiMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: data.response,
          timestamp: new Date(),
          metadata: {
            riskLevel: data.risk_level,
            isEmergency: data.is_emergency,
            confidenceScore: data.confidence_score,
            sources: data.sources,
            recommendations: data.recommendations,
          },
        };

        dispatch({ type: MEDICAL_ACTIONS.ADD_CHAT_MESSAGE, payload: userMessage });
        dispatch({ type: MEDICAL_ACTIONS.ADD_CHAT_MESSAGE, payload: aiMessage });
        dispatch({ type: MEDICAL_ACTIONS.SET_PROCESSING, payload: false });

        // Update stats
        dispatch({
          type: MEDICAL_ACTIONS.UPDATE_MEDICAL_STATS,
          payload: {
            totalChats: state.medicalStats.totalChats + 1,
            emergencyDetections: data.is_emergency 
              ? state.medicalStats.emergencyDetections + 1 
              : state.medicalStats.emergencyDetections,
          },
        });

        // Show emergency alert if needed
        if (data.is_emergency) {
          toast.error('🚨 Emergency detected! Please seek immediate medical attention!', {
            duration: 10000,
          });
        }
      },
      onError: (error) => {
        dispatch({ 
          type: MEDICAL_ACTIONS.SET_ERROR, 
          payload: error.response?.data?.detail || 'Failed to send message' 
        });
        toast.error('Failed to send message. Please try again.');
      },
    }
  );

  // Send voice message mutation
  const sendVoiceMessage = useMutation(
    async (audioFile) => {
      const formData = new FormData();
      formData.append('audio_file', audioFile);

      const response = await api.post('/voice/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    },
    {
      onMutate: () => {
        dispatch({ type: MEDICAL_ACTIONS.SET_PROCESSING, payload: true });
      },
      onSuccess: (data) => {
        const voiceMessage = {
          id: Date.now(),
          type: 'user',
          content: data.transcription,
          timestamp: new Date(),
          isVoice: true,
        };

        const aiMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: data.response,
          timestamp: new Date(),
          metadata: {
            riskLevel: data.risk_level,
            isEmergency: data.is_emergency,
            confidenceScore: data.confidence_score,
            sources: data.sources,
            recommendations: data.recommendations,
          },
        };

        dispatch({ type: MEDICAL_ACTIONS.ADD_CHAT_MESSAGE, payload: voiceMessage });
        dispatch({ type: MEDICAL_ACTIONS.ADD_CHAT_MESSAGE, payload: aiMessage });
        dispatch({ type: MEDICAL_ACTIONS.SET_PROCESSING, payload: false });

        // Update stats
        dispatch({
          type: MEDICAL_ACTIONS.UPDATE_MEDICAL_STATS,
          payload: {
            totalVoiceChats: state.medicalStats.totalVoiceChats + 1,
            emergencyDetections: data.is_emergency 
              ? state.medicalStats.emergencyDetections + 1 
              : state.medicalStats.emergencyDetections,
          },
        });

        // Show emergency alert if needed
        if (data.is_emergency) {
          toast.error('🚨 Emergency detected! Please seek immediate medical attention!', {
            duration: 10000,
          });
        }
      },
      onError: (error) => {
        dispatch({ 
          type: MEDICAL_ACTIONS.SET_ERROR, 
          payload: error.response?.data?.detail || 'Failed to process voice message' 
        });
        toast.error('Failed to process voice message. Please try again.');
      },
    }
  );

  // Change model mutation
  const changeModel = useMutation(
    async (modelName) => {
      const response = await api.post('/models/llm/change', null, {
        params: { model_name: modelName },
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        dispatch({ type: MEDICAL_ACTIONS.SET_CURRENT_MODEL, payload: data.model_info });
        toast.success(`Model changed to ${data.model_info.model_name}`);
        queryClient.invalidateQueries('systemInfo');
      },
      onError: (error) => {
        toast.error('Failed to change model. Please try again.');
        console.error('Model change error:', error);
      },
    }
  );

  // Utility functions
  const clearChatHistory = () => {
    dispatch({ type: MEDICAL_ACTIONS.CLEAR_CHAT_HISTORY });
    localStorage.removeItem('medicalChatHistory');
    toast.success('Chat history cleared');
  };

  const updateVoiceSettings = (settings) => {
    dispatch({ type: MEDICAL_ACTIONS.UPDATE_VOICE_SETTINGS, payload: settings });
    localStorage.setItem('medicalVoiceSettings', JSON.stringify({
      ...state.voiceSettings,
      ...settings,
    }));
  };

  const clearError = () => {
    dispatch({ type: MEDICAL_ACTIONS.CLEAR_ERROR });
  };

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('medicalChatHistory');
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory);
        dispatch({ type: MEDICAL_ACTIONS.LOAD_CHAT_HISTORY, payload: parsedHistory });
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }

    const savedVoiceSettings = localStorage.getItem('medicalVoiceSettings');
    if (savedVoiceSettings) {
      try {
        const parsedSettings = JSON.parse(savedVoiceSettings);
        dispatch({ type: MEDICAL_ACTIONS.UPDATE_VOICE_SETTINGS, payload: parsedSettings });
      } catch (error) {
        console.error('Failed to load voice settings:', error);
      }
    }
  }, []);

  // Save chat history to localStorage
  useEffect(() => {
    if (state.chatHistory.length > 0) {
      localStorage.setItem('medicalChatHistory', JSON.stringify(state.chatHistory));
    }
  }, [state.chatHistory]);

  const value = {
    ...state,
    sendTextMessage: sendTextMessage.mutate,
    sendVoiceMessage: sendVoiceMessage.mutate,
    changeModel: changeModel.mutate,
    clearChatHistory,
    updateVoiceSettings,
    clearError,
    isLoading: sendTextMessage.isLoading || sendVoiceMessage.isLoading || changeModel.isLoading,
  };

  return (
    <MedicalContext.Provider value={value}>
      {children}
    </MedicalContext.Provider>
  );
};

// Custom hook
export const useMedical = () => {
  const context = useContext(MedicalContext);
  if (!context) {
    throw new Error('useMedical must be used within a MedicalProvider');
  }
  return context;
};

export default MedicalContext;
