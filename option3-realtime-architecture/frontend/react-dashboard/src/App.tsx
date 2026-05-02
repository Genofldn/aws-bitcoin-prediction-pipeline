import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, ConfigProvider, theme, notification } from 'antd';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Amplify } from 'aws-amplify';
import { withAuthenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import styled, { ThemeProvider, createGlobalStyle } from 'styled-components';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import Alerts from './pages/Alerts';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { DataProvider } from './contexts/DataContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';

// Configure AWS Amplify
Amplify.configure({
  Auth: {
    region: process.env.REACT_APP_AWS_REGION || 'eu-west-2',
    userPoolId: process.env.REACT_APP_USER_POOL_ID,
    userPoolWebClientId: process.env.REACT_APP_USER_POOL_WEB_CLIENT_ID,
  }
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Styled components
const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-color: ${props => props.theme.backgroundColor};
    color: ${props => props.theme.textColor};
    transition: background-color 0.3s ease, color 0.3s ease;
  }

  .ant-layout {
    min-height: 100vh;
  }

  .ant-layout-content {
    padding: 24px;
    background: ${props => props.theme.backgroundColor};
  }

  /* Custom scrollbar */
  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${props => props.theme.scrollbarTrack};
  }

  ::-webkit-scrollbar-thumb {
    background: ${props => props.theme.scrollbarThumb};
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${props => props.theme.scrollbarThumbHover};
  }

  /* Animation keyframes */
  @keyframes pulse {
    0% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
    100% {
      opacity: 1;
    }
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .fade-in {
    animation: fadeIn 0.5s ease-in-out;
  }

  .pulse {
    animation: pulse 2s infinite;
  }
`;

const StyledLayout = styled(Layout)`
  min-height: 100vh;
`;

const { Content } = Layout;

// Theme definitions
const lightTheme = {
  backgroundColor: '#ffffff',
  textColor: '#000000',
  cardBackground: '#f8f9fa',
  borderColor: '#e9ecef',
  scrollbarTrack: '#f1f1f1',
  scrollbarThumb: '#c1c1c1',
  scrollbarThumbHover: '#a8a8a8',
};

const darkTheme = {
  backgroundColor: '#001529',
  textColor: '#ffffff',
  cardBackground: '#1f1f1f',
  borderColor: '#434343',
  scrollbarTrack: '#2c2c2c',
  scrollbarThumb: '#555555',
  scrollbarThumbHover: '#777777',
};

interface AppProps {
  signOut?: () => void;
  user?: any;
}

const App: React.FC<AppProps> = ({ signOut, user }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  // Theme configuration for Ant Design
  const themeConfig = {
    algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: '#1890ff',
      colorSuccess: '#52c41a',
      colorWarning: '#faad14',
      colorError: '#ff4d4f',
      borderRadius: 6,
      wireframe: false,
    },
  };

  // Theme toggle handler
  const toggleTheme = useCallback(() => {
    setIsDarkMode(prev => {
      const newValue = !prev;
      localStorage.setItem('darkMode', JSON.stringify(newValue));
      return newValue;
    });
  }, []);

  // Handle sidebar collapse
  const handleSidebarCollapse = useCallback((collapsed: boolean) => {
    setCollapsed(collapsed);
  }, []);

  // Handle WebSocket connection status
  const handleConnectionStatus = useCallback((status: 'connecting' | 'connected' | 'disconnected') => {
    setConnectionStatus(status);
    
    // Show notification on status change
    if (status === 'connected') {
      notification.success({
        message: 'Connected',
        description: 'Real-time data connection established',
        duration: 2,
      });
    } else if (status === 'disconnected') {
      notification.error({
        message: 'Disconnected',
        description: 'Real-time data connection lost. Attempting to reconnect...',
        duration: 4,
      });
    }
  }, []);

  // Initialize notification configuration
  useEffect(() => {
    notification.config({
      placement: 'topRight',
      duration: 3,
      maxCount: 3,
    });
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider theme={themeConfig}>
        <ThemeProvider theme={isDarkMode ? darkTheme : lightTheme}>
          <CustomThemeProvider value={{ isDarkMode, toggleTheme }}>
            <GlobalStyle />
            <WebSocketProvider 
              wsUrl={process.env.REACT_APP_WEBSOCKET_URL || 'wss://your-websocket-api.execute-api.eu-west-2.amazonaws.com/production'}
              onConnectionStatus={handleConnectionStatus}
            >
              <DataProvider>
                <Router>
                  <StyledLayout>
                    <Sidebar 
                      collapsed={collapsed} 
                      onCollapse={handleSidebarCollapse}
                      isDarkMode={isDarkMode}
                    />
                    <Layout style={{ marginLeft: collapsed ? 80 : 200 }}>
                      <Header
                        collapsed={collapsed}
                        onCollapse={handleSidebarCollapse}
                        isDarkMode={isDarkMode}
                        onToggleTheme={toggleTheme}
                        connectionStatus={connectionStatus}
                        user={user}
                        onSignOut={signOut}
                      />
                      <Content>
                        <Routes>
                          <Route path="/" element={<Navigate to="/dashboard" replace />} />
                          <Route path="/dashboard" element={<Dashboard />} />
                          <Route path="/analytics" element={<Analytics />} />
                          <Route path="/alerts" element={<Alerts />} />
                          <Route path="/settings" element={<Settings />} />
                          <Route path="*" element={<Navigate to="/dashboard" replace />} />
                        </Routes>
                      </Content>
                    </Layout>
                  </StyledLayout>
                </Router>
              </DataProvider>
            </WebSocketProvider>
          </CustomThemeProvider>
        </ThemeProvider>
      </ConfigProvider>
    </QueryClientProvider>
  );
};

export default withAuthenticator(App, {
  socialProviders: [],
  signUpAttributes: ['email'],
  hideSignUp: true, // Only allow admin sign-ins
  loginMechanisms: ['email'],
});