import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { notification } from 'antd';

interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp: string;
  connectionId?: string;
}

interface WebSocketContextType {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
  sendMessage: (message: any) => void;
  subscribe: (subscriptions: string[]) => void;
  unsubscribe: (subscriptions: string[]) => void;
  lastMessage: WebSocketMessage | null;
  connectionStats: {
    messagesReceived: number;
    messagesLost: number;
    reconnectAttempts: number;
    lastReconnectTime: Date | null;
    uptime: number;
  };
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: React.ReactNode;
  wsUrl: string;
  onConnectionStatus?: (status: 'connecting' | 'connected' | 'disconnected') => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  wsUrl,
  onConnectionStatus,
  onMessage,
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionStats, setConnectionStats] = useState({
    messagesReceived: 0,
    messagesLost: 0,
    reconnectAttempts: 0,
    lastReconnectTime: null as Date | null,
    uptime: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const connectionTimeRef = useRef<Date | null>(null);
  const uptimeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<any[]>([]);
  const subscribedChannelsRef = useRef<string[]>(['price', 'predictions']);
  const lastMessageTimeRef = useRef<Date>(new Date());
  const missedHeartbeatsRef = useRef<number>(0);

  // Connection configuration
  const maxReconnectAttempts = 10;
  const reconnectDelay = 1000; // Start with 1 second
  const heartbeatInterval = 30000; // 30 seconds
  const maxMissedHeartbeats = 3;

  // Update connection status
  const updateConnectionStatus = useCallback((status: 'connecting' | 'connected' | 'disconnected') => {
    setConnectionStatus(status);
    setIsConnected(status === 'connected');
    onConnectionStatus?.(status);

    if (status === 'connected') {
      connectionTimeRef.current = new Date();
      setConnectionStats(prev => ({
        ...prev,
        reconnectAttempts: 0,
      }));
    } else if (status === 'disconnected') {
      connectionTimeRef.current = null;
    }
  }, [onConnectionStatus]);

  // Start uptime tracking
  const startUptimeTracking = useCallback(() => {
    if (uptimeIntervalRef.current) {
      clearInterval(uptimeIntervalRef.current);
    }

    uptimeIntervalRef.current = setInterval(() => {
      if (connectionTimeRef.current) {
        const uptime = Date.now() - connectionTimeRef.current.getTime();
        setConnectionStats(prev => ({
          ...prev,
          uptime: Math.floor(uptime / 1000),
        }));
      }
    }, 1000);
  }, []);

  // Stop uptime tracking
  const stopUptimeTracking = useCallback(() => {
    if (uptimeIntervalRef.current) {
      clearInterval(uptimeIntervalRef.current);
      uptimeIntervalRef.current = null;
    }
  }, []);

  // Send heartbeat
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'heartbeat',
        timestamp: new Date().toISOString(),
      }));
    }
  }, []);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    heartbeatIntervalRef.current = setInterval(() => {
      const now = new Date();
      const timeSinceLastMessage = now.getTime() - lastMessageTimeRef.current.getTime();

      // Check if we've missed too many heartbeats
      if (timeSinceLastMessage > heartbeatInterval * maxMissedHeartbeats) {
        missedHeartbeatsRef.current += 1;
        
        if (missedHeartbeatsRef.current >= maxMissedHeartbeats) {
          console.warn('Too many missed heartbeats, forcing reconnection');
          wsRef.current?.close();
          return;
        }
      }

      sendHeartbeat();
    }, heartbeatInterval);
  }, [sendHeartbeat]);

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // Process queued messages
  const processMessageQueue = useCallback(() => {
    while (messageQueueRef.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
      const message = messageQueueRef.current.shift();
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    updateConnectionStatus('connecting');

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus('connected');
        startHeartbeat();
        startUptimeTracking();
        missedHeartbeatsRef.current = 0;

        // Process any queued messages
        processMessageQueue();

        // Re-subscribe to channels
        if (subscribedChannelsRef.current.length > 0) {
          sendMessage({
            action: 'subscribe',
            subscriptions: subscribedChannelsRef.current,
          });
        }
      };

      wsRef.current.onmessage = (event) => {
        lastMessageTimeRef.current = new Date();
        missedHeartbeatsRef.current = 0;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);

          setConnectionStats(prev => ({
            ...prev,
            messagesReceived: prev.messagesReceived + 1,
          }));

          // Handle specific message types
          switch (message.type) {
            case 'connection_established':
              console.log('WebSocket connection established:', message.connectionId);
              break;
            case 'error':
              console.error('WebSocket error:', message.data);
              notification.error({
                message: 'WebSocket Error',
                description: message.data?.message || 'Unknown error occurred',
              });
              break;
            case 'subscription_updated':
              console.log('Subscriptions updated:', message.data);
              break;
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          setConnectionStats(prev => ({
            ...prev,
            messagesLost: prev.messagesLost + 1,
          }));
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        updateConnectionStatus('disconnected');
        stopHeartbeat();
        stopUptimeTracking();

        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && connectionStats.reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(reconnectDelay * Math.pow(2, connectionStats.reconnectAttempts), 30000);
          
          setConnectionStats(prev => ({
            ...prev,
            reconnectAttempts: prev.reconnectAttempts + 1,
            lastReconnectTime: new Date(),
          }));

          console.log(`Attempting to reconnect in ${delay}ms (attempt ${connectionStats.reconnectAttempts + 1}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else if (connectionStats.reconnectAttempts >= maxReconnectAttempts) {
          notification.error({
            message: 'Connection Failed',
            description: 'Unable to establish WebSocket connection after multiple attempts. Please refresh the page.',
            duration: 0,
          });
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStats(prev => ({
          ...prev,
          messagesLost: prev.messagesLost + 1,
        }));
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      updateConnectionStatus('disconnected');
    }
  }, [wsUrl, updateConnectionStatus, onMessage, connectionStats.reconnectAttempts, startHeartbeat, startUptimeTracking, stopHeartbeat, stopUptimeTracking, processMessageQueue]);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      // Queue message for when connection is restored
      messageQueueRef.current.push(message);
      console.warn('WebSocket not connected, message queued');
    }
  }, []);

  // Subscribe to channels
  const subscribe = useCallback((subscriptions: string[]) => {
    subscribedChannelsRef.current = [...new Set([...subscribedChannelsRef.current, ...subscriptions])];
    sendMessage({
      action: 'subscribe',
      subscriptions: subscribedChannelsRef.current,
    });
  }, [sendMessage]);

  // Unsubscribe from channels
  const unsubscribe = useCallback((subscriptions: string[]) => {
    subscribedChannelsRef.current = subscribedChannelsRef.current.filter(
      sub => !subscriptions.includes(sub)
    );
    sendMessage({
      action: 'unsubscribe',
      subscriptions: subscriptions,
    });
  }, [sendMessage]);

  // Initialize connection
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopHeartbeat();
      stopUptimeTracking();
      wsRef.current?.close(1000, 'Component unmounting');
    };
  }, [connect, stopHeartbeat, stopUptimeTracking]);

  // Handle page visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, reduce heartbeat frequency
        stopHeartbeat();
      } else {
        // Page is visible, resume normal heartbeat
        if (isConnected) {
          startHeartbeat();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isConnected, startHeartbeat, stopHeartbeat]);

  // Handle window focus/blur
  useEffect(() => {
    const handleFocus = () => {
      // Window gained focus, ensure connection is alive
      if (!isConnected && connectionStats.reconnectAttempts < maxReconnectAttempts) {
        connect();
      }
    };

    const handleBlur = () => {
      // Window lost focus, could reduce activity
    };

    window.addEventListener('focus', handleFocus);
    window.addEventListener('blur', handleBlur);

    return () => {
      window.removeEventListener('focus', handleFocus);
      window.removeEventListener('blur', handleBlur);
    };
  }, [isConnected, connectionStats.reconnectAttempts, connect]);

  const contextValue: WebSocketContextType = {
    isConnected,
    connectionStatus,
    sendMessage,
    subscribe,
    unsubscribe,
    lastMessage,
    connectionStats,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};