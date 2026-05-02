import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Row, Col, Card, Statistic, Alert, Spin, Switch, Button, Tooltip, Space } from 'antd';
import { 
  DollarOutlined, 
  TrendingUpOutlined, 
  TrendingDownOutlined, 
  EyeOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  LineChartOutlined,
  FullscreenOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import numeral from 'numeral';
import moment from 'moment';

// Components
import PriceChart from '../components/charts/PriceChart';
import PredictionChart from '../components/charts/PredictionChart';
import SentimentGauge from '../components/charts/SentimentGauge';
import VolumeChart from '../components/charts/VolumeChart';
import NewsStream from '../components/NewsStream';
import AlertsPanel from '../components/AlertsPanel';
import PerformanceMetrics from '../components/PerformanceMetrics';

// Hooks
import { useWebSocket } from '../contexts/WebSocketContext';
import { useData } from '../contexts/DataContext';

// Types
interface PriceData {
  timestamp: string;
  price: number;
  volume: number;
  change24h: number;
  marketCap: number;
}

interface PredictionData {
  timestamp: string;
  predictions: {
    price_1h: number;
    price_6h: number;
    price_24h: number;
    price_7d: number;
  };
  confidence: number;
  model_type: string;
}

// Styled components
const DashboardContainer = styled.div`
  padding: 0;
  background: transparent;
  min-height: calc(100vh - 112px);
`;

const StyledCard = styled(Card)`
  margin-bottom: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
  }

  .ant-card-head {
    border-bottom: 1px solid ${props => props.theme.borderColor};
  }

  .ant-card-body {
    padding: 20px;
  }
`;

const StatisticCard = styled(StyledCard)`
  text-align: center;
  
  .ant-statistic-title {
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 8px;
  }

  .ant-statistic-content {
    font-size: 24px;
    font-weight: bold;
  }
`;

const LiveIndicator = styled.div<{ isLive: boolean }>`
  display: inline-flex;
  align-items: center;
  padding: 4px 8px;
  border-radius: 4px;
  background-color: ${props => props.isLive ? '#f6ffed' : '#fff2e8'};
  border: 1px solid ${props => props.isLive ? '#b7eb8f' : '#ffbb96'};
  
  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: ${props => props.isLive ? '#52c41a' : '#faad14'};
    margin-right: 6px;
    animation: ${props => props.isLive ? 'pulse 2s infinite' : 'none'};
  }
  
  font-size: 12px;
  font-weight: 500;
  color: ${props => props.isLive ? '#52c41a' : '#faad14'};
`;

const ChartContainer = styled.div`
  height: 400px;
  width: 100%;
  position: relative;
`;

const ControlPanel = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding: 12px 16px;
  background: ${props => props.theme.cardBackground};
  border-radius: 6px;
  border: 1px solid ${props => props.theme.borderColor};
`;

const Dashboard: React.FC = () => {
  const { isConnected, lastMessage, sendMessage } = useWebSocket();
  const { priceData, predictionData, newsData, isLoading } = useData();
  
  // State
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null);
  const [currentPrice, setCurrentPrice] = useState<PriceData | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<PredictionData | null>(null);
  const [priceHistory, setPriceHistory] = useState<PriceData[]>([]);
  const [predictionHistory, setPredictionHistory] = useState<PredictionData[]>([]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'price_update':
          const priceUpdate = lastMessage.data;
          setCurrentPrice(priceUpdate);
          setPriceHistory(prev => [...prev.slice(-99), priceUpdate].sort((a, b) => 
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
          ));
          break;
          
        case 'prediction_update':
          const predictionUpdate = lastMessage.data;
          setLatestPrediction(predictionUpdate);
          setPredictionHistory(prev => [...prev.slice(-49), predictionUpdate].sort((a, b) => 
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
          ));
          break;
      }
    }
  }, [lastMessage]);

  // Request live data when connected
  useEffect(() => {
    if (isConnected && autoRefresh) {
      const interval = setInterval(() => {
        sendMessage({ action: 'get_prediction' });
      }, 5000); // Request prediction every 5 seconds

      return () => clearInterval(interval);
    }
  }, [isConnected, autoRefresh, sendMessage]);

  // Manual refresh
  const handleRefresh = useCallback(() => {
    sendMessage({ action: 'get_prediction' });
    sendMessage({ action: 'get_history', timeframe: selectedTimeframe });
  }, [sendMessage, selectedTimeframe]);

  // Toggle fullscreen chart
  const toggleFullscreen = useCallback((chartType: string) => {
    setFullscreenChart(fullscreenChart === chartType ? null : chartType);
  }, [fullscreenChart]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!currentPrice || !latestPrediction) {
      return {
        currentPrice: 0,
        change24h: 0,
        volume24h: 0,
        marketCap: 0,
        prediction1h: 0,
        prediction24h: 0,
        confidence: 0,
        accuracy: 0,
      };
    }

    const prediction1hChange = ((latestPrediction.predictions.price_1h - currentPrice.price) / currentPrice.price) * 100;
    const prediction24hChange = ((latestPrediction.predictions.price_24h - currentPrice.price) / currentPrice.price) * 100;

    return {
      currentPrice: currentPrice.price,
      change24h: currentPrice.change24h,
      volume24h: currentPrice.volume,
      marketCap: currentPrice.marketCap,
      prediction1h: prediction1hChange,
      prediction24h: prediction24hChange,
      confidence: latestPrediction.confidence * 100,
      accuracy: 78.5, // This would come from model performance tracking
    };
  }, [currentPrice, latestPrediction]);

  // Format currency
  const formatCurrency = useCallback((value: number) => {
    return numeral(value).format('$0,0.00');
  }, []);

  // Format percentage
  const formatPercentage = useCallback((value: number) => {
    return `${value >= 0 ? '+' : ''}${numeral(value).format('0.00')}%`;
  }, []);

  if (isLoading) {
    return (
      <DashboardContainer>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <Spin size="large" />
        </div>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Header Controls */}
          <ControlPanel>
            <Space>
              <LiveIndicator isLive={isConnected}>
                {isConnected ? 'LIVE' : 'OFFLINE'}
              </LiveIndicator>
              <span>Last Update: {moment().format('HH:mm:ss')}</span>
            </Space>
            
            <Space>
              <Tooltip title="Auto refresh every 5 seconds">
                <Switch
                  checked={autoRefresh}
                  onChange={setAutoRefresh}
                  checkedChildren="AUTO"
                  unCheckedChildren="MANUAL"
                />
              </Tooltip>
              <Button 
                icon={<ReloadOutlined />} 
                onClick={handleRefresh}
                disabled={!isConnected}
              >
                Refresh
              </Button>
            </Space>
          </ControlPanel>

          {/* Connection Alert */}
          {!isConnected && (
            <Alert
              message="Real-time connection unavailable"
              description="The dashboard is showing cached data. Real-time updates will resume when connection is restored."
              type="warning"
              showIcon
              style={{ marginBottom: 24 }}
            />
          )}

          {/* Price Statistics Row */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={12} md={6}>
              <StatisticCard>
                <Statistic
                  title="Current Price"
                  value={statistics.currentPrice}
                  formatter={(value) => formatCurrency(Number(value))}
                  prefix={<DollarOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </StatisticCard>
            </Col>
            
            <Col xs={24} sm={12} md={6}>
              <StatisticCard>
                <Statistic
                  title="24h Change"
                  value={statistics.change24h}
                  formatter={(value) => formatPercentage(Number(value))}
                  prefix={statistics.change24h >= 0 ? <TrendingUpOutlined /> : <TrendingDownOutlined />}
                  valueStyle={{ color: statistics.change24h >= 0 ? '#52c41a' : '#ff4d4f' }}
                />
              </StatisticCard>
            </Col>
            
            <Col xs={24} sm={12} md={6}>
              <StatisticCard>
                <Statistic
                  title="1h Prediction"
                  value={statistics.prediction1h}
                  formatter={(value) => formatPercentage(Number(value))}
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{ color: statistics.prediction1h >= 0 ? '#52c41a' : '#ff4d4f' }}
                />
              </StatisticCard>
            </Col>
            
            <Col xs={24} sm={12} md={6}>
              <StatisticCard>
                <Statistic
                  title="Model Confidence"
                  value={statistics.confidence}
                  formatter={(value) => `${numeral(value).format('0.0')}%`}
                  prefix={<ThunderboltOutlined />}
                  valueStyle={{ color: statistics.confidence >= 70 ? '#52c41a' : statistics.confidence >= 50 ? '#faad14' : '#ff4d4f' }}
                />
              </StatisticCard>
            </Col>
          </Row>

          {/* Main Charts Row */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} lg={16}>
              <StyledCard
                title={
                  <Space>
                    <LineChartOutlined />
                    Bitcoin Price Chart
                    <Switch 
                      size="small" 
                      checked={selectedTimeframe === '24h'} 
                      onChange={(checked) => setSelectedTimeframe(checked ? '24h' : '1h')}
                      checkedChildren="24h"
                      unCheckedChildren="1h"
                    />
                  </Space>
                }
                extra={
                  <Button 
                    icon={<FullscreenOutlined />} 
                    size="small"
                    onClick={() => toggleFullscreen('price')}
                  />
                }
              >
                <ChartContainer>
                  <PriceChart 
                    data={priceHistory}
                    predictions={predictionHistory}
                    height={360}
                    isLive={isConnected}
                    timeframe={selectedTimeframe}
                  />
                </ChartContainer>
              </StyledCard>
            </Col>
            
            <Col xs={24} lg={8}>
              <StyledCard
                title={
                  <Space>
                    <EyeOutlined />
                    Market Sentiment
                  </Space>
                }
                extra={
                  <Button 
                    icon={<FullscreenOutlined />} 
                    size="small"
                    onClick={() => toggleFullscreen('sentiment')}
                  />
                }
              >
                <ChartContainer>
                  <SentimentGauge 
                    value={newsData?.overallSentiment || 0}
                    newsCount={newsData?.articles?.length || 0}
                    height={360}
                  />
                </ChartContainer>
              </StyledCard>
            </Col>
          </Row>

          {/* Predictions and Volume Row */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} lg={12}>
              <StyledCard
                title={
                  <Space>
                    <BarChartOutlined />
                    Prediction Models
                  </Space>
                }
                extra={
                  <Button 
                    icon={<FullscreenOutlined />} 
                    size="small"
                    onClick={() => toggleFullscreen('predictions')}
                  />
                }
              >
                <ChartContainer>
                  <PredictionChart 
                    data={predictionHistory}
                    currentPrice={statistics.currentPrice}
                    height={360}
                  />
                </ChartContainer>
              </StyledCard>
            </Col>
            
            <Col xs={24} lg={12}>
              <StyledCard
                title={
                  <Space>
                    <BarChartOutlined />
                    Trading Volume
                  </Space>
                }
                extra={
                  <Button 
                    icon={<FullscreenOutlined />} 
                    size="small"
                    onClick={() => toggleFullscreen('volume')}
                  />
                }
              >
                <ChartContainer>
                  <VolumeChart 
                    data={priceHistory}
                    height={360}
                  />
                </ChartContainer>
              </StyledCard>
            </Col>
          </Row>

          {/* Bottom Row - News and Performance */}
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <StyledCard
                title="Live News Stream"
                extra={<Button size="small" onClick={handleRefresh}>Refresh</Button>}
              >
                <NewsStream 
                  news={newsData?.articles || []}
                  maxHeight={400}
                  isLive={isConnected}
                />
              </StyledCard>
            </Col>
            
            <Col xs={24} lg={12}>
              <StyledCard title="Model Performance">
                <PerformanceMetrics 
                  accuracy={statistics.accuracy}
                  predictions={predictionHistory}
                  actualPrices={priceHistory}
                />
              </StyledCard>
            </Col>
          </Row>

          {/* Alerts Panel */}
          <AlertsPanel />
        </motion.div>
      </AnimatePresence>
    </DashboardContainer>
  );
};

export default Dashboard;