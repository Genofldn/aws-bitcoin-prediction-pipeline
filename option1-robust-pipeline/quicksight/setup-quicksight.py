#!/usr/bin/env python3
"""
QuickSight Setup and Configuration Script for Bitcoin Prediction Pipeline

This script automates the setup of QuickSight data sources, datasets, and dashboards
for visualizing Bitcoin prediction data.
"""

import boto3
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickSightSetup:
    def __init__(self, region: str = 'eu-west-2', account_id: str = None):
        """
        Initialize QuickSight setup
        
        Args:
            region: AWS region
            account_id: AWS account ID
        """
        self.region = region
        self.account_id = account_id or boto3.client('sts').get_caller_identity()['Account']
        
        # Initialize AWS clients
        self.quicksight = boto3.client('quicksight', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        
        # Configuration
        self.namespace = 'default'
        self.data_source_id = 'bitcoin-prediction-datasource'
        self.dataset_id = 'bitcoin-prediction-dataset'
        self.dashboard_id = 'bitcoin-prediction-dashboard'
        self.analysis_id = 'bitcoin-prediction-analysis'
        
        # Table names
        self.predictions_table = 'bitcoin-predictions'
        self.raw_data_table = 'bitcoin-raw-data'

    def check_quicksight_setup(self) -> Dict[str, Any]:
        """Check if QuickSight is set up and user exists"""
        try:
            # Check if user exists
            user_response = self.quicksight.describe_user(
                UserName=f'quicksight-user-{self.account_id}',
                AwsAccountId=self.account_id,
                Namespace=self.namespace
            )
            
            logger.info("QuickSight user found")
            return {
                'user_exists': True,
                'user_arn': user_response['User']['Arn'],
                'user_role': user_response['User']['Role']
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning("QuickSight user not found - manual setup may be required")
                return {'user_exists': False}
            else:
                logger.error(f"Error checking QuickSight setup: {str(e)}")
                raise

    def create_data_source(self) -> bool:
        """Create DynamoDB data source in QuickSight"""
        try:
            logger.info("Creating QuickSight data source...")
            
            # Data source configuration for DynamoDB
            data_source_parameters = {
                'DynamoDbParameters': {
                    'Region': self.region
                }
            }
            
            permissions = [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/{self.namespace}/quicksight-user-{self.account_id}',
                    'Actions': [
                        'quicksight:DescribeDataSource',
                        'quicksight:DescribeDataSourcePermissions',
                        'quicksight:PassDataSource'
                    ]
                }
            ]
            
            response = self.quicksight.create_data_source(
                AwsAccountId=self.account_id,
                DataSourceId=self.data_source_id,
                Name='Bitcoin Prediction DynamoDB Source',
                Type='DYNAMODB',
                DataSourceParameters=data_source_parameters,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinPrediction'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            
            logger.info(f"Data source created: {response['Arn']}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Data source already exists")
                return True
            else:
                logger.error(f"Failed to create data source: {str(e)}")
                return False

    def create_dataset(self) -> bool:
        """Create dataset for Bitcoin predictions"""
        try:
            logger.info("Creating QuickSight dataset...")
            
            # Physical table configuration
            physical_table_map = {
                'PredictionsTable': {
                    'DynamoDbSource': {
                        'DataSourceArn': f'arn:aws:quicksight:{self.region}:{self.account_id}:datasource/{self.data_source_id}',
                        'TableName': self.predictions_table,
                        'Columns': [
                            {'Name': 'id', 'Type': 'STRING'},
                            {'Name': 'timestamp', 'Type': 'INTEGER'},
                            {'Name': 'prediction_type', 'Type': 'STRING'},
                            {'Name': 'current_price', 'Type': 'DECIMAL'},
                            {'Name': 'confidence_score', 'Type': 'DECIMAL'},
                            {'Name': 'model_version', 'Type': 'STRING'},
                            {'Name': 'created_at', 'Type': 'STRING'},
                            {'Name': 'predictions', 'Type': 'STRING'},
                            {'Name': 'changes', 'Type': 'STRING'}
                        ]
                    }
                }
            }
            
            # Logical table configuration with transformations
            logical_table_map = {
                'ProcessedPredictions': {
                    'Alias': 'Bitcoin Predictions',
                    'Source': {
                        'PhysicalTableId': 'PredictionsTable'
                    },
                    'DataTransforms': [
                        {
                            'ProjectOperation': {
                                'ProjectedColumns': [
                                    'id', 'timestamp', 'prediction_type', 'current_price',
                                    'confidence_score', 'model_version', 'created_at',
                                    'predictions', 'changes'
                                ]
                            }
                        },
                        {
                            'FilterOperation': {
                                'ConditionExpression': 'prediction_type = "bitcoin_price"'
                            }
                        }
                    ]
                }
            }
            
            # Import mode and permissions
            import_mode = 'DIRECT_QUERY'  # Use SPICE for better performance in production
            
            permissions = [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/{self.namespace}/quicksight-user-{self.account_id}',
                    'Actions': [
                        'quicksight:DescribeDataSet',
                        'quicksight:DescribeDataSetPermissions',
                        'quicksight:PassDataSet',
                        'quicksight:DescribeIngestion',
                        'quicksight:ListIngestions'
                    ]
                }
            ]
            
            response = self.quicksight.create_data_set(
                AwsAccountId=self.account_id,
                DataSetId=self.dataset_id,
                Name='Bitcoin Prediction Dataset',
                PhysicalTableMap=physical_table_map,
                LogicalTableMap=logical_table_map,
                ImportMode=import_mode,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinPrediction'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            
            logger.info(f"Dataset created: {response['Arn']}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Dataset already exists")
                return True
            else:
                logger.error(f"Failed to create dataset: {str(e)}")
                return False

    def create_analysis(self) -> bool:
        """Create QuickSight analysis"""
        try:
            logger.info("Creating QuickSight analysis...")
            
            # Analysis definition
            definition = {
                'DataSetIdentifierDeclarations': [
                    {
                        'DataSetArn': f'arn:aws:quicksight:{self.region}:{self.account_id}:dataset/{self.dataset_id}',
                        'Identifier': 'bitcoin_predictions'
                    }
                ],
                'Sheets': [
                    {
                        'SheetId': 'sheet1',
                        'Name': 'Bitcoin Price Predictions',
                        'Visuals': [
                            {
                                'LineChartVisual': {
                                    'VisualId': 'price_trend_chart',
                                    'Title': {
                                        'Visibility': 'VISIBLE',
                                        'Text': 'Bitcoin Price Trend and Predictions'
                                    },
                                    'ChartConfiguration': {
                                        'FieldWells': {
                                            'LineChartAggregatedFieldWells': {
                                                'Category': [
                                                    {
                                                        'DateDimensionField': {
                                                            'FieldId': 'timestamp',
                                                            'Column': {
                                                                'DataSetIdentifier': 'bitcoin_predictions',
                                                                'ColumnName': 'timestamp'
                                                            }
                                                        }
                                                    }
                                                ],
                                                'Values': [
                                                    {
                                                        'NumericalMeasureField': {
                                                            'FieldId': 'current_price',
                                                            'Column': {
                                                                'DataSetIdentifier': 'bitcoin_predictions',
                                                                'ColumnName': 'current_price'
                                                            }
                                                        }
                                                    }
                                                ]
                                            }
                                        },
                                        'SortConfiguration': {
                                            'CategorySort': [
                                                {
                                                    'FieldSort': {
                                                        'FieldId': 'timestamp',
                                                        'Direction': 'ASC'
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            },
                            {
                                'GaugeChartVisual': {
                                    'VisualId': 'confidence_gauge',
                                    'Title': {
                                        'Visibility': 'VISIBLE',
                                        'Text': 'Prediction Confidence Score'
                                    },
                                    'ChartConfiguration': {
                                        'FieldWells': {
                                            'Values': [
                                                {
                                                    'NumericalMeasureField': {
                                                        'FieldId': 'confidence_score',
                                                        'Column': {
                                                            'DataSetIdentifier': 'bitcoin_predictions',
                                                            'ColumnName': 'confidence_score'
                                                        },
                                                        'AggregationFunction': {
                                                            'SimpleNumericalAggregation': 'AVERAGE'
                                                        }
                                                    }
                                                }
                                            ]
                                        },
                                        'GaugeChartOptions': {
                                            'PrimaryValueDisplayType': 'ACTUAL'
                                        }
                                    }
                                }
                            }
                        ],
                        'Layouts': [
                            {
                                'Configuration': {
                                    'GridLayout': {
                                        'Elements': [
                                            {
                                                'ElementId': 'price_trend_chart',
                                                'ElementType': 'VISUAL',
                                                'ColumnIndex': 0,
                                                'ColumnSpan': 24,
                                                'RowIndex': 0,
                                                'RowSpan': 12
                                            },
                                            {
                                                'ElementId': 'confidence_gauge',
                                                'ElementType': 'VISUAL',
                                                'ColumnIndex': 0,
                                                'ColumnSpan': 12,
                                                'RowIndex': 12,
                                                'RowSpan': 8
                                            }
                                        ]
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
            
            permissions = [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/{self.namespace}/quicksight-user-{self.account_id}',
                    'Actions': [
                        'quicksight:RestoreAnalysis',
                        'quicksight:UpdateAnalysisPermissions',
                        'quicksight:DeleteAnalysis',
                        'quicksight:DescribeAnalysisPermissions',
                        'quicksight:QueryAnalysis',
                        'quicksight:DescribeAnalysis',
                        'quicksight:UpdateAnalysis'
                    ]
                }
            ]
            
            response = self.quicksight.create_analysis(
                AwsAccountId=self.account_id,
                AnalysisId=self.analysis_id,
                Name='Bitcoin Prediction Analysis',
                Definition=definition,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinPrediction'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            
            logger.info(f"Analysis created: {response['Arn']}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Analysis already exists")
                return True
            else:
                logger.error(f"Failed to create analysis: {str(e)}")
                return False

    def create_dashboard(self) -> bool:
        """Create QuickSight dashboard from analysis"""
        try:
            logger.info("Creating QuickSight dashboard...")
            
            # Dashboard source entity
            source_entity = {
                'SourceAnalysis': {
                    'Arn': f'arn:aws:quicksight:{self.region}:{self.account_id}:analysis/{self.analysis_id}',
                    'DataSetReferences': [
                        {
                            'DataSetArn': f'arn:aws:quicksight:{self.region}:{self.account_id}:dataset/{self.dataset_id}',
                            'DataSetPlaceholder': 'bitcoin_predictions'
                        }
                    ]
                }
            }
            
            permissions = [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/{self.namespace}/quicksight-user-{self.account_id}',
                    'Actions': [
                        'quicksight:DescribeDashboard',
                        'quicksight:ListDashboardVersions',
                        'quicksight:UpdateDashboardPermissions',
                        'quicksight:QueryDashboard',
                        'quicksight:UpdateDashboard',
                        'quicksight:DeleteDashboard',
                        'quicksight:DescribeDashboardPermissions',
                        'quicksight:UpdateDashboardPublishedVersion'
                    ]
                }
            ]
            
            # Dashboard parameters for refresh
            dashboard_publish_options = {
                'AdHocFilteringOption': {
                    'AvailabilityStatus': 'ENABLED'
                },
                'ExportToCSVOption': {
                    'AvailabilityStatus': 'ENABLED'
                },
                'SheetControlsOption': {
                    'VisibilityState': 'EXPANDED'
                }
            }
            
            response = self.quicksight.create_dashboard(
                AwsAccountId=self.account_id,
                DashboardId=self.dashboard_id,
                Name='Bitcoin Prediction Dashboard',
                Permissions=permissions,
                SourceEntity=source_entity,
                DashboardPublishOptions=dashboard_publish_options,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinPrediction'},
                    {'Key': 'Environment', 'Value': 'Production'},
                    {'Key': 'AutoRefresh', 'Value': 'Hourly'}
                ]
            )
            
            logger.info(f"Dashboard created: {response['Arn']}")
            logger.info(f"Dashboard URL: https://{self.region}.quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}")
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Dashboard already exists")
                return True
            else:
                logger.error(f"Failed to create dashboard: {str(e)}")
                return False

    def setup_auto_refresh(self) -> bool:
        """Set up auto-refresh for the dataset"""
        try:
            logger.info("Setting up dataset auto-refresh...")
            
            # Create refresh schedule
            refresh_schedule = {
                'ScheduleId': 'hourly-refresh',
                'ScheduleFrequency': {
                    'Interval': 'HOURLY'
                },
                'StartAfterDateTime': datetime.now(timezone.utc),
                'RefreshType': 'FULL_REFRESH'
            }
            
            response = self.quicksight.create_refresh_schedule(
                DataSetId=self.dataset_id,
                AwsAccountId=self.account_id,
                Schedule=refresh_schedule
            )
            
            logger.info("Auto-refresh schedule created")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Refresh schedule already exists")
                return True
            else:
                logger.warning(f"Failed to create refresh schedule: {str(e)}")
                # This is not critical, so return True
                return True

    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL"""
        try:
            response = self.quicksight.get_dashboard_embed_url(
                AwsAccountId=self.account_id,
                DashboardId=self.dashboard_id,
                IdentityType='IAM'
            )
            
            return response['EmbedUrl']
            
        except ClientError as e:
            logger.warning(f"Could not get embed URL: {str(e)}")
            # Return the standard URL
            return f"https://{self.region}.quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}"

    def check_refresh_status(self) -> Dict[str, Any]:
        """Check the refresh status of the dataset"""
        try:
            response = self.quicksight.list_ingestions(
                DataSetId=self.dataset_id,
                AwsAccountId=self.account_id
            )
            
            if response['Ingestions']:
                latest_ingestion = response['Ingestions'][0]
                return {
                    'ingestion_id': latest_ingestion['IngestionId'],
                    'ingestion_status': latest_ingestion['IngestionStatus'],
                    'created_time': latest_ingestion['CreatedTime'].isoformat(),
                    'ingestion_size_in_bytes': latest_ingestion.get('IngestionSizeInBytes', 0),
                    'row_info': latest_ingestion.get('RowInfo', {})
                }
            else:
                return {'status': 'no_ingestions_found'}
                
        except ClientError as e:
            logger.error(f"Failed to check refresh status: {str(e)}")
            return {'error': str(e)}

    def setup_complete_quicksight(self) -> Dict[str, Any]:
        """Complete QuickSight setup"""
        logger.info("Starting complete QuickSight setup...")
        
        results = {
            'setup_successful': True,
            'components_created': [],
            'errors': [],
            'dashboard_url': None
        }
        
        # Check QuickSight setup
        setup_check = self.check_quicksight_setup()
        if not setup_check.get('user_exists', False):
            results['errors'].append("QuickSight user not found - manual setup required")
            results['setup_successful'] = False
            return results
        
        # Create data source
        if self.create_data_source():
            results['components_created'].append('data_source')
        else:
            results['errors'].append('Failed to create data source')
            results['setup_successful'] = False
        
        # Create dataset
        if self.create_dataset():
            results['components_created'].append('dataset')
        else:
            results['errors'].append('Failed to create dataset')
            results['setup_successful'] = False
        
        # Create analysis
        if self.create_analysis():
            results['components_created'].append('analysis')
        else:
            results['errors'].append('Failed to create analysis')
            results['setup_successful'] = False
        
        # Create dashboard
        if self.create_dashboard():
            results['components_created'].append('dashboard')
            results['dashboard_url'] = self.get_dashboard_url()
        else:
            results['errors'].append('Failed to create dashboard')
            results['setup_successful'] = False
        
        # Setup auto-refresh
        if self.setup_auto_refresh():
            results['components_created'].append('auto_refresh')
        else:
            results['errors'].append('Failed to setup auto-refresh')
            # Don't fail the whole setup for this
        
        return results

def main():
    parser = argparse.ArgumentParser(description='QuickSight Setup for Bitcoin Prediction Pipeline')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    parser.add_argument('--account-id', help='AWS account ID (auto-detected if not provided)')
    parser.add_argument('--configure-datasource', action='store_true', help='Configure data source only')
    parser.add_argument('--create-dashboard', action='store_true', help='Create dashboard only')
    parser.add_argument('--check-refresh-status', action='store_true', help='Check refresh status')
    parser.add_argument('--setup-all', action='store_true', help='Complete setup (default)')
    
    args = parser.parse_args()
    
    # Default to complete setup if no specific action specified
    if not any([args.configure_datasource, args.create_dashboard, args.check_refresh_status]):
        args.setup_all = True
    
    # Initialize QuickSight setup
    qs_setup = QuickSightSetup(region=args.region, account_id=args.account_id)
    
    try:
        if args.configure_datasource:
            success = qs_setup.create_data_source()
            print(f"Data source configuration: {'SUCCESS' if success else 'FAILED'}")
        
        elif args.create_dashboard:
            success = qs_setup.create_dashboard()
            print(f"Dashboard creation: {'SUCCESS' if success else 'FAILED'}")
            if success:
                url = qs_setup.get_dashboard_url()
                print(f"Dashboard URL: {url}")
        
        elif args.check_refresh_status:
            status = qs_setup.check_refresh_status()
            print("Refresh Status:")
            print(json.dumps(status, indent=2, default=str))
        
        elif args.setup_all:
            results = qs_setup.setup_complete_quicksight()
            
            print("\n=== QuickSight Setup Results ===")
            print(f"Setup Successful: {results['setup_successful']}")
            print(f"Components Created: {', '.join(results['components_created'])}")
            
            if results['errors']:
                print(f"Errors: {', '.join(results['errors'])}")
            
            if results['dashboard_url']:
                print(f"Dashboard URL: {results['dashboard_url']}")
            
            print("\nSetup completed!")
            
            # Exit with appropriate code
            exit(0 if results['setup_successful'] else 1)
        
    except Exception as e:
        logger.error(f"QuickSight setup failed: {str(e)}")
        print(f"ERROR: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()