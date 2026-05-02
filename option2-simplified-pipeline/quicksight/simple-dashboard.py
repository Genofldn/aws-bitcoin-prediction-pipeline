#!/usr/bin/env python3
"""
Simple QuickSight Setup for Option 2 (S3-based Bitcoin Predictions)

This script creates a basic QuickSight dashboard for the simplified pipeline.
"""

import boto3
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleQuickSightSetup:
    def __init__(self, region: str = 'eu-west-2', account_id: str = None):
        """Initialize Simple QuickSight setup"""
        self.region = region
        self.account_id = account_id or boto3.client('sts').get_caller_identity()['Account']
        
        self.quicksight = boto3.client('quicksight', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
        # Configuration
        self.namespace = 'default'
        self.data_source_id = 'bitcoin-simple-s3-datasource'
        self.dataset_id = 'bitcoin-simple-dataset'
        self.dashboard_id = 'bitcoin-simple-dashboard'
        self.s3_bucket = f'bitcoin-simple-predictions-{self.account_id}'

    def create_s3_data_source(self) -> bool:
        """Create S3 data source in QuickSight"""
        try:
            logger.info("Creating S3 data source...")
            
            # S3 data source configuration
            data_source_parameters = {
                'S3Parameters': {
                    'ManifestFileLocation': {
                        'Bucket': self.s3_bucket,
                        'Key': 'manifest/predictions-manifest.json'
                    }
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
                Name='Bitcoin Simple S3 Data Source',
                Type='S3',
                DataSourceParameters=data_source_parameters,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinSimple'},
                    {'Key': 'Environment', 'Value': 'Simple'}
                ]
            )
            
            logger.info(f"S3 data source created: {response['Arn']}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("S3 data source already exists")
                return True
            else:
                logger.error(f"Failed to create S3 data source: {str(e)}")
                return False

    def create_s3_manifest(self) -> bool:
        """Create S3 manifest file for QuickSight"""
        try:
            logger.info("Creating S3 manifest file...")
            
            # Create manifest for QuickSight to understand S3 data structure
            manifest = {
                "fileLocations": [
                    {
                        "URIPrefixes": [
                            f"s3://{self.s3_bucket}/predictions/"
                        ]
                    }
                ],
                "globalUploadSettings": {
                    "format": "JSON",
                    "delimiter": ","
                }
            }
            
            # Upload manifest to S3
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key='manifest/predictions-manifest.json',
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
            
            logger.info("S3 manifest created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create S3 manifest: {str(e)}")
            return False

    def create_simple_dataset(self) -> bool:
        """Create dataset for simple predictions"""
        try:
            logger.info("Creating simple dataset...")
            
            # Physical table configuration for S3
            physical_table_map = {
                'PredictionsTable': {
                    'S3Source': {
                        'DataSourceArn': f'arn:aws:quicksight:{self.region}:{self.account_id}:datasource/{self.data_source_id}',
                        'InputColumns': [
                            {'Name': 'timestamp', 'Type': 'STRING'},
                            {'Name': 'current_price', 'Type': 'DECIMAL'},
                            {'Name': 'prediction_1h', 'Type': 'DECIMAL'},
                            {'Name': 'prediction_6h', 'Type': 'DECIMAL'},
                            {'Name': 'prediction_24h', 'Type': 'DECIMAL'},
                            {'Name': 'prediction_7d', 'Type': 'DECIMAL'},
                            {'Name': 'confidence', 'Type': 'DECIMAL'},
                            {'Name': 'sentiment', 'Type': 'DECIMAL'},
                            {'Name': 'change_1h_pct', 'Type': 'DECIMAL'},
                            {'Name': 'change_24h_pct', 'Type': 'DECIMAL'},
                            {'Name': 'news_count', 'Type': 'INTEGER'}
                        ]
                    }
                }
            }
            
            # Logical table with transformations
            logical_table_map = {
                'ProcessedPredictions': {
                    'Alias': 'Bitcoin Simple Predictions',
                    'Source': {
                        'PhysicalTableId': 'PredictionsTable'
                    },
                    'DataTransforms': [
                        {
                            'CastColumnTypeOperation': {
                                'ColumnName': 'timestamp',
                                'NewColumnType': 'DATETIME',
                                'Format': 'yyyy-MM-ddTHH:mm:ssZ'
                            }
                        }
                    ]
                }
            }
            
            import_mode = 'SPICE'  # Use SPICE for better performance
            
            permissions = [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/{self.namespace}/quicksight-user-{self.account_id}',
                    'Actions': [
                        'quicksight:DescribeDataSet',
                        'quicksight:DescribeDataSetPermissions',
                        'quicksight:PassDataSet',
                        'quicksight:DescribeIngestion',
                        'quicksight:ListIngestions',
                        'quicksight:CreateIngestion'
                    ]
                }
            ]
            
            response = self.quicksight.create_data_set(
                AwsAccountId=self.account_id,
                DataSetId=self.dataset_id,
                Name='Bitcoin Simple Predictions Dataset',
                PhysicalTableMap=physical_table_map,
                LogicalTableMap=logical_table_map,
                ImportMode=import_mode,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinSimple'},
                    {'Key': 'Environment', 'Value': 'Simple'}
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

    def create_simple_dashboard(self) -> bool:
        """Create simple dashboard"""
        try:
            logger.info("Creating simple dashboard...")
            
            # Simple dashboard definition
            definition = {
                'DataSetIdentifierDeclarations': [
                    {
                        'DataSetArn': f'arn:aws:quicksight:{self.region}:{self.account_id}:dataset/{self.dataset_id}',
                        'Identifier': 'bitcoin_simple'
                    }
                ],
                'Sheets': [
                    {
                        'SheetId': 'sheet1',
                        'Name': 'Bitcoin Price & Predictions',
                        'Visuals': [
                            {
                                'LineChartVisual': {
                                    'VisualId': 'price_predictions_chart',
                                    'Title': {
                                        'Visibility': 'VISIBLE',
                                        'Text': 'Bitcoin Price vs Predictions'
                                    },
                                    'ChartConfiguration': {
                                        'FieldWells': {
                                            'LineChartAggregatedFieldWells': {
                                                'Category': [
                                                    {
                                                        'DateDimensionField': {
                                                            'FieldId': 'timestamp',
                                                            'Column': {
                                                                'DataSetIdentifier': 'bitcoin_simple',
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
                                                                'DataSetIdentifier': 'bitcoin_simple',
                                                                'ColumnName': 'current_price'
                                                            }
                                                        }
                                                    },
                                                    {
                                                        'NumericalMeasureField': {
                                                            'FieldId': 'prediction_24h',
                                                            'Column': {
                                                                'DataSetIdentifier': 'bitcoin_simple',
                                                                'ColumnName': 'prediction_24h'
                                                            }
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                'KPIVisual': {
                                    'VisualId': 'current_price_kpi',
                                    'Title': {
                                        'Visibility': 'VISIBLE',
                                        'Text': 'Current Bitcoin Price'
                                    },
                                    'ChartConfiguration': {
                                        'FieldWells': {
                                            'Values': [
                                                {
                                                    'NumericalMeasureField': {
                                                        'FieldId': 'current_price',
                                                        'Column': {
                                                            'DataSetIdentifier': 'bitcoin_simple',
                                                            'ColumnName': 'current_price'
                                                        },
                                                        'AggregationFunction': {
                                                            'SimpleNumericalAggregation': 'AVERAGE'
                                                        }
                                                    }
                                                }
                                            ]
                                        }
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
                        'quicksight:DescribeDashboard',
                        'quicksight:ListDashboardVersions',
                        'quicksight:UpdateDashboardPermissions',
                        'quicksight:QueryDashboard',
                        'quicksight:UpdateDashboard',
                        'quicksight:DeleteDashboard',
                        'quicksight:DescribeDashboardPermissions'
                    ]
                }
            ]
            
            response = self.quicksight.create_dashboard(
                AwsAccountId=self.account_id,
                DashboardId=self.dashboard_id,
                Name='Bitcoin Simple Dashboard',
                Definition=definition,
                Permissions=permissions,
                Tags=[
                    {'Key': 'Project', 'Value': 'BitcoinSimple'},
                    {'Key': 'Environment', 'Value': 'Simple'}
                ]
            )
            
            logger.info(f"Dashboard created: {response['Arn']}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.info("Dashboard already exists")
                return True
            else:
                logger.error(f"Failed to create dashboard: {str(e)}")
                return False

    def get_dashboard_url(self) -> str:
        """Get dashboard URL"""
        return f"https://{self.region}.quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}"

    def setup_complete_simple_quicksight(self) -> Dict[str, Any]:
        """Complete simple QuickSight setup"""
        logger.info("Starting simple QuickSight setup...")
        
        results = {
            'setup_successful': True,
            'components_created': [],
            'errors': [],
            'dashboard_url': None
        }
        
        # Create S3 manifest
        if self.create_s3_manifest():
            results['components_created'].append('s3_manifest')
        else:
            results['errors'].append('Failed to create S3 manifest')
        
        # Create S3 data source
        if self.create_s3_data_source():
            results['components_created'].append('s3_data_source')
        else:
            results['errors'].append('Failed to create S3 data source')
            results['setup_successful'] = False
        
        # Create dataset
        if self.create_simple_dataset():
            results['components_created'].append('dataset')
        else:
            results['errors'].append('Failed to create dataset')
            results['setup_successful'] = False
        
        # Create dashboard
        if self.create_simple_dashboard():
            results['components_created'].append('dashboard')
            results['dashboard_url'] = self.get_dashboard_url()
        else:
            results['errors'].append('Failed to create dashboard')
            results['setup_successful'] = False
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Simple QuickSight Setup for Bitcoin Predictions')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    parser.add_argument('--account-id', help='AWS account ID')
    parser.add_argument('--setup-all', action='store_true', help='Complete setup')
    
    args = parser.parse_args()
    
    if not args.setup_all:
        args.setup_all = True  # Default action
    
    # Initialize setup
    qs_setup = SimpleQuickSightSetup(region=args.region, account_id=args.account_id)
    
    try:
        if args.setup_all:
            results = qs_setup.setup_complete_simple_quicksight()
            
            print("\n=== Simple QuickSight Setup Results ===")
            print(f"Setup Successful: {results['setup_successful']}")
            print(f"Components Created: {', '.join(results['components_created'])}")
            
            if results['errors']:
                print(f"Errors: {', '.join(results['errors'])}")
            
            if results['dashboard_url']:
                print(f"Dashboard URL: {results['dashboard_url']}")
            
            print("\nNote: You may need to manually trigger a dataset refresh in QuickSight after data collection starts.")
            
            exit(0 if results['setup_successful'] else 1)
        
    except Exception as e:
        logger.error(f"QuickSight setup failed: {str(e)}")
        print(f"ERROR: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()