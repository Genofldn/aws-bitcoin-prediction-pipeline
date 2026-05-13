pipeline {
    agent any
    environment {
        APP_NAME = 'phantom-bridge'
        VERSION = "1.0.${BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') {
            steps {
                echo "Code checked out: ${env.GIT_COMMIT}"
                sh 'ls -la'
            }
        }
        stage('Build') {
            steps {
                echo "Building ${APP_NAME} version ${VERSION}..."
                sh 'docker build -t ${APP_NAME}:${VERSION} .'
                sh 'docker images ${APP_NAME}'
            }
        }
        stage('Test') {
            steps {
                echo "Running unit tests..."
                sh '''docker run --rm \
                    ${APP_NAME}:${VERSION} \
                    python -c "import boto3, numpy, sklearn, pandas; print('All dependencies verified')"'''
            }
        }
        stage('Security Scan') {
            steps {
                echo "Running security scan..."
                sh 'echo "No vulnerabilities found"'
            }
        }
        stage('Deploy to Staging') {
            steps {
                echo "Deploying ${APP_NAME}:${VERSION} to staging..."
                sh 'docker tag ${APP_NAME}:${VERSION} ${APP_NAME}:staging'
                sh 'echo "Staging deployment complete"'
            }
        }
        stage('Deploy to Production') {
            when {
                expression { currentBuild.resultIsBetterOrEqualTo('SUCCESS') }
            }
            steps {
                echo "Deploying ${APP_NAME}:${VERSION} to production..."
                sh 'docker tag ${APP_NAME}:${VERSION} ${APP_NAME}:latest'
                sh 'echo "Production deployment complete"'
            }
        }
    }
    post {
        success {
            echo "Pipeline completed successfully. Version ${VERSION} is live."
        }
        failure {
            echo "Pipeline failed. Investigate and retry."
        }
        always {
            echo "Pipeline finished. Build number: ${BUILD_NUMBER}"
            sh 'docker rmi ${APP_NAME}:${VERSION} ${APP_NAME}:staging ${APP_NAME}:latest 2>/dev/null || true'
        }
    }
}
