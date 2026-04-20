pipeline {
    agent any

    environment {
        PYTHON          = "python3"
        VENV_DIR        = ".venv"
        MLFLOW_TRACKING_URI = "http://localhost:5001"
        DOCKER_IMAGE_API = "credit-risk-api"
        DOCKER_IMAGE_TAG = "${env.BUILD_NUMBER}"
        REPORTS_DIR     = "reports"
    }

    options {
        timeout(time: 90, unit: "MINUTES")
        buildDiscarder(logRotator(numToKeepStr: "10"))
        timestamps()
    }

    stages {

        stage("Checkout") {
            steps {
                checkout scm
            }
        }

        stage("Setup") {
            steps {
                sh """
                    ${PYTHON} -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-dev.txt
                    mkdir -p ${REPORTS_DIR}
                """
            }
        }

        stage("Lint & Format Check") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    ruff check src/ scripts/ tests/ api/ app/
                    black --check src/ scripts/ tests/ api/ app/
                """
            }
        }

        stage("Unit Tests") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    pytest tests/ -v \
                      --cov=src --cov=api \
                      --cov-report=xml:reports/coverage.xml \
                      --cov-report=term-missing \
                      --cov-fail-under=80 \
                      --junit-xml=reports/junit.xml
                """
            }
            post {
                always {
                    junit "reports/junit.xml"
                    publishCoverage adapters: [coberturaAdapter("reports/coverage.xml")]
                }
            }
        }

        stage("Data Validation") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    python scripts/validate_data.py --params params.yaml
                """
            }
        }

        stage("DVC Reproduce") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    dvc repro
                """
            }
        }

        stage("AUC Gate") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    python scripts/check_auc_gate.py \
                      --params  params.yaml \
                      --metrics ${REPORTS_DIR}/metrics.json
                """
            }
        }

        stage("Build API Image") {
            steps {
                sh """
                    docker build \
                      -f docker/Dockerfile.api \
                      -t ${DOCKER_IMAGE_API}:${DOCKER_IMAGE_TAG} \
                      -t ${DOCKER_IMAGE_API}:latest \
                      .
                """
            }
        }

        stage("Deploy (Staging)") {
            steps {
                sh """
                    docker compose up -d api mlflow prometheus grafana
                """
            }
        }

        stage("Smoke Test") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    python scripts/smoke_test.py \
                      --base-url http://localhost:8000 \
                      --retries 15 \
                      --delay 5
                """
            }
        }

        stage("Drift Detection") {
            steps {
                sh """
                    . ${VENV_DIR}/bin/activate
                    python monitoring/drift_detector.py \
                      --reference data/processed/train.parquet \
                      --current   data/processed/test.parquet \
                      --output    ${REPORTS_DIR}/drift_report.html
                """
            }
            post {
                always {
                    publishHTML(target: [
                        allowMissing: true,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: "${REPORTS_DIR}",
                        reportFiles: "drift_report.html",
                        reportName: "Evidently Drift Report"
                    ])
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline PASSED — build #${env.BUILD_NUMBER}"
        }
        failure {
            echo "Pipeline FAILED — check logs above"
            sh "docker compose down || true"
        }
        always {
            archiveArtifacts artifacts: "reports/**/*", allowEmptyArchive: true
        }
    }
}
