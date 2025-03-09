import os
import logging
import hashlib
import shlex
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
from google.cloud import storage
from google.api_core import retry
import requests
import shap
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess
import docker
from tenacity import retry, stop_after_attempt, wait_exponential
from threading import Thread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration Class
class Config:
    GCP_BUCKET = os.getenv("GCP_BUCKET", "zero-day-models")
    MODEL_FILE = os.getenv("MODEL_FILE", "xgb_zero_day_v3.json")
    VIRUS_TOTAL_KEY = os.getenv("VIRUSTOTAL_API_KEY")
    
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.enterprise.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    EMAIL_ADDRESS = os.getenv("ALERT_EMAIL")
    EMAIL_PASSWORD = os.getenv("EMAIL_CREDENTIAL")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

    THREAT_INTEL_ENABLED = os.getenv("THREAT_INTEL", "true").lower() == "true"
    SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "malware-analysis:v2.1")

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Global Model and Explainer Cache
model = None
explainer = None

def initialize_app():
    """Load ML model and initialize components on startup"""
    global model, explainer
    
    try:
        model = load_model()
        logger.info("✅ XGBoost model loaded successfully")
        
        # Initialize SHAP explainer with 100 background samples
        background = shap.maskers.Independent(np.random.rand(100, 3), max_samples=100)
        explainer = shap.Explainer(model, background)
        logger.info("✅ SHAP explainer initialized")
        
    except Exception as e:
        logger.critical(f"🚨 Critical initialization failure: {str(e)}")
        raise RuntimeError("Application initialization failed")

def load_model():
    """Load XGBoost model from GCP Cloud Storage with retries"""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _load_model():
        storage_client = storage.Client()
        bucket = storage_client.bucket(Config.GCP_BUCKET)
        blob = bucket.blob(Config.MODEL_FILE)
        local_path = f"/tmp/{Config.MODEL_FILE}"
        
        try:
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded model to {local_path}")
            return xgb.XGBClassifier().load_model(local_path)
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            os.remove(local_path)
            raise

    return _load_model()

# Feature Engineering
class FeatureEngineer:
    @staticmethod
    def sanitize_process_name(name: str) -> str:
        """Sanitize and normalize process names"""
        return name.lower().strip()[:256]
    
    @staticmethod
    def generate_features(name: str, cpu: float, memory: int) -> np.ndarray:
        """Create feature vector with anomaly detection"""
        sanitized = FeatureEngineer.sanitize_process_name(name)
        
        # Generate secure hash
        hash_digest = hashlib.sha256(sanitized.encode()).hexdigest()
        hash_int = int(hash_digest[:16], 16)  # Use first 16 chars for efficiency
        
        # Add statistical features
        return np.array([
            hash_int % (10**8),
            cpu,
            memory,
            np.log1p(cpu),
            np.sqrt(memory),
            (cpu > 90) * 1.0,  # High CPU flag
            (memory > 95) * 1.0  # High Memory flag
        ]).reshape(1, -1)

# Threat Intelligence Engine
class ThreatIntel:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def check_ip_reputation(ip: str) -> dict:
        """Check IP against multiple threat intelligence sources"""
        if not Config.THREAT_INTEL_ENABLED:
            return {"status": "disabled"}
            
        headers = {"x-apikey": Config.VIRUS_TOTAL_KEY}
        
        try:
            # VirusTotal Check
            vt_response = requests.get(
                f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
                headers=headers,
                timeout=5
            )
            vt_data = vt_response.json() if vt_response.status_code == 200 else {}
            
            # AbuseIPDB Check (example)
            # abuse_response = requests.get(...)
            
            return {
                "virustotal": vt_data.get("data", {}),
                # "abuseipdb": abuse_data,
                "last_checked": datetime.utcnow().isoformat()
            }
        except requests.exceptions.RequestException as e:
            logger.warning(f"Threat intel check failed: {str(e)}")
            return {"error": "Threat intelligence service unavailable"}

# Automated Response System
class ResponseSystem:
    @staticmethod
    def block_process(process_name: str):
        """Safely terminate suspicious processes"""
        sanitized = shlex.quote(FeatureEngineer.sanitize_process_name(process_name))
        
        try:
            # Check if process exists first
            check = subprocess.run(
                ["pgrep", "-f", sanitized],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if check.returncode == 0:
                subprocess.run(["pkill", "-9", "-f", sanitized], check=True)
                logger.info(f"Terminated process: {sanitized}")
                return True
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Process termination failed: {str(e)}")
            return False

    @staticmethod
    def block_ip_address(ip: str) -> bool:
        """Manage firewall rules safely"""
        sanitized = shlex.quote(ip)
        rule_exists = subprocess.run(
            ["iptables", "-C", "INPUT", "-s", sanitized, "-j", "DROP"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode == 0
        
        if not rule_exists:
            try:
                subprocess.run(
                    ["iptables", "-A", "INPUT", "-s", sanitized, "-j", "DROP"],
                    check=True
                )
                logger.info(f"Blocked IP: {sanitized}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"IP blocking failed: {str(e)}")
                return False
        return True

# Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400
            
        # Validate input parameters
        required = {"process_name", "cpu_usage", "memory_usage"}
        if missing := required - set(data.keys()):
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Extract and validate inputs
        process_name = FeatureEngineer.sanitize_process_name(data["process_name"])
        cpu = float(data["cpu_usage"])
        memory = int(data["memory_usage"])
        ip_address = data.get("source_ip")

        # Generate enhanced feature vector
        features = FeatureEngineer.generate_features(process_name, cpu, memory)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Generate explanation
        explanation = explainer(features).values.tolist() if explainer else []
        
        # Threat intelligence lookup
        threat_info = {}
        if ip_address and Config.THREAT_INTEL_ENABLED:
            threat_info = ThreatIntel.check_ip_reputation(ip_address)
        
        # Prepare response
        response = {
            "process": process_name,
            "prediction": "malicious" if prediction == 1 else "benign",
            "confidence": round(probability, 4),
            "threat_intel": threat_info,
            "analysis_id": hashlib.sha256(process_name.encode()).hexdigest()[:16]
        }
        
        # Take mitigation actions if malicious
        if prediction == 1:
            logger.warning(f"🚨 Detected malicious process: {process_name}")
            
            # Async mitigation actions
            Thread(target=execute_mitigation, args=(process_name, ip_address)).start()
            
            # Generate detailed report
            report = generate_incident_report(response, explanation)
            Thread(target=send_alert, args=("Zero-Day Detected", report)).start()

        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        return jsonify({"error": "Invalid input values"}), 400
    except Exception as e:
        logger.critical(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

def execute_mitigation(process_name: str, ip: str = None):
    """Execute containment measures in parallel"""
    # Process termination
    ResponseSystem.block_process(process_name)
    
    # Network containment
    if ip:
        ResponseSystem.block_ip_address(ip)
    
    # Sandbox analysis
    if docker_available():
        run_sandbox_analysis(process_name)

def docker_available() -> bool:
    """Check if Docker engine is accessible"""
    try:
        docker.from_env().ping()
        return True
    except docker.errors.DockerException:
        logger.warning("Docker engine unavailable")
        return False

def run_sandbox_analysis(process_name: str):
    """Execute suspicious process in isolated environment"""
    try:
        client = docker.from_env()
        container = client.containers.run(
            Config.SANDBOX_IMAGE,
            command=f"analyze {shlex.quote(process_name)}",
            detach=True,
            network_mode="none",
            mem_limit="512m"
        )
        logger.info(f"Started sandbox analysis: {container.id[:12]}")
    except docker.errors.ImageNotFound:
        logger.error(f"Sandbox image {Config.SANDBOX_IMAGE} not found")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {str(e)}")

def generate_incident_report(prediction: dict, explanation: list) -> str:
    """Generate detailed forensic report"""
    return f"""
    Zero-Day Incident Report
    ========================
    Timestamp: {datetime.utcnow().isoformat()}
    Process: {prediction['process']}
    Analysis ID: {prediction['analysis_id']}
    Confidence: {prediction['confidence']:.2%}
    Threat Indicators: {prediction['threat_intel']}
    
    Model Explanation:
    {explanation}

    Actions Taken:
    - Process termination attempted
    - Network containment implemented
    - Sandbox analysis initiated
    """

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def send_alert(subject: str, body: str):
    """Send encrypted email alert with error handling"""
    try:
        msg = MIMEMultipart()
        msg["From"] = Config.EMAIL_ADDRESS
        msg["To"] = Config.ADMIN_EMAIL
        msg["Subject"] = f"[ZDAY-ALERT] {subject}"
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Alert email dispatched successfully")
    except smtplib.SMTPException as e:
        logger.error(f"Email alert failed: {str(e)}")

if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)