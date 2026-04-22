"""
Professional Post-Processing Pipeline
Ensures system reliability with comprehensive error handling, validation, and logging
"""

import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading
import traceback


class ProcessingStatus(Enum):
    """Processing status enumeration"""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PARTIAL = "partial"


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


class PostProcessingError(Exception):
    """Custom exception for post-processing failures"""
    pass


@dataclass
class ProcessingMetadata:
    """Metadata for processing results"""
    timestamp: str
    processing_time_ms: float
    status: str
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ZoneResult:
    """Validated zone analysis result"""
    zone_id: int
    density_score: float
    density_level: str
    pixel_count: int
    zone_area: int
    threshold: float
    confidence: float = 1.0  # 0-1 confidence in the result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def is_valid(self) -> Tuple[bool, str]:
        """Validate zone result"""
        issues = []
        
        if not 0 <= self.density_score <= 1:
            issues.append(f"Density score {self.density_score} out of range [0, 1]")
        
        if self.density_level not in ["LOW", "LOW-MEDIUM", "MEDIUM-HIGH", "HIGH"]:
            issues.append(f"Invalid density level: {self.density_level}")
        
        if self.pixel_count < 0:
            issues.append(f"Negative pixel count: {self.pixel_count}")
        
        if self.zone_area <= 0:
            issues.append(f"Invalid zone area: {self.zone_area}")
        
        if not 0 <= self.confidence <= 1:
            issues.append(f"Confidence {self.confidence} out of range [0, 1]")
        
        if self.pixel_count > self.zone_area:
            issues.append(f"Pixel count {self.pixel_count} exceeds zone area {self.zone_area}")
        
        return len(issues) == 0, "; ".join(issues) if issues else ""


class ResultValidator:
    """Validates analysis results before post-processing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validation_stats = {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
    
    def validate_zones_dict(self, results: Dict[int, Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate zones dictionary from analyzer
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.validation_stats['total_validations'] += 1
        issues = []
        
        if not isinstance(results, dict):
            issues.append(f"Results must be dict, got {type(results)}")
            self.logger.error(f"Invalid results type: {type(results)}")
            self.validation_stats['failed'] += 1
            return False, issues
        
        if not results:
            issues.append("Empty results dictionary")
            self.logger.warning("Empty results received")
            self.validation_stats['warnings'] += 1
            return False, issues
        
        # Validate each zone
        zone_issues = {}
        for zone_id, data in results.items():
            if not isinstance(zone_id, int):
                issues.append(f"Zone ID must be int, got {type(zone_id)}")
                continue
            
            if not isinstance(data, dict):
                issues.append(f"Zone {zone_id} data must be dict, got {type(data)}")
                continue
            
            # Check required keys
            required_keys = {'score', 'level', 'pixels', 'threshold'}
            missing = required_keys - set(data.keys())
            if missing:
                issues.append(f"Zone {zone_id} missing keys: {missing}")
                zone_issues[zone_id] = f"Missing keys: {missing}"
                continue
            
            # Validate score
            try:
                score = float(data['score'])
                if not 0 <= score <= 1:
                    issues.append(f"Zone {zone_id} score {score} out of range [0, 1]")
                    zone_issues[zone_id] = "Score out of range"
            except (ValueError, TypeError) as e:
                issues.append(f"Zone {zone_id} invalid score: {e}")
                zone_issues[zone_id] = f"Invalid score: {e}"
            
            # Validate level
            if data['level'] not in ["LOW", "LOW-MEDIUM", "MEDIUM-HIGH", "HIGH"]:
                issues.append(f"Zone {zone_id} invalid level: {data['level']}")
                zone_issues[zone_id] = "Invalid level"
            
            # Validate pixels
            try:
                pixels = int(data['pixels'])
                if pixels < 0:
                    issues.append(f"Zone {zone_id} negative pixels: {pixels}")
                    zone_issues[zone_id] = "Negative pixels"
            except (ValueError, TypeError) as e:
                issues.append(f"Zone {zone_id} invalid pixels: {e}")
                zone_issues[zone_id] = f"Invalid pixels: {e}"
        
        if issues:
            self.validation_stats['failed'] += 1
            return False, issues
        
        self.validation_stats['passed'] += 1
        return True, []
    
    def validate_alert_zones(self, alert_zones: List[int], total_zones: int) -> Tuple[bool, List[str]]:
        """Validate alert zones list"""
        issues = []
        
        if not isinstance(alert_zones, list):
            issues.append(f"Alert zones must be list, got {type(alert_zones)}")
            return False, issues
        
        for zone_id in alert_zones:
            if not isinstance(zone_id, int):
                issues.append(f"Zone ID must be int, got {type(zone_id)}")
            elif not 0 <= zone_id < total_zones:
                issues.append(f"Zone ID {zone_id} out of range [0, {total_zones-1}]")
        
        return len(issues) == 0, issues
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation statistics report"""
        total = self.validation_stats['total_validations']
        passed = self.validation_stats['passed']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': self.validation_stats['failed'],
            'warnings': self.validation_stats['warnings'],
            'success_rate': f"{success_rate:.2f}%"
        }


class ResultProcessor:
    """Processes and enriches analysis results"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.lock = threading.Lock()
        self.processing_history: List[Dict[str, Any]] = []
    
    def enrich_results(self, 
                      results: Dict[int, Dict[str, Any]], 
                      frame_number: int,
                      processing_time_ms: float) -> Dict[str, Any]:
        """
        Enrich results with additional metadata and analysis
        
        Args:
            results: Zone analysis results
            frame_number: Frame number being processed
            processing_time_ms: Time taken to process frame
        
        Returns:
            Enriched results dictionary
        """
        try:
            # Calculate aggregate statistics
            scores = [data['score'] for data in results.values()]
            levels = [data['level'] for data in results.values()]
            
            aggregate = {
                'total_zones': len(results),
                'avg_density': sum(scores) / len(scores) if scores else 0,
                'max_density': max(scores) if scores else 0,
                'min_density': min(scores) if scores else 0,
                'density_variance': self._calculate_variance(scores),
                'high_density_zones': sum(1 for level in levels if level == "HIGH"),
                'medium_high_zones': sum(1 for level in levels if level == "MEDIUM-HIGH"),
                'medium_low_zones': sum(1 for level in levels if level == "LOW-MEDIUM"),
                'low_zones': sum(1 for level in levels if level == "LOW"),
            }
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(aggregate, levels)
            
            # Enrich individual zone results
            enriched_results = {}
            for zone_id, data in results.items():
                enriched_results[zone_id] = {
                    **data,
                    'risk_flag': data['level'] in ["HIGH", "MEDIUM-HIGH"],
                    'deviation_from_mean': scores[zone_id] - aggregate['avg_density']
                    if zone_id < len(scores) else 0
                }
            
            return {
                'frame_number': frame_number,
                'timestamp': datetime.now().isoformat(),
                'zones': enriched_results,
                'aggregate': aggregate,
                'risk_level': risk_level,
                'processing_time_ms': processing_time_ms,
                'status': ProcessingStatus.SUCCESS.value
            }
        
        except Exception as e:
            self.logger.error(f"Error enriching results: {e}", exc_info=True)
            raise PostProcessingError(f"Failed to enrich results: {e}")
    
    @staticmethod
    def _calculate_variance(scores: List[float]) -> float:
        """Calculate variance in scores"""
        if not scores or len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance
    
    @staticmethod
    def _calculate_risk_level(aggregate: Dict[str, Any], levels: List[str]) -> str:
        """Calculate overall risk level"""
        if aggregate['high_density_zones'] > 0:
            return "CRITICAL"
        elif aggregate['medium_high_zones'] > len(levels) * 0.3:
            return "HIGH"
        elif aggregate['medium_high_zones'] > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def add_to_history(self, processed_result: Dict[str, Any]) -> None:
        """Add result to processing history"""
        with self.lock:
            self.processing_history.append(processed_result)


class OutputValidator:
    """Validates output files and formats"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate_video_output(self, video_path: Path) -> Tuple[bool, str]:
        """Validate output video file"""
        if not video_path.exists():
            return False, f"Video file does not exist: {video_path}"
        
        file_size = video_path.stat().st_size
        if file_size == 0:
            return False, f"Video file is empty: {video_path}"
        
        if file_size > 10 * 1024 * 1024 * 1024:  # 10 GB
            self.logger.warning(f"Video file very large: {file_size / 1024 / 1024 / 1024:.2f} GB")
        
        return True, "Video output valid"
    
    def validate_json_report(self, report_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate JSON report structure"""
        try:
            # Try to serialize to JSON
            json.dumps(report_data)
        except (TypeError, ValueError) as e:
            return False, f"Report not JSON serializable: {e}"
        
        # Check required fields
        required_fields = ['timestamp', 'status']
        missing = [f for f in required_fields if f not in report_data]
        if missing:
            return False, f"Missing required fields: {missing}"
        
        return True, "Report valid"


class ResultPersistence:
    """Handles persistence of analysis results"""
    
    def __init__(self, logger: logging.Logger, output_dir: Path = Path("results")):
        self.logger = logger
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def save_json_result(self, result: Dict[str, Any], filename: str) -> Tuple[bool, str]:
        """Save result as JSON"""
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Saved JSON result: {output_path}")
            return True, str(output_path)
        except Exception as e:
            self.logger.error(f"Failed to save JSON result: {e}", exc_info=True)
            return False, str(e)
    
    def save_pickle_result(self, result: Any, filename: str) -> Tuple[bool, str]:
        """Save result as pickle (for Python objects)"""
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            self.logger.info(f"Saved pickle result: {output_path}")
            return True, str(output_path)
        except Exception as e:
            self.logger.error(f"Failed to save pickle result: {e}", exc_info=True)
            return False, str(e)
    
    def load_json_result(self, filename: str) -> Tuple[bool, Dict[str, Any]]:
        """Load result from JSON"""
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'r') as f:
                result = json.load(f)
            self.logger.info(f"Loaded JSON result: {output_path}")
            return True, result
        except Exception as e:
            self.logger.error(f"Failed to load JSON result: {e}", exc_info=True)
            return False, {}


class HealthChecker:
    """Monitors system health and detects anomalies"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {
            'frames_processed': 0,
            'frames_with_errors': 0,
            'total_alerts': 0,
            'processing_times': [],
            'last_check': datetime.now()
        }
        self.lock = threading.Lock()
    
    def check_frame_processing(self, frame_number: int, processing_time_ms: float, 
                               success: bool) -> Dict[str, Any]:
        """Check health of frame processing"""
        with self.lock:
            self.metrics['frames_processed'] += 1
            if not success:
                self.metrics['frames_with_errors'] += 1
            self.metrics['processing_times'].append(processing_time_ms)
            
            # Keep only last 100 processing times
            if len(self.metrics['processing_times']) > 100:
                self.metrics['processing_times'] = self.metrics['processing_times'][-100:]
        
        # Check for anomalies
        issues = self._detect_anomalies()
        
        return {
            'frame_number': frame_number,
            'success': success,
            'processing_time_ms': processing_time_ms,
            'error_rate': self._get_error_rate(),
            'avg_processing_time_ms': sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) 
                                        if self.metrics['processing_times'] else 0,
            'health_issues': issues,
            'is_healthy': len(issues) == 0
        }
    
    def record_alert(self) -> None:
        """Record an alert event"""
        with self.lock:
            self.metrics['total_alerts'] += 1
    
    def _get_error_rate(self) -> float:
        """Get error rate percentage"""
        if self.metrics['frames_processed'] == 0:
            return 0.0
        return (self.metrics['frames_with_errors'] / self.metrics['frames_processed']) * 100
    
    def _detect_anomalies(self) -> List[str]:
        """Detect system anomalies"""
        issues = []
        
        # Check error rate
        error_rate = self._get_error_rate()
        if error_rate > 5:
            issues.append(f"High error rate: {error_rate:.2f}%")
        
        # Check processing time anomalies
        if len(self.metrics['processing_times']) >= 2:
            times = self.metrics['processing_times']
            avg = sum(times) / len(times)
            recent = times[-1]
            
            if recent > avg * 2:
                issues.append(f"Slow frame processing: {recent:.0f}ms (avg: {avg:.0f}ms)")
        
        return issues
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        with self.lock:
            issues = self._detect_anomalies()
            error_rate_val = self._get_error_rate()
            return {
                'total_frames_processed': self.metrics['frames_processed'],
                'frames_with_errors': self.metrics['frames_with_errors'],
                'error_rate': f"{error_rate_val:.2f}%",
                'error_rate_percent': error_rate_val,
                'total_alerts': self.metrics['total_alerts'],
                'avg_processing_time_ms': sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) 
                                           if self.metrics['processing_times'] else 0,
                'last_check': self.metrics['last_check'].isoformat(),
                'health_issues': issues,
                'is_healthy': len(issues) == 0
            }


class PostProcessor:
    """Main post-processor orchestrating all validations and enrichment"""
    
    def __init__(self, output_dir: Path = Path("results"), log_level: str = "INFO"):
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize components
        self.validator = ResultValidator(self.logger)
        self.processor = ResultProcessor(self.logger)
        self.output_validator = OutputValidator(self.logger)
        self.persistence = ResultPersistence(self.logger, output_dir)
        self.health_checker = HealthChecker(self.logger)
        
        self.logger.info("PostProcessor initialized successfully")
    
    @staticmethod
    def _setup_logging(log_level: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("PostProcessor")
        logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # File handler
        log_file = Path("results") / "postprocessing.log"
        Path("results").mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def process_frame_results(self, 
                             results: Dict[int, Dict[str, Any]],
                             frame_number: int,
                             processing_time_ms: float,
                             total_zones: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Process frame results with full validation and enrichment
        
        Args:
            results: Zone analysis results
            frame_number: Current frame number
            processing_time_ms: Time to process frame
            total_zones: Total number of zones
        
        Returns:
            Tuple of (success, processed_result)
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Validate input
            is_valid, issues = self.validator.validate_zones_dict(results)
            if not is_valid:
                self.logger.warning(f"Frame {frame_number} validation failed: {issues}")
                self.health_checker.check_frame_processing(frame_number, processing_time_ms, False)
                return False, {'error': 'Validation failed', 'issues': issues}
            
            # Step 2: Process and enrich
            enriched = self.processor.enrich_results(results, frame_number, processing_time_ms)
            
            # Step 3: Extract and validate alerts
            alert_zones = [zid for zid, data in results.items() if data['level'] == "HIGH"]
            is_valid, alert_issues = self.validator.validate_alert_zones(alert_zones, total_zones)
            if not is_valid:
                self.logger.warning(f"Alert validation failed: {alert_issues}")
            
            enriched['alerts'] = alert_zones
            
            # Step 4: Health check
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            health_status = self.health_checker.check_frame_processing(
                frame_number, elapsed_ms, True
            )
            enriched['health'] = health_status
            
            if alert_zones:
                self.health_checker.record_alert()
            
            # Step 5: Add to history
            self.processor.add_to_history(enriched)
            
            self.logger.debug(f"Frame {frame_number} processed successfully")
            return True, enriched
        
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_number}: {e}", exc_info=True)
            self.health_checker.check_frame_processing(frame_number, processing_time_ms, False)
            return False, {'error': str(e), 'traceback': traceback.format_exc()}
    
    def get_session_report(self) -> Dict[str, Any]:
        """Get comprehensive session report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'validation_stats': self.validator.get_validation_report(),
            'health_status': self.health_checker.get_health_report(),
            'processed_frames': len(self.processor.processing_history)
        }
    
    def save_session_results(self, session_id: str) -> Tuple[bool, str]:
        """Save all session results"""
        try:
            report = self.get_session_report()
            
            # Save report
            filename = f"session_{session_id}_report.json"
            success, path = self.persistence.save_json_result(report, filename)
            if success:
                self.logger.info(f"Session report saved: {path}")
            
            # Save processing history
            history_filename = f"session_{session_id}_history.pkl"
            success, path = self.persistence.save_pickle_result(
                self.processor.processing_history,
                history_filename
            )
            if success:
                self.logger.info(f"Processing history saved: {path}")
            
            return True, f"Session {session_id} saved successfully"
        
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}", exc_info=True)
            return False, str(e)
