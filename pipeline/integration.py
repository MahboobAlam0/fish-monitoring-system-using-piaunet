"""
Integration layer for post-processing with monitoring system
Demonstrates production-ready error handling and recovery
"""

import time
import cv2
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, Generator
from datetime import datetime
import uuid

from pipeline.post_processing import PostProcessor, PostProcessingError


class MonitoringSession:
    """Manages a monitoring session with post-processing"""
    
    def __init__(self, session_id: Optional[str] = None, output_dir: Path = Path("results")):
        """
        Initialize monitoring session
        
        Args:
            session_id: Unique session identifier (generated if not provided)
            output_dir: Directory for output files
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize post-processor
        self.post_processor = PostProcessor(output_dir=output_dir, log_level="INFO")
        self.logger = self.post_processor.logger
        
        # Session metadata
        self.start_time = datetime.now()
        self.frames_processed = 0
        self.frames_failed = 0
        self.video_writer = None
        self.recovery_count = 0
        
        self.logger.info(f"Session {self.session_id} started")
    
    def process_frame_with_recovery(self,
                                   frame: any, # type: ignore
                                   mask: any, # type: ignore
                                   analysis_results: dict,
                                   frame_number: int,
                                   total_zones: int,
                                   processing_time_ms: float,
                                   max_retries: int = 2) -> Tuple[bool, dict]:
        """
        Process frame with automatic error recovery
        
        Args:
            frame: Input frame
            mask: Analysis mask
            analysis_results: Zone analysis results
            frame_number: Frame number
            total_zones: Total zones
            processing_time_ms: Processing time
            max_retries: Maximum retry attempts
        
        Returns:
            Tuple of (success, processed_result)
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Attempt post-processing
                success, result = self.post_processor.process_frame_results(
                    analysis_results,
                    frame_number,
                    processing_time_ms,
                    total_zones
                )
                
                if success:
                    self.frames_processed += 1
                    return True, result
                else:
                    last_error = result.get('error', 'Unknown error')
                    self.logger.warning(f"Frame {frame_number} post-processing failed: {last_error}")
            
            except PostProcessingError as e:
                last_error = str(e)
                self.logger.error(f"PostProcessingError on frame {frame_number}: {e}")
            
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Unexpected error on frame {frame_number}: {e}", exc_info=True)
            
            retry_count += 1
            
            if retry_count <= max_retries:
                # Exponential backoff
                wait_time = 0.1 * (2 ** (retry_count - 1))
                self.logger.info(f"Retrying frame {frame_number} (attempt {retry_count}/{max_retries}) "
                               f"after {wait_time}s")
                time.sleep(wait_time)
                self.recovery_count += 1
        
        # All retries failed
        self.frames_failed += 1
        self.logger.error(f"Frame {frame_number} failed after {max_retries} retries: {last_error}")
        return False, {'error': last_error, 'frame': frame_number, 'retries_exhausted': True}
    
    def finalize_session(self) -> dict:
        """Finalize session and generate report"""
        try:
            # Calculate session duration
            duration = datetime.now() - self.start_time
            
            # Get session report
            report = self.post_processor.get_session_report()
            
            # Add session-level metrics
            report['session_id'] = self.session_id
            report['duration_seconds'] = duration.total_seconds()
            report['frames_processed'] = self.frames_processed
            report['frames_failed'] = self.frames_failed
            report['recovery_count'] = self.recovery_count
            report['success_rate'] = (
                self.frames_processed / (self.frames_processed + self.frames_failed) * 100
                if (self.frames_processed + self.frames_failed) > 0 else 0
            )
            
            # Save session
            success, msg = self.post_processor.save_session_results(self.session_id)
            if success:
                self.logger.info(msg)
            else:
                self.logger.error(f"Failed to save session: {msg}")
            
            report['saved'] = success
            report['save_message'] = msg
            
            self.logger.info(f"Session {self.session_id} finalized")
            return report
        
        except Exception as e:
            self.logger.error(f"Error finalizing session: {e}", exc_info=True)
            return {'error': str(e), 'session_id': self.session_id}
    
    def cleanup(self) -> None:
        """Cleanup session resources"""
        try:
            if self.video_writer:
                self.video_writer.release()
            
            self.logger.info(f"Session {self.session_id} cleaned up")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)


@contextmanager
def monitoring_session(output_dir: Path = Path("results")) -> Generator[MonitoringSession, None, None]:
    """
    Context manager for monitoring session
    Ensures proper cleanup even if errors occur
    
    Usage:
        with monitoring_session() as session:
            # Process frames
            success, result = session.process_frame_with_recovery(...)
    """
    session = MonitoringSession(output_dir=output_dir)
    try:
        yield session
    except Exception as e:
        session.logger.error(f"Session error: {e}", exc_info=True)
        raise
    finally:
        session.cleanup()
        report = session.finalize_session()
        print("\n" + "=" * 60)
        print("SESSION REPORT")
        print("=" * 60)
        print(f"Session ID: {report.get('session_id', 'N/A')}")
        print(f"Duration: {report.get('duration_seconds', 0):.2f}s")
        print(f"Frames Processed: {report.get('frames_processed', 0)}")
        print(f"Frames Failed: {report.get('frames_failed', 0)}")
        print(f"Success Rate: {report.get('success_rate', 0):.2f}%")
        print(f"Recovery Events: {report.get('recovery_count', 0)}")
        print("=" * 60)


class RobustVideoProcessor:
    """Production-grade video processor with comprehensive error handling"""
    
    def __init__(self, model, config, output_dir: Path = Path("results")):
        """Initialize robust processor"""
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("RobustVideoProcessor")
        
        # Import here to avoid circular imports
        from pipeline.inference import run_inference
        from pipeline.mask_processing import MaskProcessor
        
        self.model = model
        self.run_inference = run_inference
        self.mask_processor = MaskProcessor()
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> dict:
        """
        Process video with robust error handling
        
        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process (for testing)
        
        Returns:
            Processing result summary
        """
        with monitoring_session(output_dir=self.output_dir) as session:
            try:
                # Validate input
                video_file = Path(video_path)
                if not video_file.exists():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                
                session.logger.info(f"Starting video processing: {video_file}")
                
                # Open video
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    raise IOError(f"Cannot open video: {video_file}")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                session.logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
                
                # Setup output video
                output_video = self.output_dir / f"session_{session.session_id}_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
                out = None
                
                try:
                    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
                    if not out.isOpened():
                        session.logger.warning("Could not create output video writer")
                        out = None
                except Exception as e:
                    session.logger.warning(f"Failed to create video writer: {e}")
                    out = None
                
                # Process frames
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if max_frames and frame_count >= max_frames:
                        break
                    
                    # Time frame processing
                    frame_start = time.time()
                    
                    try:
                        # Run inference with error handling
                        logits = self.run_inference(self.model, frame, self.config.DEVICE)
                        mask, _ = self.mask_processor.process(
                            logits, height, width, 
                            self.config.THRESHOLD, 
                            self.config.MIN_AREA
                        )
                        
                        # Analyze zones
                        from pipeline.density import ZonalDensityMonitor
                        from pipeline.visualization import ZonalVisualizer
                        
                        monitor = ZonalDensityMonitor(
                            grid_rows=self.config.ZONAL_GRID_ROWS,
                            grid_cols=self.config.ZONAL_GRID_COLS,
                            enable_logging=self.config.ZONAL_ENABLE_LOGGING
                        )
                        monitor.initialize_zones(height, width)
                        visualizer = ZonalVisualizer(monitor)
                        
                        results = monitor.analyze_frame(mask)
                        alerts = monitor.get_alerts(results, self.config.ZONAL_ALERT_THRESHOLD)
                        
                        # Post-process with recovery
                        processing_time = (time.time() - frame_start) * 1000
                        success, processed = session.process_frame_with_recovery(
                            frame, mask, results,
                            frame_count,
                            monitor.num_zones,
                            processing_time,
                            max_retries=2
                        )
                        
                        # Visualize
                        if success or out:
                            try:
                                frame_vis = visualizer.draw_full_analysis(
                                    frame, mask, results, alerts if success else [],
                                    show_zones=self.config.ZONAL_SHOW_ZONES,
                                    show_heatmap=self.config.ZONAL_SHOW_HEATMAP,
                                    show_summary=self.config.ZONAL_SHOW_SUMMARY
                                )
                                
                                if out:
                                    out.write(frame_vis)
                            except Exception as e:
                                session.logger.error(f"Visualization error on frame {frame_count}: {e}")
                    
                    except Exception as e:
                        session.logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                    
                    frame_count += 1
                    
                    # Progress logging
                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        session.logger.info(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                # Cleanup
                cap.release()
                if out:
                    out.release()
                
                session.logger.info(f"Video processing completed: {frame_count} frames")
                
                return {
                    'success': True,
                    'frames_processed': frame_count,
                    'output_video': str(output_video),
                    'session_id': session.session_id
                }
            
            except Exception as e:
                session.logger.error(f"Video processing failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e),
                    'session_id': session.session_id
                }
