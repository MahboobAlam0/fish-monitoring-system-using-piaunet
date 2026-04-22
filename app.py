"""
Monitoring System - Zonal Density Analysis with Explainable AI
Physics-Informed Attention U-Net with Seg-Grad-CAM Explanations
"""

# ==================== ENVIRONMENT SETUP ====================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==================== IMPORTS ====================
import gradio as gr
import cv2
import torch
import tempfile
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

from config import (
    DEVICE,
    THRESHOLD,
    MIN_AREA,
    ZONAL_GRID_ROWS,
    ZONAL_GRID_COLS,
    ZONAL_LOW_THRESHOLD,
    ZONAL_MEDIUM_THRESHOLD,
    ZONAL_HIGH_THRESHOLD,
    ZONAL_ALERT_THRESHOLD,
    ZONAL_ENABLE_LOGGING,
    ZONAL_SHOW_ZONES,
    ZONAL_SHOW_HEATMAP,
    ZONAL_SHOW_SUMMARY
)

from models.piaunet_load import load_model
# from models.model1_load import load_model
from pipeline.inference import run_inference
from pipeline.mask_processing import MaskProcessor
from pipeline.density import ZonalDensityMonitor
from pipeline.visualization import ZonalVisualizer
from pipeline.post_processing import PostProcessor
from pipeline.integration import monitoring_session
from pipeline.gradcam import SegGradCAM
from pipeline.explain import explain_prediction
from pipeline.xai_visualization import create_xai_figure
from pipeline.temporal_gradcam import TemporalGradCAM
import numpy as np

# Production logging configuration
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Suppress the known Windows asyncio "WinError 10054 / connection forcibly closed" noise.
# This error fires whenever a browser tab closes a Gradio streaming connection mid-flight.
# It is benign — Gradio handles it internally — but it floods the console on Windows.
class _WindowsAsyncioFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "WinError 10054" not in msg and "ConnectionResetError" not in msg

logging.getLogger("asyncio").addFilter(_WindowsAsyncioFilter())

logger = logging.getLogger(__name__)
logger.info("="*60)
logger.info("Zonal Density Monitoring System - Starting")
logger.info(f"Device: {DEVICE}")
logger.info("="*60)

# Load model with error handling
try:
    device = DEVICE
    model = load_model("weights/best_model.pth", device)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise SystemExit(f"Fatal: Cannot start without model. {str(e)}")

def _warmup_model(model, device, height=512, width=512, num_runs=2):
    """Warm up model with dummy inference passes to avoid slow first frame"""
    try:
        dummy_frame = torch.zeros(1, 3, height, width, device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_frame)
        logger.info("[OK] Model warmup completed")
    except Exception:
        logger.warning("[WARN] Model warmup skipped")


# Initialize model with warmup
try:
    _warmup_model(model, device)
    pp = PostProcessor(log_level="ERROR")
    gradcam = SegGradCAM(model=model, target_layer_name="dec1", device=str(device))
    logger.info("[OK] All components initialized successfully (including Grad-CAM)")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise


def _save_session_metadata(session_id: str, metadata: Dict[str, Any]) -> None:
    """Save session metadata for audit trail"""
    try:
        meta_file = RESULTS_DIR / f"session_{session_id}_metadata.json"
        metadata['timestamp'] = datetime.now().isoformat()
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save metadata: {str(e)}")


def _generate_legend_html(legend_data: Dict[str, int]) -> str:
    """
    Generate professional HTML legend for zone summary display
    
    Args:
        legend_data: Dictionary with zone counts by level
    
    Returns:
        HTML string for legend display
    """
    html = """
    <div style="
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <span style="font-size: 24px; margin-right: 12px;">📊</span>
            <h3 style="margin: 0; color: #1e293b; font-size: 20px; font-weight: 700;">Zone Density Summary</h3>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr style="border-bottom: 2px solid #cbd5e1; background: #f8fafc;">
                <td style="padding: 12px 8px; font-weight: 700; color: #1e293b;">Total Zones</td>
                <td style="padding: 12px 8px; text-align: right; font-weight: 700; color: #3b82f6; font-size: 18px;">{total}</td>
            </tr>
            
            <tr style="border-bottom: 1px solid #e2e8f0; background: #f0fdf4;">
                <td style="padding: 12px 8px; color: #245e3b;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #10b981; border-radius: 3px; margin-right: 8px;"></span>
                    Low Density
                </td>
                <td style="padding: 12px 8px; text-align: right; font-weight: 600; color: #10b981;">{LOW}</td>
            </tr>
            
            <tr style="border-bottom: 1px solid #e2e8f0; background: #f0f9ff;">
                <td style="padding: 12px 8px; color: #164e63;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #06b6d4; border-radius: 3px; margin-right: 8px;"></span>
                    Low-Medium Density
                </td>
                <td style="padding: 12px 8px; text-align: right; font-weight: 600; color: #06b6d4;">{LOW-MEDIUM}</td>
            </tr>
            
            <tr style="border-bottom: 1px solid #e2e8f0; background: #fffbeb;">
                <td style="padding: 12px 8px; color: #78350f;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #f59e0b; border-radius: 3px; margin-right: 8px;"></span>
                    Medium-High Density
                </td>
                <td style="padding: 12px 8px; text-align: right; font-weight: 600; color: #f59e0b;">{MEDIUM-HIGH}</td>
            </tr>
            
            <tr style="background: #fef2f2;">
                <td style="padding: 12px 8px; color: #7c2d12;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 3px; margin-right: 8px;"></span>
                    High Density
                </td>
                <td style="padding: 12px 8px; text-align: right; font-weight: 600; color: #ef4444;">{HIGH}</td>
            </tr>
        </table>
        
        <div style="
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 13px;
            color: #64748b;
        ">
            <p style="margin: 0; line-height: 1.5;">
                ⓘ <strong>Interpretation:</strong> Each zone is color-coded based on detected fish density. 
                Higher density zones indicate greater biomass concentration in that area.
            </p>
        </div>
    </div>
    """.format(
        total=legend_data.get('total', 0),
        LOW=legend_data.get('LOW', 0),
        **{'LOW-MEDIUM': legend_data.get('LOW-MEDIUM', 0), 
           'MEDIUM-HIGH': legend_data.get('MEDIUM-HIGH', 0),
           'HIGH': legend_data.get('HIGH', 0)}
    )
    return html


def _get_input_preview(input_file: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Extract input preview for display.
    Returns (processed_input_for_display, file_type)
    """
    try:
        if input_file is None:
            return None, None
        
        input_path = Path(input_file)
        ext = input_path.suffix.lower()
        is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        
        if is_image:
            # Load and return image
            frame = cv2.imread(input_file)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame, "image"
        else:
            # Return video path
            return input_file, "video"
    except Exception as e:
        logger.warning(f"Failed to load input preview: {str(e)}")
    
    return None, None


def process_with_integrated_analysis(
    input_file,
    grid_rows: int,
    grid_cols: int,
    alert_threshold: str,
    show_heatmap: bool,
    show_zones: bool,
    show_summary: bool,
    show_summary_panel: bool,
    use_sliding_window: bool,
    enable_xai: bool,
    enable_video_xai: bool,
    video_sample_rate: int,
    video_max_frames: int
) -> Tuple[Optional[Any], Optional[Any], Optional[str], Optional[Any], str, str, Optional[Any], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Process image/video with zonal density monitoring + optional XAI explanation.
    
    Args:
        input_file: Input image or video file
        grid_rows: Number of zone grid rows
        grid_cols: Number of zone grid columns
        alert_threshold: Alert density threshold level
        show_heatmap: Display density heatmap
        show_zones: Display zone grid overlay
        show_summary: Display summary statistics
        show_summary_panel: Display summary panel
        use_sliding_window: Enable sliding window inference
        enable_xai: Generate Seg-Grad-CAM explanation (images only)
        enable_video_xai: Generate Temporal CAM for videos
        video_sample_rate: Frame sample rate for video XAI
        video_max_frames: Max frames for video XAI
    
    Returns:
        Tuple of (input_image, video_output, image_output, input_video, report, legend_html, 
                  xai_visualization, xai_text, xai_download_file, video_xai_output, video_xai_text)
    """
    # First, run the standard zonal density analysis
    input_image, video_output, image_output, input_video, report, legend_html = process_with_zonal_density(
        input_file, grid_rows, grid_cols, alert_threshold,
        show_heatmap, show_zones, show_summary, show_summary_panel, use_sliding_window
    )
    
    # Initialize XAI outputs with helpful default descriptions
    xai_visualization = None
    xai_text = "Spatial Grad-CAM applies to Images."
    xai_download_file = None
    video_xai_output = None
    video_xai_text = "Temporal CAM applies to Videos."
    
    # Process XAI capabilities based on input mapping
    if input_file is not None:
        try:
            if isinstance(input_file, dict):
                file_path = input_file.get("name", input_file)
            else:
                file_path = input_file
            
            input_path = Path(file_path)
            ext = input_path.suffix.lower()
            
            # Scenario A: Image is uploaded
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                video_xai_text = "Temporal CAM is not applicable for static images. Please upload a video."
                if enable_xai:
                    logger.info(f"[XAI] Generating explanation for image: {input_path.name}")
                    xai_visualization, xai_text = explain_prediction_image(file_path)
                else:
                    xai_text = "Image XAI is disabled. Check 'Enable Image Grad-CAM' to analyze."
                    
            # Scenario B: Video is uploaded
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                xai_text = "Spatial Grad-CAM is not applicable for videos. Please upload an image."
                if enable_video_xai:
                    logger.info(f"[VIDEO XAI] Generating temporal explanation for video: {input_path.name}")
                    video_xai_output, video_xai_text = explain_prediction_video(
                        file_path,
                        sample_rate=video_sample_rate,
                        max_frames=video_max_frames,
                        fps=5.0
                    )
                else:
                    video_xai_text = "Video XAI is disabled. Check 'Enable Video Temporal CAM' to analyze."
                    
        except Exception as e:
            error_msg = f"Error generating XAI explanation: {str(e)}"
            xai_text = error_msg
            video_xai_text = error_msg
            logger.error(f"[XAI] {error_msg}")
    
    return input_image, video_output, image_output, input_video, report, legend_html, xai_visualization, xai_text, str(xai_download_file) if xai_download_file else None, video_xai_output, video_xai_text


def process_with_zonal_density(
    input_file, 
    grid_rows: int, 
    grid_cols: int, 
    alert_threshold: str, 
    show_heatmap: bool, 
    show_zones: bool, 
    show_summary: bool,
    show_summary_panel: bool,
    use_sliding_window: bool
) -> Tuple[Optional[Any], Optional[Any], Optional[str], Optional[Any], str, str]:
    """
    Process image/video with zonal density monitoring.
    
    Args:
        input_file: Input image or video file
        grid_rows: Number of zone grid rows
        grid_cols: Number of zone grid columns
        alert_threshold: Alert density threshold level
        show_heatmap: Display density heatmap
        show_zones: Display zone grid overlay
        show_summary: Display summary statistics
        use_sliding_window: Enable sliding window inference
    
    Returns:
        Tuple of (input_image, processed_video, processed_image, input_video, report, legend_html)
    
    Raises:
        ValueError: On invalid inputs
    """
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    
    try:
        # Input validation
        if input_file is None:
            raise ValueError("No file uploaded. Please select an image or video file.")
        
        if isinstance(input_file, dict):
            input_file = input_file["name"]
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise ValueError(f"File not found: {input_file}")
        
        if not (grid_rows > 0 and grid_cols > 0):
            raise ValueError("Grid dimensions must be positive")
        
        ext = input_path.suffix.lower()
        is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        
        if not is_image and ext not in [".mp4", ".avi", ".mov", ".mkv"]:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"[{session_id}] Processing {'image' if is_image else 'video'}: {input_path.name}")
        
        # Initialize components
        mask_processor = MaskProcessor()
        zonal_monitor = ZonalDensityMonitor(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            low_threshold=ZONAL_LOW_THRESHOLD,
            medium_threshold=ZONAL_MEDIUM_THRESHOLD,
            high_threshold=ZONAL_HIGH_THRESHOLD,
            enable_logging=ZONAL_ENABLE_LOGGING
        )
        visualizer = ZonalVisualizer(zonal_monitor)
        
        # Get input preview for display
        input_preview, input_type = _get_input_preview(input_file)
        
        # ================= IMAGE PROCESSING ================= #
        if is_image:
            video_out, image_out, report, legend_html = _process_image(
                input_file, session_id, mask_processor, zonal_monitor, visualizer,
                alert_threshold, show_heatmap, show_zones, show_summary, show_summary_panel, use_sliding_window
            )
            return input_preview, video_out, image_out, None, report, legend_html
        
        # ================= VIDEO PROCESSING ================= #
        else:
            video_out, image_out, report, legend_html = _process_video(
                input_file, session_id, mask_processor, zonal_monitor, visualizer,
                alert_threshold, show_heatmap, show_zones, show_summary, show_summary_panel, use_sliding_window
            )
            return None, video_out, image_out, input_preview, report, legend_html
    
    except ValueError as e:
        error_msg = f"Validation Error: {str(e)}"
        logger.error(f"[{session_id}] {error_msg}")
        return None, None, None, None, error_msg, ""
    except Exception as e:
        error_msg = f"Processing Error: {str(e)}"
        logger.exception(f"[{session_id}] {error_msg}")
        return None, None, None, None, error_msg, ""
    finally:
        elapsed = time.time() - start_time
        logger.info(f"[{session_id}] Session completed in {elapsed:.1f}s")


def _process_image(
    input_file: str,
    session_id: str,
    mask_processor: MaskProcessor,
    zonal_monitor: ZonalDensityMonitor,
    visualizer: ZonalVisualizer,
    alert_threshold: str,
    show_heatmap: bool,
    show_zones: bool,
    show_summary: bool,
    show_summary_panel: bool,
    use_sliding_window: bool
) -> Tuple[None, Any, str, str]:
    """Process single image - returns (video_out, image_out, report, legend_html)"""
    try:
        frame = cv2.imread(input_file)
        if frame is None:
            raise ValueError(f"Failed to load image: {input_file}")

        h, w = frame.shape[:2]
        zonal_monitor.initialize_zones(h, w)

        # Run inference
        start_time = time.time()
        logits = run_inference(model, frame, device, use_sliding_window=use_sliding_window)
        mask, prob_map = mask_processor.process(logits, h, w, THRESHOLD, MIN_AREA, frame)
        
        # Analyze zones
        results = zonal_monitor.analyze_frame(mask)
        processing_time = (time.time() - start_time) * 1000
        
        # Validation & enrichment
        is_valid, issues = pp.validator.validate_zones_dict(results)
        if not is_valid:
            logger.warning(f"[{session_id}] Validation issues: {issues}")
        
        enriched = pp.processor.enrich_results(results, 0, processing_time)
        results = enriched.get('zones', results)
        
        # Health check
        pp.health_checker.check_frame_processing(0, processing_time, success=True)
        health = pp.health_checker.get_health_report()
        
        # Get alerts
        alerts = zonal_monitor.get_alerts(results, alert_threshold)
        
        if ZONAL_ENABLE_LOGGING:
            zonal_monitor.log_analysis(results, frame_num=0)

        # Get legend data BEFORE drawing anything on image
        legend_data = visualizer.get_legend_data(results)
        
        # Visualize (without summary panel drawn on image)
        frame_vis = visualizer.draw_full_analysis(
            frame, mask, results, alerts,
            show_zones=show_zones,
            show_heatmap=show_heatmap,
            show_summary=False,  # Don't draw summary on image
            show_alerts=show_zones,
            show_contours=False,
            show_labels=False,  # No zone labels - clean image
            show_zone_labels=False,
            show_summary_panel=False,  # Don't draw panel on image
            prob_map=prob_map
        )
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)

        # Generate report
        report = _generate_zonal_report(
            results, alerts, zonal_monitor.num_zones,
            processing_time, health, session_id
        )
        
        # Generate legend HTML
        legend_html = _generate_legend_html(legend_data)
        
        logger.info(f"[{session_id}] Image processed successfully - {len(alerts)} alerts")
        return None, frame_vis, report, legend_html

    except Exception:
        logger.exception(f"[{session_id}] Image processing failed")
        raise


def _process_video(
    input_file: str,
    session_id: str,
    mask_processor: MaskProcessor,
    zonal_monitor: ZonalDensityMonitor,
    visualizer: ZonalVisualizer,
    alert_threshold: str,
    show_heatmap: bool,
    show_zones: bool,
    show_summary: bool,
    show_summary_panel: bool,
    use_sliding_window: bool
) -> Tuple[str, None, str, str]:
    """Process video with frame-by-frame monitoring - returns (video_path, None, report, legend_html)"""
    cap = None
    out = None
    
    try:
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_file}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 or fps == 0:
            raise ValueError("Invalid video properties")

        zonal_monitor.initialize_zones(h, w)

        # Create output video
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(tmp.name, fourcc, max(1, fps), (w, h))

        frame_count = 0
        alert_frames = []
        high_density_zones = set()
        processed_frames = 0
        failed_frames = 0
        last_results = {}

        logger.info(f"[{session_id}] Starting video processing: {total_frames} frames @ {fps:.1f}fps")

        with monitoring_session(output_dir=RESULTS_DIR) as session:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Run inference
                    logits = run_inference(model, frame, device, use_sliding_window=use_sliding_window)
                    mask, prob_map = mask_processor.process(logits, h, w, THRESHOLD, MIN_AREA, frame)

                    # Analyze zones
                    results = zonal_monitor.analyze_frame(mask)
                    alerts = zonal_monitor.get_alerts(results, alert_threshold)
                    
                    # Recovery & enrichment
                    success, processed = session.process_frame_with_recovery(
                        frame, mask, results, frame_count, zonal_monitor.num_zones,
                        0, max_retries=2
                    )
                    
                    if success and 'zones' in processed:
                        results = processed['zones']
                        alerts = zonal_monitor.get_alerts(results, alert_threshold)
                        processed_frames += 1
                    else:
                        failed_frames += 1
                    
                    # Track alerts
                    if alerts:
                        alert_frames.append(frame_count)
                        high_density_zones.update(alerts)
                    
                    # Keep last results for legend
                    last_results = results

                    if ZONAL_ENABLE_LOGGING:
                        zonal_monitor.log_analysis(results, frame_num=frame_count)

                    # Visualize (without summary panel on video)
                    frame_vis = visualizer.draw_full_analysis(
                        frame, mask, results, alerts,
                        show_zones=show_zones,
                        show_heatmap=show_heatmap,
                        show_summary=False,  # Don't draw summary on video
                        show_alerts=show_zones,
                        show_contours=False,
                        show_labels=False,  # No zone labels - clean image
                        show_zone_labels=False,
                        show_summary_panel=False,  # Don't draw panel on video
                        prob_map=prob_map
                    )
                    out.write(frame_vis)

                except Exception:
                    failed_frames += 1
                    logger.warning(f"[{session_id}] Frame {frame_count}: Processing failed")

                frame_count += 1
                if frame_count % max(1, total_frames // 10) == 0:
                    logger.info(f"[{session_id}] Progress: {frame_count}/{total_frames}")

        cap.release()
        out.release()

        # Finalize & get health
        session.finalize_session()
        session_report = session.post_processor.health_checker.get_health_report()

        # Generate report
        report = _generate_zonal_video_report(
            zonal_monitor, alert_frames, high_density_zones,
            fps, total_frames, processed_frames, failed_frames,
            session_report, session_id
        )
        
        # Generate legend HTML from last frame results
        legend_data = visualizer.get_legend_data(last_results) if last_results else {'total': 0, 'LOW': 0, 'LOW-MEDIUM': 0, 'MEDIUM-HIGH': 0, 'HIGH': 0}
        legend_html = _generate_legend_html(legend_data)

        logger.info(f"[{session_id}] Video complete - {len(alert_frames)} alert frames")
        return tmp.name, None, report, legend_html

    except Exception:
        logger.exception(f"[{session_id}] Video processing failed")
        raise
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()


def _generate_zonal_report(
    results: Dict,
    alerts: list,
    num_zones: int,
    processing_time: float,
    health: Optional[Dict] = None,
    session_id: str = ""
) -> str:
    """Generate professional report for single image analysis"""
    
    report = "╔" + "═"*58 + "╗\n"
    report += "║  ZONAL DENSITY ANALYSIS REPORT - IMAGE                      ║\n"
    report += "╚" + "═"*58 + "╝\n\n"
    
    if session_id:
        report += f"Session ID: {session_id}\n"
        report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Processing metrics
    report += "PROCESSING METRICS\n"
    report += "-" * 40 + "\n"
    report += f"Processing Time: {processing_time:.1f}ms\n"
    report += f"Total Zones Analyzed: {num_zones}\n"
    report += "Status: Completed Successfully\n\n"
    
    # Zone-by-zone analysis
    report += "ZONE ANALYSIS\n"
    report += "-" * 40 + "\n"
    report += f"{'Zone ID':<8} {'Density Level':<15} {'Score':<12}\n"
    report += "-" * 40 + "\n"
    
    for zone_id, data in sorted(results.items()):
        level = data.get('level', 'UNKNOWN')
        score = data.get('score', 0.0)
        report += f"{zone_id:<8} {level:<15} {score:<12.4f}\n"
    
    report += "\n" + "="*40 + "\n\n"
    
    # Summary statistics
    level_counts = {}
    total_score = 0
    for data in results.values():
        level = data.get('level', 'UNKNOWN')
        level_counts[level] = level_counts.get(level, 0) + 1
        total_score += data.get('score', 0.0)
    
    avg_score = total_score / len(results) if results else 0
    
    report += "DENSITY DISTRIBUTION\n"
    report += "-" * 40 + "\n"
    report += f"Average Density: {avg_score:.4f}\n"
    report += f"Low Zones: {level_counts.get('LOW', 0)}\n"
    report += f"Low-Medium Zones: {level_counts.get('LOW-MEDIUM', 0)}\n"
    report += f"Medium-High Zones: {level_counts.get('MEDIUM-HIGH', 0)}\n"
    report += f"High Zones: {level_counts.get('HIGH', 0)}\n\n"
    
    # Alerts
    if alerts:
        report += "ALERTS TRIGGERED\n"
        report += "-" * 40 + "\n"
        report += f"Alert Zones: {sorted(alerts)}\n"
    else:
        report += "NO ALERTS\n"
        report += "-" * 40 + "\n"
        report += "All zones within normal parameters\n"
    
    report += "\n"
    
    if health:
        report += "="*40 + "\n"
        report += "SYSTEM HEALTH\n"
        report += "-" * 40 + "\n"
        report += f"Status: {'✓ HEALTHY' if health.get('is_healthy') else '⚠ ISSUES'}\n"
        report += f"Error Rate: {health.get('error_rate_percent', 0):.1f}%\n"
        
        if health.get('health_issues'):
            report += f"Issues: {health.get('health_issues')}\n"
    
    return report


def _generate_zonal_video_report(
    monitor: ZonalDensityMonitor,
    alert_frames: list,
    alert_zones: set,
    fps: float,
    total_frames: int,
    processed_frames: int,
    failed_frames: int,
    session_health: Optional[Dict] = None,
    session_id: str = ""
) -> str:
    """Generate comprehensive professional report for video analysis"""
    
    report = "╔" + "═"*58 + "╗\n"
    report += "║  ZONAL DENSITY ANALYSIS REPORT - VIDEO                     ║\n"
    report += "╚" + "═"*58 + "╝\n\n"
    
    if session_id:
        report += f"Session ID: {session_id}\n"
        report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Video specifications
    report += "VIDEO SPECIFICATIONS\n"
    report += "-" * 40 + "\n"
    report += f"Total Frames: {total_frames}\n"
    report += f"Frame Rate: {fps:.2f} fps\n"
    report += f"Duration: {total_frames/fps if fps > 0 else 0:.2f} seconds\n"
    report += f"Grid Size: {monitor.num_zones} zones\n\n"
    
    # Processing statistics
    report += "PROCESSING STATISTICS\n"
    report += "-" * 40 + "\n"
    report += f"Successfully Processed: {processed_frames}/{total_frames}\n"
    success_rate = (processed_frames/total_frames*100) if total_frames > 0 else 0
    report += f"Success Rate: {success_rate:.1f}%\n"
    
    if failed_frames > 0:
        report += f"Failed Frames: {failed_frames} ({failed_frames/total_frames*100:.1f}%)\n"
    
    report += f"Status: {'COMPLETE' if failed_frames == 0 else 'PARTIAL'}\n\n"
    
    # Alert statistics
    report += "ALERT SUMMARY\n"
    report += "-" * 40 + "\n"
    report += f"Alert Frames Detected: {len(alert_frames)}\n"
    
    if total_frames > 0:
        alert_rate = len(alert_frames)/total_frames*100
        report += f"Alert Rate: {alert_rate:.2f}%\n"
    
    if alert_zones:
        report += f"Zones with Alerts: {sorted(alert_zones)}\n"
    else:
        report += "Zones with Alerts: None\n"
    
    if alert_frames:
        report += f"Sample Alert Frames: {alert_frames[:min(10, len(alert_frames))]}\n"
        if len(alert_frames) > 10:
            report += f"... and {len(alert_frames) - 10} more alert frames\n"
    
    report += "\n" + "="*40 + "\n\n"
    
    # Per-zone statistics
    report += "ZONE STATISTICS\n"
    report += "-" * 40 + "\n"
    report += f"{'Zone':<6} {'Mean Density':<15} {'Max Density':<15}\n"
    report += "-" * 40 + "\n"
    
    for zone_id in range(monitor.num_zones):
        stats = monitor.get_zone_stats(zone_id)
        if stats:
            mean = stats.get('mean_density', 0)
            max_val = stats.get('max_density', 0)
            report += f"{zone_id:<6} {mean:<15.4f} {max_val:<15.4f}\n"
    
    report += "\n"
    
    # System health
    if session_health:
        report += "="*40 + "\n"
        report += "SYSTEM HEALTH\n"
        report += "-" * 40 + "\n"
        status = 'HEALTHY' if session_health.get('is_healthy') else '⚠ ISSUES'
        report += f"Status: {status}\n"
        report += f"Error Rate: {session_health.get('error_rate_percent', 0):.1f}%\n"
        
        if session_health.get('health_issues'):
            report += f"Issues: {session_health.get('health_issues')}\n"
    
    return report


# ==================== EXPLAINABLE AI FUNCTIONS ====================

def explain_prediction_image(input_file) -> Tuple[Any, str]:
    """
    Generate XAI explanation for a single image using Seg-Grad-CAM.
    
    Args:
        input_file: Path to input image
    
    Returns:
        Tuple of (visualization_figure, explanation_text)
    """
    if input_file is None:
        return None, "Error: No image provided. Please upload an image."
    
    try:
        input_path = Path(input_file)
        logger.info(f"[XAI] Processing image: {input_path}")
        
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            return None, f"Error: Cannot read image from {input_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        logger.info(f"[XAI] Image loaded: {image.shape}")
        
        # Resize to model input size
        input_size = (512, 512)
        image_resized = cv2.resize(image, input_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        logger.info(f"[XAI] Input tensor shape: {image_tensor.shape}, requires_grad: {image_tensor.requires_grad}")
        
        # Generate Grad-CAM
        logger.info("[XAI] Generating Grad-CAM heatmap")
        try:
            heatmap, weights, _ = gradcam(image_tensor, target_class=1)
            logger.info(f"[XAI] Heatmap generated: shape={heatmap.shape}, range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")
            
            if heatmap.max() == 0:
                logger.warning("[XAI] WARNING: Heatmap is all zeros! Gradient flow may have failed.")
            
        except Exception as e:
            logger.error(f"[XAI] Grad-CAM generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Grad-CAM failed: {str(e)}")
        
        # Get model predictions
        logger.info("[XAI] Getting model predictions")
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            seg_main = outputs[0]
        
        # Extract predictions and resize to original size
        seg_pred = torch.softmax(seg_main, dim=1)[0, 1].cpu().numpy()
        logger.info(f"[XAI] Segmentation range: [{seg_pred.min():.4f}, {seg_pred.max():.4f}]")
        
        seg_pred = cv2.resize(seg_pred, (w_orig, h_orig))
        heatmap = cv2.resize(heatmap, (w_orig, h_orig))
        
        image_original = cv2.resize(image_resized, (w_orig, h_orig))
        
        # Generate text explanation
        logger.info("[XAI] Generating text explanation")
        explanation = explain_prediction(
            heatmap=heatmap,
            seg_pred=seg_pred
        )
        
        # Create visualization figure
        logger.info("[XAI] Creating visualization figure")
        fig = create_xai_figure(
            input_image=image_original,
            seg_pred=seg_pred,
            cam_heatmap=heatmap,
            explanation_text=explanation,
            figsize=(16, 12)
        )
        
        # Save figure to temp file so it acts as an Image for Gradio
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(temp_file.name, bbox_inches='tight', dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        logger.info("[XAI] XAI explanation completed successfully")
        # Return: figure path (for display/download), explanation (text)
        return temp_file.name, explanation
        
    except Exception as e:
        logger.error(f"[XAI] Explanation error: {str(e)}", exc_info=True)
        return None, f"Error in XAI explanation: {str(e)}"


def explain_prediction_video(
    input_file: str,
    sample_rate: int = 2,
    max_frames: int = 30,
    fps: float = 5.0
) -> Tuple[Optional[str], str]:
    """
    Generate temporal XAI video using Temporal Grad-CAM.
    Video is written to a system temp file (NOT saved to results/).
    The OS cleans it up automatically after the session ends.
    """
    if input_file is None:
        return None, "Error: No video provided. Please upload a video."

    try:
        import tempfile, os
        input_path = Path(input_file)
        logger.info(f"[VIDEO XAI] Processing video: {input_path}")

        temporal_cam = TemporalGradCAM(
            gradcam_engine=gradcam,
            model=model,
            device=device
        )

        # Extract frames and generate CAM
        frames = temporal_cam.extract_frames(
            str(input_path), sample_rate=sample_rate, max_frames=max_frames
        )
        if not frames:
            return None, "Error: No frames could be extracted from the video."

        heatmaps, confidences, activations = temporal_cam.generate_temporal_cam(frames)

        # Write to temp file — NOT to results/, OS cleans it up automatically
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        temporal_cam.create_temporal_visualization(
            frames, heatmaps, confidences, tmp.name, fps=fps, cmap='jet'
        )

        stats_text = f"""TEMPORAL GRAD-CAM ANALYSIS RESULTS
====================================

📊 Video Statistics
-------------------
Total Frames Analyzed: {len(frames)}
Frame Sample Rate: Every {sample_rate} frames
Output FPS: {fps}

🎯 Model Confidence (Segmentation)
-----------------------------------
Mean Confidence: {float(np.mean(confidences)):.2%}
Max Confidence:  {float(np.max(confidences)):.2%}
Min Confidence:  {float(np.min(confidences)):.2%}

⚡ Attention Activation
------------------------
Mean CAM Activation: {float(np.mean(activations)):.4f}
Max CAM Activation:  {float(np.max(activations)):.4f}
Min CAM Activation:  {float(np.min(activations)):.4f}

📈 Frame-by-Frame Metrics
---------------------------
Frame | Confidence | CAM Activation
------|------------|---------------"""

        for idx, (conf, act) in enumerate(zip(confidences, activations), 1):
            stats_text += f"\n{idx:>5} | {conf:>9.2%} | {act:>14.4f}"

        stats_text += """

💡 Interpretation
------------------
✓ Mean Confidence > 80% = Model is very certain about fish detection
✓ High CAM Activation   = Strong attention focus on specific regions
✓ Consistent Confidence = Stable model predictions across frames
✓ Variable Activation   = Attention shifts based on fish location/visibility
"""
        logger.info(f"[VIDEO XAI] Analysis complete. Temp file: {tmp.name}")
        return tmp.name, stats_text

    except Exception as e:
        logger.error(f"[VIDEO XAI] Error: {str(e)}", exc_info=True)
        return None, f"Error in video XAI analysis: {str(e)}"



CSS_STYLE = """
/* Dark Minimalist Dashboard Theme Overrides */
body, .gradio-container {
    background-color: #0b111e !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container { max-width: 1400px !important; padding: 20px !important; border: none !important; box-shadow: none !important;}

/* Top Header */
.top-header {
    display: flex !important; align-items: center !important; padding: 16px 24px !important;
    background: #111827 !important; 
    border-bottom: 2px solid #0284c7 !important;
    margin-bottom: 24px !important;
    border-radius: 4px;
}
.header-title {
    font-size: 20px; font-weight: 700; color: #f8fafc; text-transform: uppercase;
    letter-spacing: 1px; display: flex; align-items: center; gap: 12px;
}
.header-stats {
    display: flex; gap: 20px; flex-grow: 1; justify-content: flex-end; align-items: center;
    font-size: 13px; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
}

/* Sidebar & General Layout */
.sidebar {
    background: #111827 !important; 
    border: 1px solid #1f2937 !important;
    padding: 24px !important; border-radius: 4px !important; margin-right: 12px !important;
}
.side-title {
    font-size: 13px; color: #38bdf8; letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 12px; font-weight: 700; padding-bottom: 4px;
    margin-top: 24px; border-bottom: 1px solid #1f2937;
}

.main-content { padding: 0 !important; }

/* Panels */
.panel-wrapper {
    background: #111827 !important; 
    border: 1px solid #1f2937 !important;
    border-radius: 4px !important; padding: 16px !important; margin-bottom: 16px !important;
}
.panel-header {
    font-size: 14px; font-weight: 700; color: #f1f5f9; margin-bottom: 16px;
    display: flex; align-items: center; justify-content: flex-start; text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #1f2937; padding-bottom: 8px;
}

/* Media Views (Zero padding, flat) */
.media-view {
    border: 1px solid #1f2937 !important; 
    background: #000000 !important;
    border-radius: 4px !important; overflow: hidden !important; width: 100% !important;
}

/* Buttons */
.submit-btn {
    background: #0369a1 !important; color: #fff !important;
    font-weight: 700 !important; border: none !important; border-radius: 4px !important;
    padding: 16px !important; font-size: 14px !important; text-transform: uppercase !important;
    transition: background 0.2s !important;
    width: 100% !important; margin-top: 20px !important; letter-spacing: 1px !important;
}
.submit-btn:hover { background: #0284c7 !important; }

/* Control overwrites */
label { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 12px !important;}
input[type="number"], select, textarea { background: #0b111e !important; border: 1px solid #1f2937 !important; border-radius: 4px !important; color: #e2e8f0 !important; }
input[type="number"]:focus, select:focus { border-color: #0284c7 !important; outline: none; }
.slider { accent-color: #0284c7 !important; }
"""

minimal_theme = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0b111e",
    body_text_color="#94a3b8",
    block_background_fill="#111827",
    block_border_width="1px",
    block_border_color="#1f2937",
    block_title_text_color="#f8fafc",
    block_title_text_weight="700",
    button_primary_background_fill="#0369a1",
    button_primary_text_color="#ffffff",
    panel_background_fill="#111827",
    slider_color="#0284c7"
)

with gr.Blocks(title="Fish Density Monitoring System", theme=minimal_theme, css=CSS_STYLE) as demo:
    # Professional Header
    gr.HTML("""
    <div class='top-header'>
       <div class='header-title'>Fish Density Monitoring System</div>
       <div class='header-stats'>
           <div>Zonal Segmentation & Biomass Analysis</div>
       </div>
    </div>
    """)
    
    with gr.Row():
        # ─── LEFT SIDEBAR: CONFIGURATION ───
        with gr.Column(scale=1, min_width=320, elem_classes="sidebar"):
            
            gr.HTML("<div class='side-title' style='margin-top: 5px;'>Media Input</div>")
            file_input = gr.File(label="Upload Image or Video", file_count="single", file_types=["image", "video"], type="filepath")
            
            gr.HTML("<div class='side-title'>Zone Configuration</div>")
            grid_rows = gr.Slider(minimum=1, maximum=10, step=1, value=ZONAL_GRID_ROWS, label="Grid Rows")
            grid_cols = gr.Slider(minimum=1, maximum=10, step=1, value=ZONAL_GRID_COLS, label="Grid Columns")
            alert_threshold = gr.Dropdown(choices=["LOW", "LOW-MEDIUM", "MEDIUM-HIGH", "HIGH"], value=ZONAL_ALERT_THRESHOLD, label="Alert Threshold")
            
            gr.HTML("<div class='side-title'>Inference Engine</div>")
            use_sliding_window = gr.Checkbox(value=True, label="Enable High-Res Sliding Window (Slower)")
            
            gr.HTML("<div class='side-title'>Visualization Options</div>")
            show_heatmap = gr.Checkbox(value=False, label="Show Density Heatmap")
            show_zones = gr.Checkbox(value=False, label="Show Zone Grid")
            show_summary = gr.Checkbox(value=ZONAL_SHOW_SUMMARY, label="Show Summary Stats Overlay")
            show_summary_panel = gr.Checkbox(value=True, label="Show Summary Panel Legend")
            
            gr.HTML("<div class='side-title'>Explainable AI Logic</div>")
            enable_xai = gr.Checkbox(value=True, label="Enable Image Grad-CAM")
            enable_video_xai = gr.Checkbox(value=False, label="Enable Video Temporal CAM")
            with gr.Row():
                video_sample_rate = gr.Slider(value=2, label="Video Sample Rate", minimum=1, maximum=10, step=1)
                video_max_frames = gr.Slider(value=30, label="Max Analysis Frames", minimum=5, maximum=120, step=5)
            
            process_button = gr.Button("Analyze Data", elem_classes="submit-btn")
            
        # ─── MAIN CONTENT: RESULTS ───
        with gr.Column(scale=2, elem_classes="main-content"):
            
            # Row 1: Inputs vs Outputs
            with gr.Row():
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Input Media</div>")
                    input_image_preview = gr.Image(label="", show_label=False, type="pil", interactive=False, elem_classes="media-view")
                    input_video_preview = gr.Video(label="", show_label=False, format="mp4", interactive=False, elem_classes="media-view")
                    
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Segmentation Output</div>")
                    image_output = gr.Image(label="", show_label=False, elem_classes="media-view")
                    video_output = gr.Video(label="", show_label=False, format="mp4", elem_classes="media-view")
                    
            # Row 2: Reports
            with gr.Row():
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Detailed Report</div>")
                    report_output = gr.Textbox(label="", show_label=False, lines=10, max_lines=20, interactive=False)
                    
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Zone Summary Legend</div>")
                    legend_output = gr.HTML(value="<p style='color:#64748b; font-style:italic;'>Awaiting analysis...</p>")
                    
            # Row 3: XAI Views
            with gr.Row():
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Explainable AI (Image)</div>")
                    xai_visualization = gr.Image(label="Heatmap", show_label=False, elem_classes="media-view")
                    xai_text = gr.Textbox(label="Insights", lines=5, interactive=False)
                    
                with gr.Column(elem_classes="panel-wrapper"):
                    gr.HTML("<div class='panel-header'>Explainable AI (Video)</div>")
                    video_xai_output = gr.Video(label="Temporal CAM", show_label=False, elem_classes="media-view")
                    video_xai_text = gr.Textbox(label="Temporal Analysis", lines=5, interactive=False)

    # Hidden Backend Data Hookups
    with gr.Row(visible=False):
        xai_dummy_download = gr.File()

    # Event handlers
    process_button.click(
        fn=process_with_integrated_analysis,
        inputs=[
            file_input, grid_rows, grid_cols, alert_threshold,
            show_heatmap, show_zones, show_summary,
            show_summary_panel, use_sliding_window, enable_xai, enable_video_xai,
            video_sample_rate, video_max_frames
        ],
        outputs=[
            input_image_preview, video_output, image_output, input_video_preview, 
            report_output, legend_output, xai_visualization, xai_text, xai_dummy_download,
            video_xai_output, video_xai_text
        ]
    )

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameters have been moved.*")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        quiet=False,
        css=CSS_STYLE
    )
    logger.info("Server terminated")