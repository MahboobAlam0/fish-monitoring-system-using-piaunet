DEVICE = "cuda"

IMG_SIZE = 256

THRESHOLD = 0.50       # Minimum hard floor (confidence margin in mask_processing.py is the real gate)
MIN_AREA = 150         # Minimum connected-component area — 150px keeps small fish at full resolution

IOU_THRESHOLD = 0.6
MAX_AGE = 30

# ROI margins (dynamic)
ROI_MARGIN_X = 50
ROI_MARGIN_Y = 100

# counting parameters
DIST_THRESHOLD = 70
MEMORY_SIZE = 100

# ========== ZONAL DENSITY MONITORING ==========
# Grid configuration
ZONAL_GRID_ROWS = 5          # Number of rows in zone grid
ZONAL_GRID_COLS = 5          # Number of columns in zone grid

# Density thresholds for zones (BALANCED for actual data distribution 0.15-1.0)
ZONAL_LOW_THRESHOLD = 0.25         # Below this: LOW density
ZONAL_MEDIUM_THRESHOLD = 0.45      # Between LOW and MEDIUM: LOW-MEDIUM
ZONAL_HIGH_THRESHOLD = 0.70        # Between MEDIUM and HIGH: MEDIUM-HIGH, above: HIGH

# Alert settings
ZONAL_ALERT_THRESHOLD = "MEDIUM-HIGH"   # Alert on MEDIUM-HIGH + HIGH zones
ZONAL_ENABLE_LOGGING = True       # Enable density logging to file

# Visualization settings
ZONAL_SHOW_ZONES = True          # Draw zone grid
ZONAL_SHOW_HEATMAP = True        # Draw density heatmap
ZONAL_SHOW_SUMMARY = True        # Show summary statistics
ZONAL_SHOW_ALERTS = True         # Highlight zones with alerts