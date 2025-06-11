import os

# Base directory for the project (assuming this file is in the root or a known subdir)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Server settings
    PORT = int(os.getenv("PORT", 8000))  # FastAPI typically runs on 8000
    HOST = os.getenv("HOST", "0.0.0.0")

    # Model settings
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # --- Paths for H5 models ---
    H5_LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "best_bisindo_landmark_model.h5")
    H5_VIDEO_LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "best_bisindo_video_lstm_model.h5")
    # Hapus baris ini: H5_VIDEO_TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, "best_bisindo_video_transformer_model.h5")

    IMAGE_CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "image_class_mapping.json")
    VIDEO_CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "video_class_mapping.json")

    # Model parameters
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    NUM_FRAMES_VIDEO = 30
    NUM_LANDMARK_FEATURES = 21 * 3
    MODEL_CONFIDENCE_THRESHOLD = 0.5

    # Directories
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    PUBLIC_DIR = os.path.join(BASE_DIR, "public")

    # Upload settings
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png"]
    ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/quicktime"]

    # Realtime settings
    REALTIME_SESSION_TIMEOUT_MS = 60 * 60 * 1000
    REALTIME_STABLE_FRAME_THRESHOLD = 5
    REALTIME_MOVEMENT_THRESHOLD = 0.03
    REALTIME_MIN_SEQUENCE_FRAMES = 15
    REALTIME_CONFIDENCE_THRESHOLD = 0.5

    @staticmethod
    def create_dirs():
        """Ensures that necessary directories exist."""
        for path in [Config.UPLOADS_DIR, Config.TEMP_DIR, Config.MODELS_DIR, Config.PUBLIC_DIR]:
            os.makedirs(path, exist_ok=True)

# Call to create directories on import (or when app starts)
Config.create_dirs()