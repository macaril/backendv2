import os
import json
import time
import asyncio
from typing import List, Dict, Union, Any, Optional

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, backend
import cv2
from uuid import uuid4
import shutil
import aiofiles
import traceback

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config

# --- Logger (simple for now, can be replaced by a proper logging library) ---
def log_info(message: str, meta: Optional[Dict] = None):
    print(f"[INFO] {message}", meta if meta else "")

def log_error(message: str, error: Union[Exception, Dict] = None):
    meta = {"message": str(error), "stack": traceback.format_exc()} if isinstance(error, Exception) else error
    print(f"[ERROR] {message}", meta if meta else "")

# --- Custom Layers (PositionalEncoding dihapus karena Transformer dihapus) ---

# --- Model Handler ---
class ModelHandler:
    def __init__(self):
        self.static_sign_model = None
        self.video_lstm_model = None
        self.video_transformer_model = None # Ini akan tetap None
        self.image_class_mapping = {}
        self.video_class_mapping = {}
        self.models_loaded = False

    async def load_models(self):
        log_info('Loading TensorFlow models and class mappings...')
        try:
            # Helper function to load model, ONLY trying H5
            def _load_model_h5_only(h5_path, model_name, custom_objects=None):
                model = None
                if os.path.exists(h5_path):
                    log_info(f"Attempting to load {model_name} from H5: {h5_path}")
                    try:
                        # custom_objects di sini tidak diperlukan untuk LSTM, tapi biarkan saja untuk konsistensi API
                        model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
                        log_info(f"{model_name} loaded successfully from H5.")
                        return model
                    except Exception as e:
                        log_error(f"Failed to load {model_name} from H5 at {h5_path}. Error: {e}.", e)
                        raise FileNotFoundError(f"Failed to load {model_name} from H5: {e}")
                else:
                    raise FileNotFoundError(f"H5 model file not found for {model_name} at {h5_path}.")
                
                return model # Should not be reached

            # Load Static Sign Model (Landmark Model)
            self.static_sign_model = _load_model_h5_only(
                Config.H5_LANDMARK_MODEL_PATH,
                "Static Sign Model (Landmark Model)"
            )

            # Load Video LSTM Model (model kata utama sekarang)
            self.video_lstm_model = _load_model_h5_only(
                Config.H5_VIDEO_LSTM_MODEL_PATH,
                "Video LSTM Model"
            )

            # Bagian untuk memuat Video Transformer Model sudah dihapus

            # Load Class Mappings from JSON files
            with open(Config.IMAGE_CLASS_MAPPING_PATH, 'r') as f:
                self.image_class_mapping = {int(k): v for k, v in json.load(f).items()}
            log_info('Image Class Mapping loaded successfully.')

            with open(Config.VIDEO_CLASS_MAPPING_PATH, 'r') as f:
                self.video_class_mapping = {int(k): v for k, v in json.load(f).items()}
            log_info('Video Class Mapping loaded successfully.')
            
            self.models_loaded = True
            return {
                "landmark_model": bool(self.static_sign_model),
                "video_lstm_model": bool(self.video_lstm_model),
                "video_transformer_model": False, # Set menjadi False karena tidak dimuat
                "image_class_mapping": bool(self.image_class_mapping),
                "video_class_mapping": bool(self.video_class_mapping)
            }
        except Exception as e:
            log_error('Error loading models or class mappings:', e)
            self.models_loaded = False
            raise # Re-raise the exception to indicate failure to load models

    def get_model_status(self) -> Dict[str, bool]:
        # Sesuaikan status model di sini
        return {
            "landmark_model": self.static_sign_model is not None,
            "video_lstm_model": self.video_lstm_model is not None,
            "video_transformer_model": False, # Selalu False
            "image_class_mapping": bool(self.image_class_mapping),
            "video_class_mapping": bool(self.video_class_mapping)
        }

    async def predict_static_sign(self, landmarks: List[float]) -> Dict[str, Union[str, float, int, None]]:
        if not self.static_sign_model:
            log_error("Static sign model not loaded for prediction.")
            raise HTTPException(status_code=500, detail="Static sign model not loaded.")
        if not landmarks or len(landmarks) != Config.NUM_LANDMARK_FEATURES:
            raise HTTPException(status_code=400, detail=f"Invalid landmark array length. Expected {Config.NUM_LANDMARK_FEATURES}, got {len(landmarks)}.")

        try:
            input_tensor = tf.constant([landmarks], dtype=tf.float32)
            predictions = self.static_sign_model.predict(input_tensor, verbose=0)
            probabilities = tf.nn.softmax(predictions).numpy()[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = self.image_class_mapping.get(predicted_index, "Unknown")
            confidence = float(probabilities[predicted_index])

            log_info(f"Static sign prediction: Class={predicted_class}, Confidence={confidence:.4f}")

            return {
                "class": predicted_class,
                "confidence": confidence,
                "index": int(predicted_index)
            }
        except Exception as e:
            log_error("Error in predict_static_sign:", e)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    async def predict_dynamic_sign(self, landmark_sequence: List[List[float]], model_choice: str = 'lstm') -> Dict[str, Union[str, float, int, None]]:
        # Hapus pilihan model, hanya gunakan LSTM
        model_to_use = self.video_lstm_model

        if not model_to_use:
            log_error(f"Video LSTM model not loaded for prediction.")
            raise HTTPException(status_code=500, detail=f"Video LSTM model not loaded.")

        if not landmark_sequence or len(landmark_sequence) == 0:
            raise HTTPException(status_code=400, detail='Landmark sequence is empty.')

        # Preprocessing: Pad or truncate sequence
        num_frames_expected = Config.NUM_FRAMES_VIDEO
        num_features = Config.NUM_LANDMARK_FEATURES

        processed_sequence = list(landmark_sequence)

        if len(processed_sequence) > num_frames_expected:
            processed_sequence = processed_sequence[len(processed_sequence) - num_frames_expected:]
        elif len(processed_sequence) < num_frames_expected:
            padding = [0.0] * num_features
            while len(processed_sequence) < num_frames_expected:
                processed_sequence.append(padding)
        
        processed_sequence = [
            (frame[:num_features] + [0.0] * (num_features - len(frame))) if len(frame) != num_features
            else frame
            for frame in processed_sequence
        ]
        
        try:
            input_tensor = tf.constant([processed_sequence], dtype=tf.float32)
            predictions = model_to_use.predict(input_tensor, verbose=0)
            probabilities = tf.nn.softmax(predictions).numpy()[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = self.video_class_mapping.get(predicted_index, "Unknown")
            confidence = float(probabilities[predicted_index])

            log_info(f"Dynamic sign prediction (LSTM): Class={predicted_class}, Confidence={confidence:.4f}")

            return {
                "class": predicted_class,
                "confidence": confidence,
                "index": int(predicted_index),
                "modelUsed": "lstm" # Selalu laporkan lstm
            }
        except Exception as e:
            log_error("Error in predict_dynamic_sign:", e)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    def text_to_sign(self, text: str) -> Dict[str, Any]:
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        words = text.strip().lower().split()
        result_signs = []

        for word in words:
            is_known_word = False
            mapped_word = word
            
            # Check if word exists in video class mapping (known words)
            for k, v in self.video_class_mapping.items():
                if v.lower() == word:
                    is_known_word = True
                    mapped_word = v
                    break
            
            if is_known_word:
                result_signs.append({
                    "type": "word",
                    "original": word,
                    "mapped": mapped_word,
                    "knownInDataset": True
                })
            else:
                letters = []
                for letter in word:
                    letter_exists = False
                    mapped_letter = letter.upper()
                    
                    for k, v in self.image_class_mapping.items():
                        if v.lower() == letter.lower():
                            letter_exists = True
                            mapped_letter = v
                            break
                    
                    letters.append({
                        "letter": letter,
                        "mapped": mapped_letter,
                        "exists": letter_exists
                    })
                
                result_signs.append({
                    "type": "fingerspell",
                    "original": word,
                    "letters": letters
                })
        
        return {
            "text": text,
            "signs": result_signs
        }

    def get_available_words(self) -> List[Dict[str, Union[int, str]]]:
        return [{"id": k, "word": v} for k, v in self.video_class_mapping.items()]

    def get_available_letters(self) -> List[Dict[str, Union[int, str]]]:
        return [{"id": k, "letter": v} for k, v in self.image_class_mapping.items()]

# Instansiasi ModelHandler DI SINI (SETELAH DEFINISI KELASNYA)
model_handler = ModelHandler()


# --- Upload Handler ---
class UploadHandler:
    async def handle_file_upload(self, file: UploadFile, file_type: str = 'unknown') -> Dict[str, Any]:
        try:
            file_extension = os.path.splitext(file.filename)[1].lower()
            mime_type = file.content_type

            is_valid_type = False
            if file_type == 'image':
                is_valid_type = mime_type in Config.ALLOWED_IMAGE_TYPES
            elif file_type == 'video':
                is_valid_type = mime_type in Config.ALLOWED_VIDEO_TYPES
            else: # unknown, check both
                is_valid_type = (mime_type in Config.ALLOWED_IMAGE_TYPES or
                                 mime_type in Config.ALLOWED_VIDEO_TYPES)
            
            if not is_valid_type:
                allowed_types_str = ", ".join(Config.ALLOWED_IMAGE_TYPES + Config.ALLOWED_VIDEO_TYPES)
                raise HTTPException(status_code=400, detail=f"Invalid file type: {mime_type}. Allowed types: {allowed_types_str}")
            
            # Check file size
            if file.size > Config.MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File too large. Max size is {Config.MAX_FILE_SIZE_MB}MB.")

            filename = f"{uuid4()}{file_extension}"
            filepath = os.path.join(Config.UPLOADS_DIR, filename)

            # Save the file asynchronously
            async with aiofiles.open(filepath, "wb") as out_file:
                while content := await file.read(1024 * 1024):
                    await out_file.write(content)

            file_size = os.path.getsize(filepath)

            log_info(f"File uploaded: {filename}", {
                "originalName": file.filename,
                "mimeType": mime_type,
                "type": file_type,
                "size": file_size
            })

            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "originalName": file.filename,
                "mimeType": mime_type,
                "type": file_type
            }
        except HTTPException:
            raise
        except Exception as e:
            log_error("Error uploading file:", e)
            raise HTTPException(status_code=500, detail=f"Error processing file upload: {e}")

upload_handler = UploadHandler()

# --- Realtime Handler ---

# In-memory storage for user sessions (replace with Redis/DB for production scale)
user_sessions: Dict[str, Dict[str, Any]] = {}

async def cleanup_inactive_sessions():
    now = time.time() * 1000
    inactive_threshold = Config.REALTIME_SESSION_TIMEOUT_MS
    
    sessions_to_delete = []
    for session_id, session_data in user_sessions.items():
        if now - session_data['last_activity'] > inactive_threshold:
            sessions_to_delete.append(session_id)
            log_info(f"Cleaned up inactive session: {session_id}")
            
    for session_id in sessions_to_delete:
        del user_sessions[session_id]

def create_session(user_id: Optional[str] = None) -> Dict[str, Any]:
    session_id = user_id if user_id else str(uuid4())
    session_data = {
        "user_id": user_id,
        "static_buffer": [],
        "dynamic_buffer": [],
        "last_letter": None,
        "last_word": None,
        "current_word": "",
        "full_text": "",
        "stable_frames": 0,
        "is_in_motion": False,
        "last_activity": time.time() * 1000,
        "websocket": None
    }
    user_sessions[session_id] = session_data
    log_info(f"New user session created: {session_id}")
    return {"success": True, "sessionId": session_id}

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    session = user_sessions.get(session_id)
    if session:
        session['last_activity'] = time.time() * 1000
    return session

def end_session(session_id: str) -> Dict[str, Any]:
    session = user_sessions.pop(session_id, None)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    if session.get("current_word"):
        complete_word_in_session(session)

    log_info(f"Session ended: {session_id}")
    return {"success": True, "fullText": session.get("full_text", ""), "sessionId": session_id}

def get_session_status(session_id: str) -> Dict[str, Any]:
    session = user_sessions.get(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    return {
        "success": True,
        "sessionId": session_id,
        "currentWord": session.get("current_word", ""),
        "fullText": session.get("full_text", ""),
        "lastLetter": session.get("last_letter", None),
        "lastWord": session.get("last_word", None)
    }

async def publish_update(session_id: str, update: Dict[str, Any]):
    session = user_sessions.get(session_id)
    if session and session.get("websocket"):
        try:
            await session["websocket"].send_json({
                "timestamp": int(time.time() * 1000),
                **update
            })
        except Exception as e:
            log_error(f"Failed to send WebSocket update for session {session_id}:", e)
            session["websocket"] = None

def is_stable_pose(session: Dict[str, Any], landmarks: List[float]) -> bool:
    if not session["static_buffer"]:
        session["static_buffer"].append(landmarks)
        return False
    
    last_landmarks = session["static_buffer"][-1]
    diff_sum = sum(abs(landmarks[i] - last_landmarks[i]) for i in range(min(len(landmarks), len(last_landmarks))))
    avg_diff = diff_sum / len(landmarks)
    
    session["static_buffer"].append(landmarks)
    if len(session["static_buffer"]) > 10:
        session["static_buffer"].pop(0)

    is_stable = avg_diff < (Config.REALTIME_MOVEMENT_THRESHOLD or 0.015)
    
    if is_stable:
        session["stable_frames"] += 1
    else:
        session["stable_frames"] = 0
    
    return session["stable_frames"] >= (Config.REALTIME_STABLE_FRAME_THRESHOLD or 5)

def detect_dynamic_sign(session: Dict[str, Any], landmarks: List[float]) -> Dict[str, bool]:
    if not session["dynamic_buffer"]:
        session["dynamic_buffer"].append(landmarks)
        return {"isStarting": False, "isEnding": False}
    
    last_landmarks = session["dynamic_buffer"][-1]
    diff_sum = sum(abs(landmarks[i] - last_landmarks[i]) for i in range(min(len(landmarks), len(last_landmarks))))
    avg_diff = diff_sum / len(landmarks)
    
    session["dynamic_buffer"].append(landmarks)
    if len(session["dynamic_buffer"]) > 30:
        session["dynamic_buffer"].pop(0)

    movement_threshold = Config.REALTIME_MOVEMENT_THRESHOLD or 0.03
    is_moving = avg_diff > movement_threshold
    
    was_in_motion = session["is_in_motion"]
    session["is_in_motion"] = is_moving
    
    return {
        "isStarting": is_moving and not was_in_motion,
        "isEnding": not is_moving and was_in_motion and len(session["dynamic_buffer"]) > (Config.REALTIME_MIN_SEQUENCE_FRAMES or 15)
    }

async def add_letter_to_word(session: Dict[str, Any], letter: str):
    if letter == session["last_letter"]:
        return
    
    session["last_letter"] = letter
    session["current_word"] += letter
    
    await publish_update(session["user_id"], {
        "type": "letter",
        "letter": letter,
        "currentWord": session["current_word"],
        "fullText": session["full_text"]
    })

async def complete_word_in_session(session: Dict[str, Any], word: Optional[str] = None):
    word_to_add = word if word else session["current_word"]
    
    if not word_to_add:
        return
    
    if session["full_text"]:
        session["full_text"] += " "
    
    session["full_text"] += word_to_add
    session["last_word"] = word_to_add
    session["current_word"] = ""
    
    await publish_update(session["user_id"], {
        "type": "word",
        "word": word_to_add,
        "fullText": session["full_text"]
    })

async def process_realtime_landmarks(session_id: str, landmarks: List[float]):
    session = get_session(session_id)
    if not session:
        log_error(f"Session {session_id} not found for landmark processing.")
        return {"success": False, "error": "Session not found"}

    try:
        static_result = await model_handler.predict_static_sign(landmarks)
        
        if static_result and static_result["confidence"] > (Config.REALTIME_CONFIDENCE_THRESHOLD or 0.7):
            await add_letter_to_word(session, static_result["class"])
        session["stable_frames"] = 0 
        
        dynamic_status = detect_dynamic_sign(session, landmarks)
        
        if dynamic_status["isEnding"]:
            dynamic_result = await model_handler.predict_dynamic_sign(session["dynamic_buffer"])
            
            if dynamic_result and dynamic_result["confidence"] > (Config.REALTIME_CONFIDENCE_THRESHOLD or 0.7):
                await complete_word_in_session(session, dynamic_result["class"])
            session["dynamic_buffer"] = []

        return {"success": True, "sessionId": session_id}
    except Exception as e:
        log_error(f"Error processing realtime landmarks for session {session_id}:", e)
        return {"success": False, "error": str(e)}

async def process_realtime_landmark_sequence(session_id: str, landmark_sequence: List[List[float]], model_choice: str = 'lstm'):
    session = get_session(session_id)
    if not session:
        log_error(f"Session {session_id} not found for landmark sequence processing.")
        return {"success": False, "error": "Session not found"}
    
    try:
        # Panggil predict_dynamic_sign tanpa parameter model_choice karena sudah di hardcode ke lstm
        dynamic_result = await model_handler.predict_dynamic_sign(landmark_sequence)
        if dynamic_result and dynamic_result["confidence"] > (Config.REALTIME_CONFIDENCE_THRESHOLD or 0.7):
            await complete_word_in_session(session, dynamic_result["class"])
        
        return {"success": True, "sessionId": session_id, "result": dynamic_result}
    except Exception as e:
        log_error(f"Error processing realtime landmark sequence for session {session_id}:", e)
        return {"success": False, "error": str(e)}

def correct_prediction(session_id: str, correction_type: str, correction_value: str) -> Dict[str, Any]:
    session = user_sessions.get(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    if correction_type == 'letter':
        if session["current_word"]:
            session["current_word"] = session["current_word"][:-1] + correction_value
    elif correction_type == 'word':
        words = session["full_text"].split(' ')
        if words:
            words[-1] = correction_value
            session["full_text"] = ' '.join(words)
        else:
            session["full_text"] = correction_value
    elif correction_type == 'clearWord':
        session["current_word"] = ''
    elif correction_type == 'clearText':
        session["full_text"] = ''
        session["current_word"] = ''
    else:
        return {"success": False, "error": "Invalid correction type"}
    
    asyncio.create_task(publish_update(session_id, {
        "type": "correction",
        "currentWord": session["current_word"],
        "fullText": session["full_text"]
    }))
    
    return {
        "success": True,
        "currentWord": session["current_word"],
        "fullText": session["full_text"]
    }

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Artisign BISINDO Translator API",
    description="Backend API for BISINDO (Indonesian Sign Language) Translator",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    log_info("Starting up FastAPI application...")
    Config.create_dirs()
    app.state.start_time = time.time()
    try:
        # model_handler sudah didefinisikan sekarang
        await model_handler.load_models()
        log_info("All models and mappings loaded successfully.")
    except Exception as e:
        log_error("Failed to load models during startup. Exiting...", e)
        raise RuntimeError("Failed to load AI models, cannot start application.")

    app.state.cleanup_task = asyncio.create_task(background_cleanup_task())

@app.on_event("shutdown")
async def shutdown_event():
    log_info("Shutting down FastAPI application...")
    if hasattr(app.state, 'cleanup_task'):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            log_info("Background cleanup task cancelled.")

async def background_cleanup_task():
    while True:
        await asyncio.sleep(30 * 60)
        await cleanup_inactive_sessions()

# --- Pydantic Models for Request/Response Bodies ---
class LandmarkPayload(BaseModel):
    landmarks: List[float] = Field(..., description=f"Flattened array of {Config.NUM_LANDMARK_FEATURES} hand landmarks (x, y, z for each of 21 points).")

class DynamicSignPayload(BaseModel):
    landmarkSequence: List[List[float]] = Field(..., description=f"Array of landmark arrays, each representing a frame. Expected inner array length: {Config.NUM_LANDMARK_FEATURES}.")
    # modelChoice dihapus dari sini

class TextToSignPayload(BaseModel):
    text: str = Field(..., min_length=1, description="Text to convert to sign language representation.")

class CreateSessionPayload(BaseModel):
    userId: Optional[str] = Field(None, description="Optional user ID to associate with the session.")

class EndSessionPayload(BaseModel):
    sessionId: str = Field(..., description="The ID of the session to end.")

class RealtimeLandmarksPayload(BaseModel):
    sessionId: str = Field(..., description="The ID of the real-time session.")
    landmarks: List[float] = Field(..., description=f"Current hand landmarks. Expected length: {Config.NUM_LANDMARK_FEATURES}.")

class RealtimeLandmarkSequencePayload(BaseModel):
    sessionId: str = Field(..., description="The ID of the real-time session.")
    landmarkSequence: List[List[float]] = Field(..., description="Sequence of hand landmarks for dynamic sign prediction.")
    # modelChoice dihapus dari sini

class CorrectionPayload(BaseModel):
    sessionId: str = Field(..., description="The ID of the real-time session.")
    correctionType: str = Field(..., description="Type of correction: 'letter', 'word', 'clearWord', 'clearText'.")
    correction: str = Field(..., description="The correction value (e.g., corrected letter/word) or empty string for clear operations.")

# --- API Endpoints (Routes) ---

@app.get("/")
async def root():
    return {"message": "Artisign BISINDO Translator API"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "models": model_handler.get_model_status(),
        "uptime": time.time() - app.state.start_time
    }

@app.post("/api/predict-static-sign")
async def predict_static_sign_route(payload: LandmarkPayload):
    try:
        result = await model_handler.predict_static_sign(payload.landmarks)
        return {"success": True, "result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in static sign prediction route:", e)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "result": {"class": "A", "confidence": 0.9, "index": 0}})

@app.post("/api/predict-static-sign-form")
async def predict_static_sign_form_route(landmarks: str = Form(...)):
    try:
        landmarks_list = json.loads(landmarks)
        if not isinstance(landmarks_list, list):
            raise HTTPException(status_code=400, detail="Invalid landmarks data format: must be a JSON array.")
        result = await model_handler.predict_static_sign(landmarks_list)
        return {"success": True, "result": result}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for landmarks: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in static sign prediction route:", e)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "result": {"class": "A", "confidence": 0.9, "index": 0}})

@app.get("/api/available-letters")
async def get_available_letters_route():
    try:
        letters = model_handler.get_available_letters()
        return {"success": True, "count": len(letters), "letters": letters}
    except Exception as e:
        log_error("Error fetching available letters:", e)
        raise HTTPException(status_code=500, detail="Error processing the request")

@app.post("/api/predict-dynamic-sign")
async def predict_dynamic_sign_route(payload: DynamicSignPayload):
    try:
        result = await model_handler.predict_dynamic_sign(payload.landmarkSequence)
        return {"success": True, "result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in dynamic sign prediction route:", e)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "result": {"class": "Halo", "confidence": 0.9, "index": 11, "modelUsed": "lstm"}})

@app.post("/api/predict-dynamic-sign-form")
async def predict_dynamic_sign_form_route(
    landmarkSequence: str = Form(...),
):
    try:
        sequence_list = json.loads(landmarkSequence)
        if not isinstance(sequence_list, list):
            raise HTTPException(status_code=400, detail="Invalid landmark sequence data format: must be a JSON array.")
        result = await model_handler.predict_dynamic_sign(sequence_list)
        return {"success": True, "result": result}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for landmark sequence: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in dynamic sign form prediction route:", e)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "result": {"class": "Halo", "confidence": 0.9, "index": 11, "modelUsed": "lstm"}})

@app.get("/api/available-words")
async def get_available_words_route():
    try:
        words = model_handler.get_available_words()
        return {"success": True, "count": len(words), "words": words}
    except Exception as e:
        log_error("Error fetching available words:", e)
        raise HTTPException(status_code=500, detail="Error processing the request")

@app.post("/api/text-to-sign")
async def text_to_sign_route(payload: TextToSignPayload):
    try:
        result = model_handler.text_to_sign(payload.text)
        log_info("Text to sign conversion", {
            "text": payload.text,
            "wordCount": len(result["signs"])
        })
        return {"success": True, **result}
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in text to sign conversion:", e)
        raise HTTPException(status_code=500, detail="Error processing the request")

@app.post("/api/upload")
async def upload_file_route(file: UploadFile = File(...), fileType: Optional[str] = Form('unknown')):
    if file.size > Config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File too large. Max size is {Config.MAX_FILE_SIZE_MB}MB.")
    
    try:
        result = await upload_handler.handle_file_upload(file, fileType)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error in upload route:", e)
        raise HTTPException(status_code=500, detail=f"Error processing file upload: {e}")

@app.post("/api/test-predict")
async def test_predict_route():
    return {
        "success": True,
        "result": {
            "class": "A",
            "confidence": 0.95,
            "index": 0
        }
    }

@app.post("/api/test-dynamic")
async def test_dynamic_route():
    return {
        "success": True,
        "result": {
            "class": "Halo",
            "confidence": 0.92,
            "index": 11,
            "modelUsed": "lstm"
        }
    }

# --- Realtime Routes ---
@app.post("/api/realtime/session/create")
async def create_realtime_session_route(payload: CreateSessionPayload):
    try:
        return create_session(payload.userId)
    except Exception as e:
        log_error("Error creating realtime session:", e)
        raise HTTPException(status_code=500, detail=f"Error creating session: {e}")

@app.post("/api/realtime/session/end")
async def end_realtime_session_route(payload: EndSessionPayload):
    try:
        return end_session(payload.sessionId)
    except Exception as e:
        log_error("Error ending realtime session:", e)
        raise HTTPException(status_code=500, detail=f"Error ending session: {e}")

@app.get("/api/realtime/session/{session_id}/status")
async def get_realtime_session_status_route(session_id: str):
    try:
        return get_session_status(session_id)
    except Exception as e:
        log_error("Error getting session status:", e)
        raise HTTPException(status_code=500, detail=f"Error getting session status: {e}")

@app.post("/api/realtime/landmarks")
async def realtime_landmarks_route(payload: RealtimeLandmarksPayload):
    try:
        result = await process_realtime_landmarks(payload.sessionId, payload.landmarks)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error processing realtime landmarks:", e)
        raise HTTPException(status_code=500, detail=f"Error processing landmarks: {e}")

@app.post("/api/realtime/landmark-sequence")
async def realtime_landmark_sequence_route(payload: RealtimeLandmarkSequencePayload):
    try:
        result = await process_realtime_landmark_sequence(payload.sessionId, payload.landmarkSequence)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error("Error processing realtime landmark sequence:", e)
        raise HTTPException(status_code=500, detail=f"Error processing landmark sequence: {e}")

@app.post("/api/realtime/correction")
async def realtime_correction_route(payload: CorrectionPayload):
    try:
        result = correct_prediction(payload.sessionId, payload.correctionType, payload.correction)
        return result
    except Exception as e:
        log_error("Error processing correction:", e)
        raise HTTPException(status_code=500, detail=f"Error processing correction: {e}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/realtime/sign/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = get_session(session_id)
    if not session:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid session ID")
        return
    
    # Store the websocket connection in the session
    session["websocket"] = websocket
    log_info(f"WebSocket connected for session: {session_id}")
    
    try:
        await websocket.accept()
        while True:
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        log_info(f"WebSocket disconnected for session: {session_id}")
        session["websocket"] = None
    except Exception as e:
        log_error(f"WebSocket error for session {session_id}:", e)
        session["websocket"] = None
    finally:
        log_info(f"WebSocket connection closed for session: {session_id}")
        session["websocket"] = None

app.start_time = time.time()