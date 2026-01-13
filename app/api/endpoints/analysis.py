from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional, Tuple
import cv2
import numpy as np
import librosa
import tempfile
import os
import shutil
from PIL import Image
from PIL.ExifTags import TAGS
import base64
import io
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.analysis_log import AnalysisLog
from fastapi import Depends

router = APIRouter()

def save_analysis_log(db: Session, type: str, risk: str, score: float, summary: str, details: List[str], ela_image: str = None):
    try:
        log = AnalysisLog(
            content_type=type,
            risk_level=risk,
            confidence_score=score,
            summary=summary,
            details="|".join(details),
            ela_image_b64=ela_image,
            block_hash=f"0x{os.urandom(16).hex()}" # Mock blockchain hash for now
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log
    except Exception as e:
        print(f"Failed to save log: {e}")
        return None


def get_metadata_analysis(image_path: str) -> List[str]:
    """
    Analyze image metadata for AI signatures or editing software.
    """
    signals = []
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()
        
        if not exif_data:
            return signals

        for tag_id in exif_data:
            tag = TAGS.get(tag_id, tag_id)
            data = str(exif_data.get(tag_id)).lower()
            
            # Check for AI signatures
            ai_keywords = ['dall-e', 'midjourney', 'stable diffusion', 'adobe firefly', 'generated']
            if any(k in data for k in ai_keywords):
                signals.append(f"AI Signature found in {tag}: {data}")
            
            # Check for editing software
            if tag == 'Software':
                signals.append(f"Editing Software Detected: {data}")
                
    except Exception as e:
        print(f"Metadata error: {e}")
        
    return signals

def error_level_analysis(image_path: str, quality: int = 90) -> Tuple[float, str]:
    """
    Perform Error Level Analysis (ELA) and return score + base64 image.
    """
    original = cv2.imread(image_path)
    if original is None:
        return 0.0, ""
        
    # Save as JPEG with specific quality
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(tmp.name)
        os.unlink(tmp.name)
        
    # Calculate difference
    diff = 15 * cv2.absdiff(original, compressed)
    
    # Calculate score
    score = np.mean(diff)
    
    # Convert diff to heatmap-like image for frontend
    # Enhance the difference for visibility
    max_diff = np.max(diff)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    enhanced_diff = (diff * scale).astype(np.uint8)
    
    # Color map for better visualization (Heatmap)
    heatmap = cv2.applyColorMap(enhanced_diff, cv2.COLORMAP_JET)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', heatmap)
    b64_image = base64.b64encode(buffer).decode('utf-8')
    
    return float(score), f"data:image/jpeg;base64,{b64_image}"

def analyze_audio_features(audio_path: str):
    """
    Analyze audio for deepfake signatures.
    """
    y, sr = librosa.load(audio_path)
    
    # 1. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    avg_flatness = np.mean(flatness)
    
    # 2. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    avg_zcr = np.mean(zcr)
    
    # 3. Silence detection
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    silence_ratio = 1.0 - (np.sum([end - start for start, end in non_silent_intervals]) / len(y))
    
    return {
        "spectral_flatness": float(avg_flatness),
        "zero_crossing_rate": float(avg_zcr),
        "silence_ratio": float(silence_ratio)
    }

@router.post("/image")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        ela_score, ela_image = error_level_analysis(tmp_path)
        metadata_signals = get_metadata_analysis(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        risk_level = "LOW"
        if ela_score > 4.0: 
            risk_level = "MEDIUM"
        if ela_score > 8.0:
            risk_level = "HIGH"
        
        details = [f"ELA Score: {ela_score:.2f}"] + metadata_signals
        if ela_score > 4.0:
            details.append("Compression artifacts suggest manipulation.")
            
        # Save to DB
        log = save_analysis_log(db, "IMAGE", risk_level, ela_score, f"Image Analysis Complete. Risk Level: {risk_level}", details, ela_image)
        
        return {
            "type": "image",
            "ela_score": ela_score,
            "ela_image": ela_image,
            "risk_level": risk_level,
            "summary": f"Image Analysis Complete. Risk Level: {risk_level}",
            "details": details,
            "block_hash": log.block_hash if log else "0x000"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        features = analyze_audio_features(tmp_path)
        
        os.unlink(tmp_path)
        
        risk_level = "LOW"
        if features['silence_ratio'] < 0.05:
            risk_level = "MEDIUM"
        if features['spectral_flatness'] > 0.5: # Artificial noise floor
             risk_level = "HIGH"
             
        details = [
            f"Silence Ratio: {features['silence_ratio']:.2f}",
            f"Spectral Flatness: {features['spectral_flatness']:.4f}"
        ]
        
        # Save to DB
        log = save_analysis_log(db, "AUDIO", risk_level, 85.0, f"Audio Analysis Complete.", details)
            
        return {
            "type": "audio",
            "features": features,
            "risk_level": risk_level,
            "summary": f"Audio Analysis Complete. Silence Ratio: {features['silence_ratio']:.2f}",
            "details": details,
            "block_hash": log.block_hash if log else "0x000"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video")
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Video Analysis Logic
        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Analyze 5 frames
        frames_to_check = np.linspace(0, frame_count-1, 5, dtype=int)
        
        high_risk_frames = 0
        total_ela = 0
        
        for frame_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Save frame temporarily to run ELA
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as frame_tmp:
                cv2.imwrite(frame_tmp.name, frame)
                frame_path = frame_tmp.name
                
            score, _ = error_level_analysis(frame_path)
            total_ela += score
            if score > 6.0:
                high_risk_frames += 1
                
            os.unlink(frame_path)
            
        cap.release()
        os.unlink(tmp_path)
        
        avg_ela = total_ela / len(frames_to_check) if len(frames_to_check) > 0 else 0
        
        risk_level = "LOW"
        if high_risk_frames >= 2:
            risk_level = "MEDIUM"
        if high_risk_frames >= 4:
            risk_level = "HIGH"
            
        details = [
            f"Average Frame ELA: {avg_ela:.2f}",
            f"Suspicious Frames: {high_risk_frames}"
        ]
        
        # Save to DB
        log = save_analysis_log(db, "VIDEO", risk_level, avg_ela, f"Video Analysis: {high_risk_frames}/5 frames showed manipulation signs.", details)

        return {
            "type": "video",
            "risk_level": risk_level,
            "summary": f"Video Analysis: {high_risk_frames}/5 frames showed manipulation signs.",
            "details": details,
            "block_hash": log.block_hash if log else "0x000"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
