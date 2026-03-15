import cv2
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
import time
from ultralytics import YOLO
import pygame

# ============== CONFIGURATION ==============
SAVE_DIR = "wrong_direction_violators"
MODEL_PATH = "yolov8n_ncnn_model"

# VIDEO FILE PATH - CHANGE THIS TO YOUR VIDEO
VIDEO_PATH = "/home/metro/Downloads/test_video.mp4"  # ← PUT YOUR VIDEO PATH HERE

# Processing resolution (downscaled for speed)
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

# Detection zones (4 zones for directional tracking)
ZONE_1 = 0.20
ZONE_2 = 0.40
ZONE_3 = 0.60
ZONE_4 = 0.80

# Detection settings
CONFIDENCE_THRESHOLD = 0.45
TRACK_TIMEOUT = 25

# Performance settings
PROCESS_EVERY_N_FRAMES = 1
DISPLAY_SCALE = 0.8

# ============== INITIALIZATION ==============
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize pygame for LOUD beep
print("Initializing audio...")
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

sample_rate = 22050
duration = 0.3
frequency = 1200
beep_samples = int(sample_rate * duration)
beep_wave_mono = np.array([int(32767 * 0.9 * np.sin(2 * np.pi * frequency * i / sample_rate)) 
                           for i in range(beep_samples)], dtype=np.int16)
beep_wave_stereo = np.column_stack((beep_wave_mono, beep_wave_mono))
beep_sound = pygame.sndarray.make_sound(beep_wave_stereo)
pygame.mixer.music.set_volume(1.0)

print("✓ Audio ready")

# Load YOLO
print("Loading YOLO...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ {MODEL_PATH}/ not found!")
    exit(1)

model = YOLO(MODEL_PATH)
print("✓ Model loaded")

# Initialize video file
print(f"Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Cannot open video file: {VIDEO_PATH}")
    print("Check if:")
    print("  1. File path is correct")
    print("  2. File exists")
    print("  3. Video format is supported (mp4, avi, mov)")
    exit(1)

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_duration = total_frames / video_fps if video_fps > 0 else 0

print(f"✓ Video loaded successfully")
print(f"  Resolution: {actual_width}x{actual_height}")
print(f"  FPS: {video_fps}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {video_duration:.1f} seconds")

# Calculate zone boundaries (based on video resolution)
zone1_x = int(actual_width * ZONE_1)
zone2_x = int(actual_width * ZONE_2)
zone3_x = int(actual_width * ZONE_3)
zone4_x = int(actual_width * ZONE_4)

# Calculate scale factors for box coordinate conversion
scale_x = actual_width / PROCESS_WIDTH
scale_y = actual_height / PROCESS_HEIGHT

print(f"✓ Processing: {PROCESS_WIDTH}x{PROCESS_HEIGHT} (LOW-RES for speed)")
print(f"✓ Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
print(f"✓ Zones: {zone1_x}, {zone2_x}, {zone3_x}, {zone4_x}")

# ============== TRACKING DATA STRUCTURES ==============
track_last_zone = {}
track_zone_sequence = defaultdict(list)
track_last_seen = {}
captured_violations = set()

fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0
frame_count = 0

# Static window name
WINDOW_NAME = "Metro Direction Monitor - VIDEO PLAYBACK"

# ============== HELPER FUNCTIONS ==============
def get_zone(x_position):
    """Return which zone (1-4) the x position is in"""
    if x_position < zone1_x:
        return 0
    elif x_position < zone2_x:
        return 1
    elif x_position < zone3_x:
        return 2
    elif x_position < zone4_x:
        return 3
    else:
        return 4

def check_violation(zone_sequence):
    """Check if zone sequence indicates wrong direction"""
    if len(zone_sequence) < 3:
        return False
    
    recent = zone_sequence[-3:]
    
    if (recent[0] < recent[1] < recent[2] and 
        recent[2] - recent[0] >= 2):
        return True
    
    return False

def play_loud_beep():
    try:
        beep_sound.play()
        beep_sound.set_volume(1.0)
    except:
        pass

def capture_violation(frame, box, track_id):
    """Capture HIGH-RES violation image"""
    x1, y1, x2, y2 = box
    clean_frame = frame.copy()
    
    # Draw ONLY red bounding box and center dot
    cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.circle(clean_frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)
    
    # Save HIGH-RES image
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{SAVE_DIR}/violation_ID{track_id}_{timestamp_file}.jpg"
    cv2.imwrite(filename, clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    play_loud_beep()
    
    # Info in console
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sequence_str = " -> ".join(map(str, track_zone_sequence[track_id][-5:]))
    print(f"\n🚨 VIOLATION - ID:{track_id} 🔊")
    print(f"   Frame: {frame_count}/{total_frames}")
    print(f"   Time: {timestamp}")
    print(f"   Direction: LEFT to RIGHT")
    print(f"   Zone path: {sequence_str}")
    print(f"   {filename}")
    
    return filename

# ============== MAIN LOOP ==============
print("\n" + "="*70)
print("DIRECTIONAL ZONE TRACKING - Metro Monitor")
print("="*70)
print("MODE: VIDEO FILE PLAYBACK")
print("4 Zones: LEFT [1][2][3][4] RIGHT")
print("VIOLATION: 1→2→3 (Left to Right) 🔊")
print("Press 'q' to quit, SPACE to pause/resume, 't' to test beep, 'r' to reset")
print("="*70 + "\n")

# Create window once
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

paused = False

try:
    while True:
        if not paused:
            ret, frame_highres = cap.read()
            if not ret:
                print("\n✓ Video finished!")
                break
            
            frame_count += 1
            
            # Downscale for YOLO processing
            frame_lowres = cv2.resize(frame_highres, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Keep HIGH-RES copy for violation photos
            clean_frame_highres = frame_highres.copy()
            
            # Run YOLO tracking on LOW-RES frame
            results = model.track(
                frame_lowres,
                persist=True,
                classes=[0],
                conf=CONFIDENCE_THRESHOLD,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            # Clean up old tracks
            inactive = [tid for tid, last in track_last_seen.items() 
                       if frame_count - last > TRACK_TIMEOUT]
            for tid in inactive:
                track_last_zone.pop(tid, None)
                track_zone_sequence.pop(tid, None)
                track_last_seen.pop(tid, None)
            
            active_ids = set()
            
            # Process detections
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    # Scale bounding box from LOW-RES to HIGH-RES
                    x1_low, y1_low, x2_low, y2_low = box
                    x1_high = int(x1_low * scale_x)
                    y1_high = int(y1_low * scale_y)
                    x2_high = int(x2_low * scale_x)
                    y2_high = int(y2_low * scale_y)
                    
                    center_x_high = (x1_high + x2_high) // 2
                    center_y_high = (y1_high + y2_high) // 2
                    
                    active_ids.add(track_id)
                    track_last_seen[track_id] = frame_count
                    
                    # Determine current zone
                    current_zone = get_zone(center_x_high)
                    
                    # If zone changed, add to sequence
                    if track_id not in track_last_zone or track_last_zone[track_id] != current_zone:
                        track_last_zone[track_id] = current_zone
                        track_zone_sequence[track_id].append(current_zone)
                        
                        if len(track_zone_sequence[track_id]) > 10:
                            track_zone_sequence[track_id] = track_zone_sequence[track_id][-10:]
                    
                    # Check for violation
                    if (check_violation(track_zone_sequence[track_id]) and 
                        track_id not in captured_violations):
                        
                        # Capture HIGH-RES violation photo
                        capture_violation(clean_frame_highres, 
                                        (x1_high, y1_high, x2_high, y2_high), 
                                        track_id)
                        captured_violations.add(track_id)
                        color = (0, 0, 255)
                        label = "VIOLATOR!"
                    
                    elif track_id in captured_violations:
                        color = (0, 0, 255)
                        label = "VIOLATOR"
                    else:
                        if len(track_zone_sequence[track_id]) >= 2:
                            if track_zone_sequence[track_id][-1] > track_zone_sequence[track_id][-2]:
                                color = (255, 165, 0)
                                label = "→"
                            else:
                                color = (0, 255, 0)
                                label = "←"
                        else:
                            color = (128, 128, 128)
                            label = "NEW"
                    
                    # Draw on LOW-RES frame for display
                    cv2.rectangle(frame_lowres, (x1_low, y1_low), (x2_low, y2_low), color, 2)
                    cv2.putText(frame_lowres, f"{track_id}:{label}", (x1_low, y1_low-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame_lowres, ((x1_low+x2_low)//2, (y1_low+y2_low)//2), 3, color, -1)
            
            # Draw zone lines on LOW-RES display frame
            for zx_high in [zone1_x, zone2_x, zone3_x, zone4_x]:
                zx_low = int(zx_high / scale_x)
                cv2.line(frame_lowres, (zx_low, 0), (zx_low, PROCESS_HEIGHT), (150, 150, 150), 1)
            
            # Calculate processing FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps_end_time = time.time()
                fps_display = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # Calculate progress
            progress_pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            
            # Update window title with stats
            window_title = f"{WINDOW_NAME} | FPS:{fps_display:.1f} Frame:{frame_count}/{total_frames} ({progress_pct:.1f}%) Violations:{len(captured_violations)}"
            cv2.setWindowTitle(WINDOW_NAME, window_title)
            
            # Display LOW-RES frame
            display_frame = cv2.resize(frame_lowres, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        else:
            # Paused - just show last frame
            window_title = f"{WINDOW_NAME} | PAUSED | Frame:{frame_count}/{total_frames}"
            cv2.setWindowTitle(WINDOW_NAME, window_title)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹ Stopped by user")
            break
        elif key == ord(' '):  # SPACE bar
            paused = not paused
            print(f"{'⏸ PAUSED' if paused else '▶ RESUMED'}")
        elif key == ord('t'):
            print("🔊 Test beep...")
            play_loud_beep()
        elif key == ord('r'):
            captured_violations.clear()
            track_last_zone.clear()
            track_zone_sequence.clear()
            print("♻ RESET violations")

except KeyboardInterrupt:
    print("\n⏹ Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    
    print("\n" + "="*70)
    print("VIDEO PROCESSING COMPLETE")
    print("="*70)
    print(f"Total frames processed: {frame_count}/{total_frames}")
    print(f"Violations detected: {len(captured_violations)}")
    if captured_violations:
        print(f"Violation IDs: {sorted(captured_violations)}")
    print(f"Photos saved in: {SAVE_DIR}/")
    print(f"Photo resolution: {actual_width}x{actual_height}")
    print("="*70)
