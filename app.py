import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, LayerNormalization, Conv1D,
                                     MultiHeadAttention, GlobalAveragePooling1D,
                                     Concatenate)
from tensorflow.keras.regularizers import l2
from scipy.interpolate import interp1d

# --- 1. CONFIGURATION ---
WEIGHTS_PATH = 'action_universal_final.h5'
actions = np.array([
    "valobasha", "valo", "kharap", "olosh", 
    "boka", "chad", "akash", "dhonnobad", "shikkhok", 
    "football", "stree", "durbol"
])

# --- 2. THE PHYSICS ENGINE (Universal Extractor) ---
class UniversalMotionExtractor:
    def __init__(self):
        self.prev_landmarks = None
        self.prev_prev_landmarks = None
        
    def extract_universal_features(self, results):
        # 1. Raw Landmarks
        current_landmarks = self._extract_raw_landmarks(results)
        # 2. Body Relative Normalization
        normalized = self._normalize_to_body(current_landmarks, results)
        # 3. Motion Derivatives (Velocity/Accel)
        motion_features = self._compute_motion_derivatives(normalized)
        # 4. Spatial Relations
        spatial_features = self._compute_spatial_relationships(normalized)
        # 5. Hand Config
        config_features = self._compute_hand_configurations(normalized)
        # 6. Directional Patterns
        direction_features = self._compute_directional_patterns(motion_features)
        # 7. Interactions
        interaction_features = self._compute_hand_interactions(normalized, motion_features)
        
        # Update History
        self.prev_prev_landmarks = self.prev_landmarks
        self.prev_landmarks = normalized
        
        # Flatten and Concatenate (Total: 607 features)
        return np.concatenate([
            normalized['pose'].flatten(),           # 99
            normalized['left_hand'].flatten(),      # 63
            normalized['right_hand'].flatten(),     # 63
            motion_features['velocity'].flatten(),  # 126
            motion_features['acceleration'].flatten(), # 126
            spatial_features,                       # 30
            config_features,                        # 40
            direction_features,                     # 25
            interaction_features                    # 35
        ])
    
    def _extract_raw_landmarks(self, results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        return {'pose': pose, 'left_hand': lh, 'right_hand': rh, 
                'has_left': results.left_hand_landmarks is not None, 'has_right': results.right_hand_landmarks is not None}

    def _normalize_to_body(self, landmarks, results):
        pose = landmarks['pose']
        if results.pose_landmarks:
            left_shoulder, right_shoulder = pose[11], pose[12]
            origin = (left_shoulder + right_shoulder) / 2
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale == 0: scale = 1.0
        else:
            origin = np.zeros(3); scale = 1.0
            
        pose_norm = (pose - origin) / scale
        lh_norm = (landmarks['left_hand'] - origin) / scale if landmarks['has_left'] else np.zeros((21, 3))
        rh_norm = (landmarks['right_hand'] - origin) / scale if landmarks['has_right'] else np.zeros((21, 3))
        
        return {'pose': pose_norm, 'left_hand': lh_norm, 'right_hand': rh_norm, 
                'has_left': landmarks['has_left'], 'has_right': landmarks['has_right']}

    def _compute_motion_derivatives(self, current):
        velocity = np.zeros((42, 3))
        acceleration = np.zeros((42, 3))
        
        if self.prev_landmarks:
            lh_vel = current['left_hand'] - self.prev_landmarks['left_hand']
            rh_vel = current['right_hand'] - self.prev_landmarks['right_hand']
            velocity = np.vstack([lh_vel, rh_vel])
            
            if self.prev_prev_landmarks:
                prev_lh_vel = self.prev_landmarks['left_hand'] - self.prev_prev_landmarks['left_hand']
                prev_rh_vel = self.prev_landmarks['right_hand'] - self.prev_prev_landmarks['right_hand']
                lh_acc = lh_vel - prev_lh_vel
                rh_acc = rh_vel - prev_rh_vel
                acceleration = np.vstack([lh_acc, rh_acc])
        
        return {'velocity': velocity, 'acceleration': acceleration}

    def _compute_spatial_relationships(self, landmarks):
        feats = []
        lh, rh, pose = landmarks['left_hand'], landmarks['right_hand'], landmarks['pose']
        has_l, has_r = landmarks['has_left'], landmarks['has_right']
        
        if has_l and has_r:
            for idx in [0, 8, 4, 12, 20]: 
                feats.append(np.linalg.norm(lh[idx] - rh[idx]))
        else: feats.extend([0]*5)
        
        for body_idx in [0, 11, 12, 13, 14]:
            feats.append(np.linalg.norm(lh[0] - pose[body_idx]) if has_l else 0)
            feats.append(np.linalg.norm(rh[0] - pose[body_idx]) if has_r else 0)
            
        shoulder_y = (pose[11][1] + pose[12][1]) / 2
        feats.append(lh[0][1] - shoulder_y if has_l else 0)
        feats.append(rh[0][1] - shoulder_y if has_r else 0)
        
        curr = len(feats)
        if curr < 30: feats.extend([0] * (30 - curr))
        return np.array(feats[:30])

    def _compute_hand_configurations(self, landmarks):
        feats = []
        for hand, present in [(landmarks['left_hand'], landmarks['has_left']), (landmarks['right_hand'], landmarks['has_right'])]:
            if present:
                wrist = hand[0]
                for tip in [4, 8, 12, 16, 20]: feats.append(np.linalg.norm(hand[tip] - wrist))
                for i in range(4):
                    t1, t2 = hand[[4,8,12,16][i]], hand[[8,12,16,20][i]]
                    feats.append(np.linalg.norm(t1 - t2))
                feats.extend(hand[9] - hand[0])
                for b, t in [(1,4), (5,8), (9,12), (13,16), (17,20)]:
                    feats.append(np.linalg.norm(hand[t] - hand[b]))
            else: feats.extend([0]*20)
        
        curr = len(feats)
        if curr < 40: feats.extend([0]*(40-curr))
        return np.array(feats[:40])

    def _compute_directional_patterns(self, motion):
        vel = motion['velocity']
        feats = []
        lh_v, rh_v = vel[:21], vel[21:]
        
        for v in [lh_v, rh_v]:
            feats.extend(v.mean(axis=0))
            feats.extend(v.std(axis=0)) 
            feats.append(np.abs(v).mean())
            
        comb = vel.mean(axis=0)
        feats.extend(comb)
        feats.append(np.linalg.norm(comb))
        
        curr = len(feats)
        if curr < 25: feats.extend([0]*(25-curr))
        return np.array(feats[:25])

    def _compute_hand_interactions(self, landmarks, motion):
        feats = []
        if landmarks['has_left'] and landmarks['has_right']:
            lh_v, rh_v = motion['velocity'][0], motion['velocity'][21]
            lh_p, rh_p = landmarks['left_hand'][0], landmarks['right_hand'][0]
            feats.append(np.linalg.norm(lh_v - rh_v))
            feats.append(np.dot(lh_v, rh_v)) 
            feats.append(np.linalg.norm(lh_p - rh_p))
            feats.extend(np.abs(lh_p - rh_p))
        
        curr = len(feats)
        if curr < 35: feats.extend([0]*(35-curr))
        return np.array(feats[:35])

# --- 3. REBUILD THE BRAIN (Architecture) ---
def create_universal_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, 5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(96, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    attn = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization()(attn + x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

print("🦇 Initializing WayneTech Systems...")
model = create_universal_model((30, 607), actions.shape[0])
model.load_weights(WEIGHTS_PATH)
print("✅ SYSTEM ONLINE: Universal Motion Engine Active.")

# --- 4. EXECUTION ---
def smart_resample(frames, target_len=30):
    if len(frames) == 0: return np.zeros((target_len, 607))
    frames_arr = np.array(frames)
    if len(frames) == target_len: return frames_arr
    if len(frames) < target_len:
        return np.vstack([frames_arr, np.repeat(frames_arr[-1:], target_len - len(frames), axis=0)])
    old = np.linspace(0, len(frames)-1, len(frames))
    new = np.linspace(0, len(frames)-1, target_len)
    res = []
    for i in range(frames_arr.shape[1]):
        res.append(interp1d(old, frames_arr[:, i], kind='cubic', fill_value='extrapolate')(new))
    return np.array(res).T

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sequence = []
threshold = 0.85 

cap = cv2.VideoCapture(0)
extractor = UniversalMotionExtractor()

window_name = 'WayneTech Universal Interface'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if results.left_hand_landmarks or results.right_hand_landmarks:
            # 🦇 UNIVERSAL EXTRACTION 🦇
            features = extractor.extract_universal_features(results)
            sequence.append(features)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                input_data = smart_resample(sequence, 30)
                res = model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
                best_class = np.argmax(res)
                confidence = res[best_class]
                current_word = actions[best_class]
                
                # Dynamic Logic: Check if there is enough motion energy
                # Features 225 to 477 are velocity/acceleration
                motion_energy = np.mean(np.abs(features[225:477]))
                
                if confidence > threshold:
                    # Optional: Add a motion threshold here if 'static bias' persists
                    # if motion_energy > 0.02: 
                    h, w, _ = image.shape
                    cv2.putText(image, f"DETECTED: {current_word.upper()}", (w - 400, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"CONF: {confidence*100:.1f}% | MOTION: {motion_energy:.3f}", (w - 400, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # Reset history if hands are lost
            extractor.prev_landmarks = None
            extractor.prev_prev_landmarks = None

        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()