"""
rep_counter_interface.py
Production-ready real-time exercise classification + rep counting module.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ================= CONSTANTS =================
NOSE=0
L_SHOULDER,R_SHOULDER=11,12
L_ELBOW,R_ELBOW=13,14
L_WRIST,R_WRIST=15,16
L_HIP,R_HIP=23,24

# ================= MODEL =================
class RepCountLSTM(nn.Module):
    def __init__(self,input_size,hidden_size=128,num_layers=2,num_classes=8,dropout=0.3):
        super().__init__()
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True,
                          dropout=dropout,bidirectional=True)
        d=hidden_size*2
        self.attention=nn.Linear(d,1)
        self.classifier=nn.Sequential(
            nn.Linear(d,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,num_classes)
        )

    def forward(self,x,real_mask):
        lstm_out,_=self.lstm(x)
        attn=self.attention(lstm_out).squeeze(-1)

        empty=~real_mask.any(dim=1,keepdim=True)
        safe_mask=real_mask|empty.expand_as(real_mask)

        attn=attn.masked_fill(~safe_mask,float('-inf'))
        weights=torch.nan_to_num(torch.softmax(attn,dim=1),
                                 nan=1.0/real_mask.shape[1])

        context=(weights.unsqueeze(-1)*lstm_out).sum(dim=1)
        return self.classifier(context)

# ================= HELPERS =================
def _get_lm(kps,idx):
    return kps[idx*4:idx*4+3]

def _angle_between(a,b,c):
    ba=a-b
    bc=c-b
    denom=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-8
    cos=np.clip(np.dot(ba,bc)/denom,-1,1)
    return np.degrees(np.arccos(cos))

def _torso_size(kps):
    mid_sh=(_get_lm(kps,L_SHOULDER)+_get_lm(kps,R_SHOULDER))/2
    mid_hp=(_get_lm(kps,L_HIP)+_get_lm(kps,R_HIP))/2
    return np.linalg.norm(mid_sh-mid_hp)+1e-8

def engineer_features(kps):
    if np.all(kps==0):
        return np.zeros(46,dtype=np.float32)

    ts=_torso_size(kps)
    feats=[]

    angle_triplets=[
        (11,13,15),(12,14,16),(23,11,13),(24,12,14),
        (11,23,25),(12,24,26),(11,12,24),(23,25,27),
        (24,26,28),(25,27,28),(26,28,27),(23,24,26)
    ]

    for a,b,c in angle_triplets:
        feats.append(_angle_between(_get_lm(kps,a),_get_lm(kps,b),_get_lm(kps,c))/180.0)

    dist_pairs=[
        (15,16),(15,11),(16,12),(15,23),(16,24),
        (27,28),(25,26),(25,23),(26,24),(0,23),(13,14),(15,0)
    ]

    for a,b in dist_pairs:
        feats.append(np.linalg.norm(_get_lm(kps,a)-_get_lm(kps,b))/ts)

    mid_hip_y=(_get_lm(kps,23)[1]+_get_lm(kps,24)[1])/2
    key_joints=[0,11,12,13,14,15,16,25,26,27,28]

    for j in key_joints:
        feats.append((_get_lm(kps,j)[1]-mid_hip_y)/ts)

    for j in key_joints:
        feats.append(kps[j*4+3])

    return np.array(feats,dtype=np.float32)

def landmarks_to_keypoints(landmarks):
    kps=np.zeros(33*4,dtype=np.float32)
    for i,lm in enumerate(landmarks):
        kps[i*4]=lm.x
        kps[i*4+1]=lm.y
        kps[i*4+2]=lm.z
        kps[i*4+3]=lm.visibility
    return kps

# ================= MAIN =================
class RepCounterInterface:

    DEFAULT_CLASSES=[
        'front_raise','push_up','pull_up','bench_pressing',
        'bicep_curl','tricep_extension','lateral_raise','shoulder_press'
    ]

    def __init__(self,model_path='Combined_model.pth',inference_every=5,min_confidence=0.6):

        self.device=torch.device('cpu')
        self.inference_every=inference_every
        self.min_confidence=min_confidence

        ckpt=torch.load(model_path,map_location=self.device,weights_only=False)
        cfg=ckpt['config']

        self.classes=ckpt.get('classes',self.DEFAULT_CLASSES)
        self.max_seq_len=ckpt.get('max_seq_len',30)
        self.norm_mean=ckpt.get('norm_mean',None)
        self.norm_std=ckpt.get('norm_std',None)

        self.model=RepCountLSTM(
            cfg['input_size'],
            cfg['hidden_size'],
            cfg['num_layers'],
            cfg['num_classes'],
            cfg.get('dropout',0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.buffer=deque(maxlen=self.max_seq_len)
        self.frame_count=0

        self.reps=0
        self.phase='UNKNOWN'
        self.exercise=None
        self.confidence=0.0

        # 🔥 new state
        self.last_elbow=None
        self.current_rep_max_swing=0
        self.swing_threshold=80
        self.swing_ok=True
        self.min_elbow=999
        self.max_elbow=0

    def update(self,landmarks):
        if landmarks is None:
            return self._state()

        kps=landmarks_to_keypoints(landmarks)
        feat=engineer_features(kps)

        if self.norm_mean is not None and not np.all(feat==0):
            feat=np.clip((feat-self.norm_mean)/self.norm_std,-5,5)

        self.buffer.append(feat)
        self.frame_count+=1

        self._update_phase(kps)

        if self.frame_count % self.inference_every == 0 and len(self.buffer)>0:
            self._run_inference()

        return self._state()

    def _run_inference(self):
        frames=list(self.buffer)
        n=len(frames)

        seq=np.zeros((self.max_seq_len,46),dtype=np.float32)
        seq[:n]=frames

        mask=np.array([True]*n+[False]*(self.max_seq_len-n))

        x=torch.tensor(seq).unsqueeze(0)
        m=torch.tensor(mask).unsqueeze(0)

        with torch.no_grad():
            probs=torch.softmax(self.model(x,m),dim=1)[0].numpy()

        idx=int(np.argmax(probs))
        conf=float(probs[idx])

        if conf>=self.min_confidence:
            self.exercise=self.classes[idx]
            self.confidence=conf

    # ================= FINAL LOGIC =================
    def _update_phase(self,kps):

        if self.exercise is None:
            return

        prev=self.phase

        r_elbow=_angle_between(_get_lm(kps,R_SHOULDER),_get_lm(kps,R_ELBOW),_get_lm(kps,R_WRIST))
        l_elbow=_angle_between(_get_lm(kps,L_SHOULDER),_get_lm(kps,L_ELBOW),_get_lm(kps,L_WRIST))

        elbow_avg=(r_elbow+l_elbow)/2

        r_sh_y=_get_lm(kps,R_SHOULDER)[1]
        l_sh_y=_get_lm(kps,L_SHOULDER)[1]
        r_wr_y=_get_lm(kps,R_WRIST)[1]
        l_wr_y=_get_lm(kps,L_WRIST)[1]

        wrist_y=(r_wr_y+l_wr_y)/2
        shoulder_y=(r_sh_y+l_sh_y)/2
        nose_y=_get_lm(kps,NOSE)[1]

        ex=self.exercise

        # ===== BICEP CURL =====
        if ex=="bicep_curl":

            r_swing=_angle_between(_get_lm(kps,R_ELBOW),_get_lm(kps,R_SHOULDER),_get_lm(kps,R_HIP))
            l_swing=_angle_between(_get_lm(kps,L_ELBOW),_get_lm(kps,L_SHOULDER),_get_lm(kps,L_HIP))

            r_vis=min(kps[R_SHOULDER*4+3],kps[R_ELBOW*4+3],kps[R_WRIST*4+3])
            l_vis=min(kps[L_SHOULDER*4+3],kps[L_ELBOW*4+3],kps[L_WRIST*4+3])

            r_w=max(r_vis,0.1)
            l_w=max(l_vis,0.1)

            elbow=(r_elbow*r_w+l_elbow*l_w)/(r_w+l_w)
            swing=(r_swing*r_w+l_swing*l_w)/(r_w+l_w)

            if self.last_elbow is None:
                self.last_elbow=elbow

            elbow=0.7*self.last_elbow+0.3*elbow
            self.last_elbow=elbow

            if elbow<100:
                self.phase="UP"
            elif elbow>130:
                self.phase="DOWN"

            if self.phase=="DOWN":
                self.current_rep_max_swing=max(self.current_rep_max_swing,swing)
                if swing>self.swing_threshold:
                    self.swing_ok=False

            if prev=="UP" and self.phase=="DOWN":
                self.swing_ok=True
                self.current_rep_max_swing=0

            if prev=="DOWN" and self.phase=="UP" and self.swing_ok:
                self.reps+=1
                self.current_rep_max_swing=0

            return

        # ===== TRICEP EXTENSION =====
        elif ex=="tricep_extension":

            if r_elbow>l_elbow:
                elbow=r_elbow
                wrist=_get_lm(kps,R_WRIST)
            else:
                elbow=l_elbow
                wrist=_get_lm(kps,L_WRIST)

            nose=_get_lm(kps,NOSE)

            shoulder_mid=(_get_lm(kps,L_SHOULDER)+_get_lm(kps,R_SHOULDER))/2
            hip_mid=(_get_lm(kps,L_HIP)+_get_lm(kps,R_HIP))/2
            torso=np.linalg.norm(shoulder_mid-hip_mid)+1e-8

            near_head=(np.linalg.norm(wrist-nose)/torso)<1.5

            if self.last_elbow is None:
                self.last_elbow=elbow

            elbow=0.5*self.last_elbow+0.5*elbow
            self.last_elbow=elbow

            self.min_elbow=min(self.min_elbow,elbow)
            self.max_elbow=max(self.max_elbow,elbow)

            rom=self.max_elbow-self.min_elbow

            if elbow>140:
                self.phase="UP"
            elif elbow<115:
                self.phase="DOWN"

            if prev=="DOWN" and self.phase=="UP" and near_head and rom>35:
                self.reps+=1
                self.min_elbow=elbow
                self.max_elbow=elbow

            return

        # ===== OTHER EXERCISES =====
        if ex=="front_raise":
            if wrist_y < shoulder_y:
                self.phase="UP"
            elif wrist_y > shoulder_y+0.10:
                self.phase="DOWN"

        elif ex=="lateral_raise":
            if wrist_y < shoulder_y:
                self.phase="UP"
            elif wrist_y > shoulder_y+0.10:
                self.phase="DOWN"

        elif ex=="shoulder_press":
            if wrist_y < nose_y:
                self.phase="UP"
            elif wrist_y > shoulder_y:
                self.phase="DOWN"

        elif ex=="push_up":
            if elbow_avg>145:
                self.phase="UP"
            elif elbow_avg<95:
                self.phase="DOWN"

        elif ex=="pull_up":
            if nose_y<wrist_y:
                self.phase="UP"
            elif nose_y>wrist_y+0.08:
                self.phase="DOWN"

        elif ex=="bench_pressing":
            if elbow_avg>150:
                self.phase="UP"
            elif elbow_avg<85:
                self.phase="DOWN"

        if prev=="DOWN" and self.phase=="UP":
            self.reps+=1

    def _state(self):
        return {
            'exercise':self.exercise,
            'reps':self.reps,
            'confidence':round(self.confidence,3),
            'phase':self.phase
        }
