import torch
import torch.nn.functional as F
from model_gen import build_gen_model
import pretty_midi
import numpy as np
import random
from collections import deque

device = "cpu"

_last_velocity_jitter = 0.0

BAR_LENGTH_STEPS = 16
HIGH_PITCH_THRESHOLD = 70
LOW_PITCH_THRESHOLD = 48
LEGATO_OVERLAP_RATIO = 0.08
LEGATO_OVERLAP_BASE_SEC = 0.1
LEGATO_OVERLAP_MIN_SEC = 0.005
LEGATO_OVERLAP_MAX_SEC = 0.010
TREBLE_LEGATO_OVERLAP_SEC = 0.010
LOW_PITCH_SUSTAIN_MULT = 1.2
MAX_NOTES_PER_BAR = 8
PAD_PITCH = 0
BREATHING_PITCH = 36
BREATHING_VELOCITY = 30

VELOCITY_JITTER_RANGE = 10
VELOCITY_JITTER_MOMENTUM = 0.6

HIGH_VOICE_PITCH_SPLIT = 64
LOW_VOICE_PITCH_SPLIT = 50
HIGH_VOICE_DUR_MIN = 0.25
HIGH_VOICE_DUR_MAX = 0.5
LOW_VOICE_DUR_MIN = 1.0
LOW_VOICE_DUR_MAX = 2.0
LOW_VOICE_VELOCITY_MULT = 0.7
HIGH_VOICE_JITTER_RANGE = 16

BASS_PATTERN_ROOT_BEAT = 1
BASS_PATTERN_FIFTH_BEAT = 3
BASS_PATTERN_BASE_OCTAVE = 3
BASS_PATTERN_VELOCITY = 45

TREBLE_STEP_SEMITONES_MIN = 1
TREBLE_STEP_SEMITONES_MAX = 2
TREBLE_STEP_BONUS = 0.5
TREBLE_BIG_JUMP_SEMITONES = 5
TREBLE_BIG_JUMP_MULT = 0.5
C_TREBLE_THIRD_SEMITONES = 3
C_TREBLE_THIRD_BONUS = 0.15

MELODIC_SMOOTH_FAR_SEMITONES = 7
MELODIC_SMOOTH_FAR_MULT = 0.7
MELODIC_SMOOTH_STEP_SEMITONES = 2
MELODIC_SMOOTH_STEP_MULT = 1.2

INTERVAL_TREND_MAX_SEMITONES = 5
INTERVAL_TREND_BIAS_WEIGHT = 0.2

# =============================================================================
# 1. 调式约束 (Key Constraint) - 支持关系小调
# =============================================================================

# 定义常见的音阶 (相对于 MIDI pitch 0)
SCALES = {
    "C_major": [0, 2, 4, 5, 7, 9, 11],      # C D E F G A B
    "A_minor": [9, 11, 0, 2, 4, 5, 7],      # A B C D E F G
    "G_major": [7, 9, 11, 0, 2, 4, 6],      # G A B C D E F#
    "E_minor": [4, 6, 7, 9, 11, 0, 2],      # E F# G A B C D
    "D_major": [2, 4, 6, 7, 9, 11, 1],      # D E F# G A B C#
    "B_minor": [11, 1, 2, 4, 6, 7, 9],      # B C# D E F# G A
    "chromatic": list(range(12)),            # 全色阶（无约束）
}

# 关系小调映射（同一组音符，不同的调心）
RELATIVE_MINOR_MAP = {
    "C_major": "A_minor",    # 都用 [0, 2, 4, 5, 7, 9, 11]
    "G_major": "E_minor",    # 都用 [0, 2, 4, 5, 7, 9, 11]
    "D_major": "B_minor",    # 都用 [0, 2, 4, 5, 7, 9, 11]
    "chromatic": "chromatic",
}

def apply_key_constraint(pitch, key="C_major", root_octave=4):
    """
    将音符约束到特定调式内
    
    Args:
        pitch: MIDI pitch值 (0-127)
        key: 调式名称 (C_major, A_minor, etc.)
        root_octave: 根音所在八度 (通常4)
    
    Returns:
        约束后的pitch
    """
    if pitch == PAD_PITCH:
        return pitch
    if key not in SCALES:
        return pitch
    
    scale = SCALES[key]
    pitch_class = pitch % 12
    octave = pitch // 12
    
    # 检查是否已在调式内
    if pitch_class in scale:
        return pitch
    
    candidates = []
    for octv in (octave - 1, octave, octave + 1):
        candidates.extend([note + octv * 12 for note in scale])
    nearest = min(candidates, key=lambda x: abs(x - pitch))
    
    # 确保在有效范围 [21, 108] (钢琴音域)
    nearest = max(21, min(108, nearest))
    return nearest

# =============================================================================
# 2. 节奏模板 (Rhythmic Templates)
# =============================================================================

RHYTHM_PATTERNS = {
    # 4/4 时间 的常见模式 (quarter notes = 1单位)
    "steady": [1.0] * 8,  # 8个均匀四分音符
    "waltz": [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5],  # 3/4拍的圆舞曲
    "dotted": [1.5, 0.5, 1.0, 1.0, 1.5, 0.5],  # 附点节奏
    "swing": [1.2, 0.8, 1.0, 1.0, 1.2, 0.8, 1.0],  # 摇摆节奏
    "syncopation": [0.5, 1.0, 0.5, 1.0, 1.0, 1.0],  # 切分节奏
}

# =============================================================================
# 6. ABA 三段式结构 - 关系小调版本 (ABA Form + Relative Minor Transition)
# =============================================================================

class ABASectionControllerV2:
    """
    增强的ABA三段式结构，支持关系小调转换和灰度过渡 + 主题锚点
    A段：大调（明亮）
    B段：关系小调（忧郁）
    A'段：回到大调（明亮），每4步强制锚定A段的原始旋律
    过渡区：灰度渐变，从大调到小调再回到大调
    """
    
    def __init__(self, section_length=32, key="C_major", fade_length=4):
        self.section_length = section_length
        self.total_length = section_length * 3
        
        # 调式信息
        self.major_key = key
        self.minor_key = RELATIVE_MINOR_MAP.get(key, key)
        self.major_scale = SCALES[self.major_key]
        self.minor_scale = SCALES[self.minor_key]
        
        # 灰度过渡参数
        self.fade_length = fade_length  # 过渡步数
        
        # A段缓存：用于A'段的主题锚定
        self.motif_pitches = []
        self.motif_intervals = []
        self.section_a_pitches = []  # 完整A段的所有pitch（用于A'段锚定）
        self.section_a_durations = []  # 完整A段的所有duration（用于节奏共享）
        
        # 主题锚定参数
        self.anchor_interval = 4  # 每4步作为一个锚点

        self.scalic_flow_active = False
        self.scalic_flow_direction = 0
        self.scalic_flow_last_pitch = None
        self.scalic_flow_steps_left = 0
        
        print(f"   [Major-Minor Conversion] {self.major_key} (Major) <-> {self.minor_key} (Minor)")
        print(f"   [Fade Transition] {fade_length} steps")
        print(f"   [Theme Anchoring] Every {self.anchor_interval} steps in A' will lock to A's motif\n")

    def _next_scale_step(self, pitch, scale, direction):
        pitch = int(pitch)
        direction = 1 if direction >= 0 else -1
        scale_set = set(scale)
        for offset in range(1, 13):
            candidate = pitch + direction * offset
            if candidate % 12 in scale_set:
                return candidate
        return pitch + direction

    def maybe_apply_scalic_flow(self, global_step, previous_pitch, section, transition_state):
        if previous_pitch is None or previous_pitch == PAD_PITCH:
            return None

        in_a_to_b_window = global_step < self.section_length + self.fade_length
        if not in_a_to_b_window:
            self.scalic_flow_active = False
            return None

        if transition_state not in {"fade_to_b", "fade_in"}:
            self.scalic_flow_active = False
            return None

        if not self.scalic_flow_active:
            self.scalic_flow_active = True
            self.scalic_flow_direction = random.choice([-1, 1])
            self.scalic_flow_last_pitch = int(previous_pitch)
            self.scalic_flow_steps_left = self.fade_length

        if self.scalic_flow_steps_left <= 0:
            self.scalic_flow_active = False
            return None

        next_pitch = self._next_scale_step(self.scalic_flow_last_pitch, self.minor_scale, self.scalic_flow_direction)
        self.scalic_flow_last_pitch = next_pitch
        self.scalic_flow_steps_left -= 1
        return apply_key_constraint(next_pitch, self.minor_key)
    
    def get_section_and_transition(self, global_step):
        """
        确定当前段落和过渡状态
        返回: (section, transition_state, progress_in_transition)
        """
        # A段纯粹区间
        if global_step < self.section_length - self.fade_length:
            return "A", "pure", 0.0
        
        # A→B过渡区间 (灰度淡出大调，淡入小调)
        elif global_step < self.section_length:
            progress = (global_step - (self.section_length - self.fade_length)) / self.fade_length
            return "A", "fade_to_b", progress
        
        # B段开始过渡区间 (继续淡入小调)
        elif global_step < self.section_length + self.fade_length:
            progress = (global_step - self.section_length) / self.fade_length
            return "B", "fade_in", progress
        
        # B/C段纯粹区间
        elif global_step < self.section_length * 2 - self.fade_length:
            b_start = self.section_length
            b_end = b_start + self.section_length // 2
            if global_step < b_end:
                return "B", "pure", 0.0
            return "C", "pure", 0.0
        
        # B→A'过渡区间 (灰度淡出小调，淡入大调)
        elif global_step < self.section_length * 2:
            progress = (global_step - (self.section_length * 2 - self.fade_length)) / self.fade_length
            return "B", "fade_to_a", progress
        
        # A'段开始过渡区间 (继续淡入大调)
        elif global_step < self.section_length * 2 + self.fade_length:
            progress = (global_step - self.section_length * 2) / self.fade_length
            return "A'", "fade_in", progress
        
        # A'段纯粹区间
        else:
            return "A'", "pure", 0.0
    
    def blend_pitch_with_transition(self, major_pitch, minor_pitch, fade_progress):
        """
        在大调和小调之间混合音高
        fade_progress: 0.0 = 100%大调, 1.0 = 100%小调
        """
        # 约束到对应的调式
        major_constrained = apply_key_constraint(major_pitch, self.major_key)
        minor_constrained = apply_key_constraint(minor_pitch, self.minor_key)
        
        # 概率混合：根据过渡进度选择
        if np.random.random() < fade_progress:
            return minor_constrained
        else:
            return major_constrained
    
    def get_transition_temperature(self, global_step, base_temperature, transition_state, progress):
        """
        在过渡区间增加温度，增强过渡的音乐效果
        """
        if transition_state == "pure":
            return base_temperature
        
        # 在过渡中增加一些温度波动
        transition_boost = 0.15 * (1 - abs(progress - 0.5) * 2)  # 中间最高
        return base_temperature + transition_boost
    
    def apply_pitch_transition(self, next_pitch, global_step, section, transition_state, fade_progress):
        """
        应用段落和过渡的复合逻辑
        """
        # 纯粹段落：直接约束到调式
        if transition_state == "pure":
            if section in ["A", "A'"]:
                return apply_key_constraint(next_pitch, self.major_key)
            else:  # B
                return apply_key_constraint(next_pitch, self.minor_key)
        
        # 灰度过渡：混合两个调式
        elif transition_state in ["fade_to_b", "fade_to_a", "fade_in"]:
            major_pitch = apply_key_constraint(next_pitch, self.major_key)
            minor_pitch = apply_key_constraint(next_pitch, self.minor_key)
            
            # 根据过渡方向调整fade_progress
            if transition_state == "fade_to_b":
                # A→B: 从large(0)向小调(1)过渡
                blend_progress = fade_progress
            elif transition_state == "fade_in":
                # B→A'或逐渐进入纯B: 从0向小调过渡
                blend_progress = fade_progress
            else:  # fade_to_a
                # B→A': 从小调(1)向大调(0)过渡
                blend_progress = 1.0 - fade_progress
            
            # 混合
            return self.blend_pitch_with_transition(major_pitch, minor_pitch, blend_progress)
        
        return next_pitch
    
    def record_motif(self, pitch):
        """在A段时，记录主题动机"""
        self.motif_pitches.append(pitch)
        if len(self.motif_pitches) > 1:
            interval = pitch - self.motif_pitches[-2]
            self.motif_intervals.append(interval)
    
    def record_section_a(self, pitch, duration):
        """完整记录A段的pitch和duration，用于A'段的主题锚定和节奏共享"""
        self.section_a_pitches.append(pitch)
        self.section_a_durations.append(duration)
    
    def get_theme_anchor_pitch(self, step_in_aprime):
        """
        获取A'段在该位置应该锚定的A段音符
        逻辑：每4步强制取A段对应位置的原始pitch作为跳板点
        在锚点之间的步数让模型自由生成变奏
        """
        # 计算对应A段位置（考虑到A'段可能略短）
        anchor_step = step_in_aprime
        
        if anchor_step < len(self.section_a_pitches):
            return self.section_a_pitches[anchor_step]
        else:
            # A'段比A段长，返回A段最后一个pitch
            return self.section_a_pitches[-1] if self.section_a_pitches else 60
    
    def should_use_anchor(self, step_in_aprime):
        """
        判断当前步数是否应该使用锚点（主题强制匹配）
        每4步作为锚点
        """
        return (step_in_aprime % self.anchor_interval) == 0
    
    def get_shared_rhythm_duration(self, step_in_aprime):
        """
        获取A'段在该位置应该使用的时值
        从A段的对应位置获取相同的节奏
        """
        if step_in_aprime < len(self.section_a_durations):
            return self.section_a_durations[step_in_aprime]
        else:
            return self.section_a_durations[-1] if self.section_a_durations else 0.5
    
    def calculate_pitch_similarity(self, section_aprime_pitches):
        if not section_aprime_pitches:
            return {
                "interval_match_ratio": 0.0,
                "pitch_range_diff": 0,
                "contour_similarity": 0.0,
                "mean_pitch_a": np.mean(self.section_a_pitches) if self.section_a_pitches else 0,
                "mean_pitch_aprime": 0,
                "overall_score": 0.0,
            }

        metrics = compute_intervalic_similarity(self.section_a_pitches, section_aprime_pitches)
        overall_score = (
            metrics["interval_match_ratio"] * 0.4
            + (1.0 - min(metrics["pitch_range_diff"] / 20.0, 1.0)) * 0.3
            + metrics["contour_similarity"] * 0.3
        )
        metrics["overall_score"] = overall_score
        return metrics
    
    def get_rhythm_pattern_for_section(self, global_step):
        """为不同段落返回不同的节奏模式"""
        section, transition_state, _ = self.get_section_and_transition(global_step)
        
        if section == "A":
            return "dotted"  # 舒缓
        elif section == "B":
            return "swing"   # 密集
        else:  # A'
            return "dotted"  # 舒缓
    
    def get_temperature_for_section(self, global_step, strategy="arch"):
        """为不同段落调整温度曲线"""
        section, transition_state, fade_progress = self.get_section_and_transition(global_step)
        steps_in_section = global_step % self.section_length
        local_progress = steps_in_section / self.section_length
        
        # 基础温度
        if section == "A":
            base_temp = 0.6 + 0.2 * local_progress
        elif section == "B":
            base_temp = 1.0 + 0.3 * np.sin(np.pi * local_progress)
        else:  # A'
            base_temp = 0.9 - 0.2 * local_progress
        
        # 过渡时增加温度
        return self.get_transition_temperature(global_step, base_temp, transition_state, fade_progress)

# =============================================================================
# 获取节奏时值 (Rhythm Duration)
# =============================================================================

def get_duration_from_template(step, pattern_name="steady", tempo=120):
    """
    从节奏模板获取时值
    
    Args:
        step: 生成步数
        pattern_name: 节奏模式名称
        tempo: BPM
    
    Returns:
        该步的持续时间 (秒)
    """
    patterns = RHYTHM_PATTERNS
    pattern = patterns.get(pattern_name, patterns["steady"])
    
    # 循环模式
    duration_ratio = pattern[step % len(pattern)]
    
    # 转换为秒 (假设 quarter note = 0.5s @ 120 BPM)
    quarter_note_duration = 60 / tempo * 1.0  # 秒
    return duration_ratio * quarter_note_duration * 0.5

# =============================================================================
# 3. 温度衰减采样 (Temperature Decay)
# =============================================================================

def get_dynamic_temperature(step, total_steps=100, strategy="arch"):
    """
    动态调整温度，产生"乐句感"
    
    Args:
        step: 当前生成步数
        total_steps: 总生成步数
        strategy: 
            - "arch": 拱形 (低->高->低)
            - "decay": 衰减 (高->低)
            - "fixed": 固定 (不变)
    
    Returns:
        该步的温度值
    """
    if strategy == "arch":
        # 在乐句中间最随机，开头和结尾保守
        t = step / total_steps
        return 0.7 + 0.4 * np.sin(np.pi * t)
    elif strategy == "decay":
        # 逐渐降低温度（稳定结束）
        return 1.0 * (1 - step / total_steps) + 0.5
    else:  # fixed
        return 1.0

# =============================================================================
# 4. Top-P 采样 (Nucleus Sampling)
# =============================================================================

def top_p_sampling(probs, p=0.95):
    """
    Nucleus (Top-P) 采样：累积概率达到p时停止
    
    Args:
        probs: 概率分布 [vocab_size]
        p: 累积概率阈值 (0.9 = 90%)
    
    Returns:
        采样的token索引
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cumsum_probs > p
    
    # 确保至少保留前1个token
    sorted_indices_to_remove[0] = False
    
    # 将超过p的概率设为0
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # 重新归一化
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    # 按重新排序的概率采样
    sampled_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sampled_idx].item()

# =============================================================================
# 5. 动机缓存与重复 (Motif Caching & Repetition)
# =============================================================================

class MotifCache:
    """缓存短乐句片段，用于生成重复结构"""
    
    def __init__(self, motif_len=4):
        self.motif_len = motif_len
        self.cache = deque(maxlen=motif_len * 2)  # 保存最近两小节
        self.repeat_prob = 0.2  # 20%概率重复
    
    def add(self, pitch):
        self.cache.append(pitch)
    
    def get_motif(self):
        """获取缓存的动机"""
        if len(self.cache) >= self.motif_len:
            return list(self.cache)[-self.motif_len:]
        return None
    
    def should_repeat(self):
        """决定是否重复"""
        return np.random.random() < self.repeat_prob and self.get_motif() is not None

# =============================================================================
# 情感表达层 (Affective Expression Layer)
# =============================================================================

# 大七和弦和挂留四和弦的典型音程差（相对于根音）
HARMONIC_INTERVALS = {
    "maj7": [0, 4, 7, 11],        # 大七和弦: 根 + 大三度 + 完全五度 + 大七度
    "sus4": [0, 5, 7],             # 挂留四和弦: 根 + 完全四度 + 完全五度
    "min7": [0, 3, 7, 10],         # 小七和弦: 根 + 小三度 + 完全五度 + 小七度
    "dominant7": [0, 4, 7, 10],   # 属七和弦: 根 + 大三度 + 完全五度 + 小七度
}

HARMONIC_ALLOWED_INTERVALS = sorted({0, 5, 7} | {i for v in HARMONIC_INTERVALS.values() for i in v})

# =============================================================================
# 4/4 拍节拍器 (Metric Grid 4/4) 和强弱律动
# =============================================================================

def get_metric_grid_position(step, bar_length=BAR_LENGTH_STEPS):
    """
    获取当前 Step 在 4/4 拍中的位置
    每 16 个 Step = 1 Bar（小节）
    
    Args:
        step: 全局步数
        bar_length: 每小节的步数（16）
    
    Returns:
        dict: {bar_num, beat_in_bar, is_strong, is_secondary}
    """
    bar_num = step // bar_length
    beat_in_bar = step % bar_length
    
    # 4/4 拍的强弱分布（16步分为4拍）
    # Step 0（拍1）：强（beat 1）
    # Step 4（拍2）：弱（beat 2）
    # Step 8（拍3）：弱（beat 3）
    # Step 12（拍4）：弱（beat 4）
    # 但在我们的系统中转换为：
    # Step 0：强（+20 vel）
    # Step 4/12：次强（+10 vel）
    # 其他：弱
    
    is_strong = beat_in_bar == 0  # Step 0 = 强
    is_secondary = beat_in_bar in [4, 12]  # Step 4, 12 = 次强
    
    return {
        "bar_num": bar_num,
        "beat_in_bar": beat_in_bar,
        "is_strong": is_strong,
        "is_secondary": is_secondary,
        "beat_position": beat_in_bar // 4 + 1,  # 1-4 拍
    }

# =============================================================================
# 意境降维 - 三件套 (2026 Artistic Enhancement)
# =============================================================================

# ============ 1. 力度呼吸化 (Velocity Humanization) ============

def apply_velocity_humanization(base_velocity, jitter_range=VELOCITY_JITTER_RANGE):
    """
    随机抖动力度，模拟人类的自然呼吸感
    
    Args:
        base_velocity: 基础力度值 (0-127)
    
    Returns:
        加入随机抖动后的力度值
    """
    global _last_velocity_jitter
    raw_jitter = random.uniform(-jitter_range, jitter_range)
    jitter = _last_velocity_jitter * VELOCITY_JITTER_MOMENTUM + raw_jitter * (1.0 - VELOCITY_JITTER_MOMENTUM)
    _last_velocity_jitter = jitter
    humanized_velocity = base_velocity + jitter
    return max(30, min(127, int(humanized_velocity)))

def track_pitch_curve(pitch_history, current_pitch, window_size=4):
    """
    追踪音符走势曲线，判断是否处于上升或下降趋势
    
    Args:
        pitch_history: 历史音符列表（最多window_size个）
        current_pitch: 当前音符
        window_size: 观察窗口大小
    
    Returns:
        {"trend": "up"|"down"|"stable", "strength": 0-1 (强度系数)}
    """
    if len(pitch_history) < 2:
        return {"trend": "stable", "strength": 0}
    
    # 取最近 window_size 个音符
    recent = list(pitch_history[-window_size:]) + [current_pitch]
    
    # 计算上升/下降的步数
    up_steps = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    down_steps = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
    
    if up_steps >= down_steps + 1:
        # 倾向上升
        strength = min(1.0, up_steps / (window_size - 1))
        return {"trend": "up", "strength": strength}
    elif down_steps >= up_steps + 1:
        # 倾向下降
        strength = min(1.0, down_steps / (window_size - 1))
        return {"trend": "down", "strength": strength}
    else:
        # 稳定/混合
        return {"trend": "stable", "strength": 0.5}

def apply_velocity_curve(base_velocity, pitch_curve):
    """
    根据音符走势曲线调整力度
    
    上升时增强力度，下降时削弱力度
    
    Args:
        base_velocity: 基础力度
        pitch_curve: 从 track_pitch_curve 返回的字典
    
    Returns:
        调整后的力度值
    """
    strength = pitch_curve["strength"]
    
    if pitch_curve["trend"] == "up":
        # 上升时增强，最多增加 strength * 25%
        velocity = base_velocity * (1.0 + strength * 0.25)
    elif pitch_curve["trend"] == "down":
        # 下降时削弱，最多削弱 strength * 20%
        velocity = base_velocity * (1.0 - strength * 0.20)
    else:
        # 稳定时保持
        velocity = base_velocity
    
    return max(30, min(127, int(velocity)))

# ============ 2. 深海低音织体 (Deep Sea Texture) ============

class DeepSeaBassController:
    """
    管理深海低音"底色"的生成
    - 每2小节强制一个长低音（3拍以上）
    - 低音音符生成时增加Top-P权重
    """
    
    def __init__(self):
        self.bar_count = 0
        self.last_bass_bar = -3  # 上一次低音距离（初始化为>2）
        self.is_bass_mode = False
        self.bass_note_duration = 0
    
    def should_force_bass(self, step, bar_length=BAR_LENGTH_STEPS):
        """
        判断是否应该强制生成低音
        
        每2小节（32步）强制一次低音底色
        """
        bar_num = step // bar_length
        
        # 每2小节强制一次
        if bar_num - self.last_bass_bar >= 2:
            self.is_bass_mode = True
            self.last_bass_bar = bar_num
            self.bass_note_duration = 0
            return True
        
        return False
    
    def get_bass_duration(self):
        """获取当前低音音符应该持续的长度"""
        # 低音持续 3+ 拍（每拍4步，所以 12+ 步）
        return random.uniform(0.75, 1.2)  # 3-5拍对应0.75-1.2秒
    
    def is_in_bass_mode(self):
        """当前是否处于低音生成模式"""
        return self.is_bass_mode
    
    def finalize_bass(self):
        """结束低音模式"""
        self.is_bass_mode = False

def boost_low_pitch_probability(logits, current_pitch=None, boost_factor=2.0):
    """
    增强低音（Pitch < 48）的生成概率
    
    通过调整 logits 来增加低音的采样机会
    
    Args:
        logits: 模型输出的 logits (向量)
        current_pitch: 当前音高（未使用，留作扩展）
        boost_factor: 增强系数 (2.0 = 概率翻倍)
    
    Returns:
        增强后的 logits
    """
    # 低音范围：0-47
    logits_modified = logits.clone()
    logits_modified[:48] = logits_modified[:48] * boost_factor
    
    return logits_modified

# ============ 3. 艺术留白 (The Art of Silence) ============

def should_add_silence(step, bar_length=BAR_LENGTH_STEPS, silence_probability=0.30):
    """
    判断是否应该在此位置添加休止符（艺术留白）
    
    每小节的第4拍（Step % bar_length == 12）有 silence_probability 的概率休止
    
    Args:
        step: 当前步数
        bar_length: 每小节的步数 (默认16)
        silence_probability: 休止概率 (0.30 = 30%)
    
    Returns:
        True 如果应该添加休止符
    """
    # 计算当前位置在小节中的位置
    beat_in_bar = step % bar_length
    beat_num = beat_in_bar // 4 + 1  # 1-4 拍（每拍4步）
    
    # 只在第4拍考虑休止
    if beat_num == 4:
        return random.random() < silence_probability
    
    return False

def create_rest_note(duration=0.25):
    """
    创建一个休止符（实际是一个极低力度、不发声的音符）
    
    Args:
        duration: 休止符持续时间（秒）
    
    Returns:
        (pitch=0, duration, velocity=0)
    """
    return 0, duration, 0

# =============================================================================
# v2.1 连奏灵魂优化 (Legato Soul Enhancement)
# =============================================================================

# ============ 1. 虚拟踏板逻辑 (Sustain Pedal Logic) ============

class VirtualSustainPedalController:
    """
    管理虚拟延音踏板逻辑
    - 低音（Pitch < 50）强制延续至下一个低音出现为止
    - 模拟钢琴延音踏板的效果
    """
    
    def __init__(self):
        self.last_bass_pitch = None
        self.last_bass_duration = 0
        self.sustained_bass_end_time = 0.0
    
    def update_bass_sustain(self, pitch, duration, current_time):
        """
        更新低音续音状态
        
        Args:
            pitch: 当前音高
            duration: 基础持续时间
            current_time: 当前时刻（秒）
        
        Returns:
            adjusted_duration: 考虑延续的调整后时值
        """
        if pitch < 50:
            # 这是低音
            self.last_bass_pitch = pitch
            self.last_bass_duration = duration
            # 低音延续到下一个低音出现为止，延长1.5倍
            adjusted_duration = duration * 1.5
            self.sustained_bass_end_time = current_time + adjusted_duration
            return adjusted_duration
        else:
            # 非低音
            # 检查是否还在上一个低音的延续时间内
            # 如果是，保持低音的余音但不额外延长
            return duration
    
    def is_sustaining_bass(self):
        """当前是否仍在延续低音"""
        return self.last_bass_pitch is not None

# ============ 2. 力度动量追踪器 (Velocity Momentum Tracker) ============

class VelocityMomentumTracker:
    """
    追踪力度的连续变化趋势（动量）
    - 不在每步都抖动，而是有"惯性"的平滑变化
    - 跨越至少半个小节（8步）的渐强或渐弱过程
    """
    
    def __init__(self, window_size=8):
        self.window_size = window_size  # 半个小节
        self.velocity_history = deque(maxlen=window_size)
        self.momentum = 0  # -1 ~ 1，表示趋势
        self.target_velocity = 100
    
    def update_momentum(self, current_velocity):
        """
        更新动量，计算平滑的力度包络
        
        Args:
            current_velocity: 当前基础力度
        
        Returns:
            momentum_adjusted_velocity: 经过动量平滑的力度
        """
        self.velocity_history.append(current_velocity)
        
        if len(self.velocity_history) < 2:
            return current_velocity
        
        # 计算趋势：过去历史相对于前一半历史的变化
        if len(self.velocity_history) >= self.window_size // 2:
            recent_half = list(self.velocity_history)[-(self.window_size // 2):]
            older_half = list(self.velocity_history)[:-(self.window_size // 2) if len(self.velocity_history) > self.window_size // 2 else len(self.velocity_history) // 2]
            
            if older_half:
                trend = np.mean(recent_half) - np.mean(older_half)
                self.momentum = np.clip(trend / 100, -1, 1)
        
        # 基于动量调整力度（给予"惯性"）
        if self.momentum > 0:
            # 渐强中
            adjustment = self.momentum * 15  # 最多增加 15
        elif self.momentum < 0:
            # 渐弱中
            adjustment = self.momentum * 15  # 最多减少 15
        else:
            # 稳定
            adjustment = 0
        
        adjusted = int(current_velocity + adjustment)
        return max(30, min(127, adjusted))

# ============ 3. 柔化留白边缘 (Soften Silence Edge) ============

def should_add_selective_silence(step, pitch, bar_length=BAR_LENGTH_STEPS, silence_probability=0.30, preserve_bass=True):
    """
    改进的留白逻辑：柔化边缘并保留低音
    
    If preserve_bass=True:
      - 高音旋律线休止（pitch >= 50会产生silence）
      - 低音轨道继续鸣响，营造"余音绕梁"效果
    
    If preserve_bass=False:
      - 完全休止（回到v2.0行为）
    
    Args:
        step: 当前步数
        pitch: 当前音高
        bar_length: 每小节步数
        silence_probability: 休止概率
        preserve_bass: 是否保留低音轨道
    
    Returns:
        tuple: (should_silence, silence_type)
            - should_silence: 是否应该产生静音
            - silence_type: 'full' | 'high_only' | 'none'
    """
    beat_in_bar = step % bar_length
    beat_num = beat_in_bar // 4 + 1
    
    if beat_num == 4 and random.random() < silence_probability:
        if preserve_bass and pitch < 50:
            # 低音保留
            return False, 'none'
        elif preserve_bass and pitch >= 50:
            # 只有高音休止
            return True, 'high_only'
        else:
            # 完全休止
            return True, 'full'
    
    return False, 'none'

def create_selective_rest_note(silence_type='full', duration=0.25, bass_pitch=None):
    """
    创建选择性的休止符
    
    Args:
        silence_type: 'full' | 'high_only' | 'none'
        duration: 休止符时长
        bass_pitch: 如果silence_type='high_only'，保留的低音音高
    
    Returns:
        tuple: (pitch, duration, velocity)
    """
    if silence_type == 'full':
        return 0, duration, 0  # 完全沉默
    elif silence_type == 'high_only' and bass_pitch is not None:
        # 保留低音，但力度降低（余音效果）
        return bass_pitch, duration * 0.5, 45  # 低力度的低音续音
    else:
        return 0, duration, 0

# ============ 4. 连奏音符重叠 (Legato Note Overlap) ============

def apply_legato_overlap(duration, next_pitch_is_same_or_close=False, overlap_ratio=LEGATO_OVERLAP_RATIO):
    """
    添加连奏重叠效果
    - 音符与下一个音符产生5-10%的时间重叠
    - 模拟钢琴连奏（Legato）的连贯性
    
    Args:
        duration: 原始音符时值
        next_pitch_is_same_or_close: 下一个音符是否相近（音程≤2半音）
        overlap_ratio: 重叠比例 (0.05-0.10)
    
    Returns:
        adjusted_duration: 加入重叠后的时值
    """
    if next_pitch_is_same_or_close:
        # 相邻音符相近时，重叠更明显
        overlap_factor = 1.0 + overlap_ratio * 1.5
    else:
        # 音程较大时，重叠较温和
        overlap_factor = 1.0 + overlap_ratio
    
    return duration * overlap_factor

def compute_pitch_proximity(current_pitch, next_pitch, threshold=2):
    """
    计算两个音符的接近程度（半音距离）
    
    Args:
        current_pitch: 当前音符
        next_pitch: 下一个音符
        threshold: 认定为"接近"的阈值（半音）
    
    Returns:
        bool: 是否在阈值内
    """
    if next_pitch is None:
        return False
    interval = abs(current_pitch - next_pitch)
    return interval <= threshold

def calculate_legato_adjusted_duration(duration, pitch_history, current_pitch, next_pitch=None):
    """
    综合计算连奏调整的时值
    
    Args:
        duration: 基础时值
        pitch_history: 音符历史
        current_pitch: 当前音符
        next_pitch: 下一个音符（如果已知）
    
    Returns:
        legato_duration: 应用连奏调整后的时值
    """
    # 递推至少看历史中与当前音符接近的音符
    proximity_to_recent = False
    if len(pitch_history) > 0:
        proximity_to_recent = compute_pitch_proximity(current_pitch, pitch_history[-1])
    
    # 基础连奏重叠
    legato_duration = apply_legato_overlap(duration, proximity_to_recent, overlap_ratio=LEGATO_OVERLAP_RATIO)
    
    return legato_duration

def calculate_dynamic_velocity(pitch, section, transition_state, pitch_history, step, section_length, use_metric_grid=True, global_step=0, total_steps=96, use_velocity_humanization=True, velocity_momentum_tracker=None):
    """
    计算动态力度（Velocity），支持 4/4 拍节拍器、Ritardando 和力度呼吸化
    
    Args:
        pitch: 当前音高
        section: 当前段落（A/B/A'）
        transition_state: 过渡状态
        pitch_history: 历史音高列表
        step: 当前步数
        section_length: 段落长度
        use_metric_grid: 是否使用 4/4 拍节拍器
        global_step: 全局步数（用于 Ritardando）
        total_steps: 总步数
        use_velocity_humanization: 是否启用力度呼吸化（新增）
    
    Returns:
        力度值 (0-127)
    """
    base_velocity = 90
    
    # 1. 段落调整：B段（小调）整体下调15%
    if section == "B":
        base_velocity = int(base_velocity * 0.85)  # 下调15%
    elif section == "A'":
        base_velocity = int(base_velocity * 0.95)  # A'段略微下调，体现温暖感
    
    # 2. 4/4 拍节拍器强弱律动
    if use_metric_grid:
        metric = get_metric_grid_position(step, bar_length=BAR_LENGTH_STEPS)
        if metric["is_strong"]:
            base_velocity = int(base_velocity * 1.22)  # 约 +20
        elif metric["is_secondary"]:
            base_velocity = int(base_velocity * 1.11)  # 约 +10
        else:
            pass
    else:
        local_step = step % 4
        if local_step in [0, 2]:
            base_velocity = int(base_velocity * 1.1)
        else:
            base_velocity = int(base_velocity * 0.8)
    
    # 3. 新增：力度曲线 - 根据音符走势调整力度
    if len(pitch_history) >= 2:
        pitch_curve = track_pitch_curve(pitch_history, pitch, window_size=4)
        base_velocity = apply_velocity_curve(base_velocity, pitch_curve)
    
    # 4. Ritardando - 最后 8 个音符逐渐降低力度
    if global_step > total_steps - 8:
        ritardando_progress = (global_step - (total_steps - 8)) / 8
        base_velocity = int(base_velocity * (1.0 - ritardando_progress * 0.3))
    
    # 5. 新增：力度呼吸化 - 随机抖动（人性化处理）
    if use_velocity_humanization:
        jitter_range = HIGH_VOICE_JITTER_RANGE if pitch > HIGH_VOICE_PITCH_SPLIT else VELOCITY_JITTER_RANGE
        base_velocity = apply_velocity_humanization(base_velocity, jitter_range=jitter_range)
    
    # 6. 新增（v2.1）：力度动量平滑 - 平滑的包络而非突兀变化
    if velocity_momentum_tracker is not None:
        base_velocity = velocity_momentum_tracker.update_momentum(base_velocity)
    
    # 7. 防止过饱和
    return max(40, min(120, base_velocity))

def is_harmonic_interval(current_pitch, previous_pitch):
    """
    检查两个音符是否构成和谐的和弦音程
    （针对"紫色调印象派"的高级品味）
    """
    if previous_pitch is None:
        return True
    
    # 计算音程差（取mod 12，忽略八度）
    interval = (current_pitch - previous_pitch) % 12
    
    # 检查是否在任何和弦的音程中
    for chord_intervals in HARMONIC_INTERVALS.values():
        if interval in chord_intervals:
            return True
    
    # 检查完全八度（0）、完全五度（7）、完全四度（5）
    if interval in [0, 5, 7]:
        return True
    
    return False

def apply_harmonic_bias(probs, previous_pitch, section=None):
    """
    对采样概率进行偏差调整，偏好和谐的音程
    这创造出"高级忧郁感"而不是直白的和谐
    """
    if previous_pitch is None:
        return probs

    previous_pitch = int(previous_pitch)

    allowed = torch.tensor(HARMONIC_ALLOWED_INTERVALS, device=probs.device)
    idx = torch.arange(probs.numel(), device=probs.device)
    intervals = (idx - previous_pitch) % 12
    harmonic_mask = (intervals.unsqueeze(-1) == allowed).any(-1)

    near_mask = torch.abs(idx - previous_pitch) <= INTERVAL_TREND_MAX_SEMITONES

    abs_distance = torch.abs(idx - previous_pitch)
    far_mask = abs_distance > MELODIC_SMOOTH_FAR_SEMITONES
    step_mask = abs_distance <= MELODIC_SMOOTH_STEP_SEMITONES

    pad_mask = idx == PAD_PITCH
    harmonic_mask = harmonic_mask & (~pad_mask)
    near_mask = near_mask & (~pad_mask)
    far_mask = far_mask & (~pad_mask)
    step_mask = step_mask & (~pad_mask)

    bias_multiplier = torch.full_like(probs, 0.95)
    bias_multiplier[harmonic_mask] = 1.2
    bias_multiplier[far_mask] = bias_multiplier[far_mask] * MELODIC_SMOOTH_FAR_MULT
    bias_multiplier[step_mask] = bias_multiplier[step_mask] * MELODIC_SMOOTH_STEP_MULT
    bias_multiplier[near_mask] = bias_multiplier[near_mask] + INTERVAL_TREND_BIAS_WEIGHT

    treble_mask = (idx > HIGH_VOICE_PITCH_SPLIT) & (~pad_mask)
    treble_step_mask = (
        treble_mask
        & (abs_distance >= TREBLE_STEP_SEMITONES_MIN)
        & (abs_distance <= TREBLE_STEP_SEMITONES_MAX)
    )
    treble_big_jump_mask = treble_mask & (abs_distance > TREBLE_BIG_JUMP_SEMITONES)
    bias_multiplier[treble_big_jump_mask] = bias_multiplier[treble_big_jump_mask] * TREBLE_BIG_JUMP_MULT
    bias_multiplier[treble_step_mask] = bias_multiplier[treble_step_mask] + TREBLE_STEP_BONUS

    if section == "C":
        treble_third_mask = treble_mask & (abs_distance == C_TREBLE_THIRD_SEMITONES)
        bias_multiplier[treble_third_mask] = bias_multiplier[treble_third_mask] + C_TREBLE_THIRD_BONUS

    bias_multiplier[pad_mask] = 1.0

    biased_probs = probs * bias_multiplier
    return biased_probs / biased_probs.sum()

def should_add_breathing_space(global_step, section_length, fade_length, section, transition_state):
    """
    在段落末尾检查是否需要加入呼吸空间（休止或长延音）
    仅在A段结尾和B段结尾各加一次
    """
    # 只在A→B和B→A的过渡最后一步加入呼吸
    if section == "A":
        step_in_section = global_step
    elif section == "B":
        step_in_section = global_step - section_length
    elif section == "A'":
        step_in_section = global_step - section_length * 2
    else:
        step_in_section = global_step

    is_end_of_a = (
        section == "A"
        and transition_state == "fade_to_b"
        and step_in_section == section_length - 1
    )
    is_end_of_b = (
        section == "B"
        and transition_state == "fade_to_a"
        and step_in_section == section_length - 1
    )
    
    return is_end_of_a or is_end_of_b

def get_staccato_legato_duration(pitch, base_duration):
    """
    根据音高调整时值（高音staccato，低音legato）
    
    Args:
        pitch: MIDI pitch值
        base_duration: 基础时值
    
    Returns:
        调整后的时值
    """
    if pitch > 72:  # 高音（C5以上）
        # Staccato：减少60%时值，营造"露珠"感觉
        return base_duration * 0.4
    elif pitch < LOW_PITCH_THRESHOLD:  # 低音（C3以下）
        # Legato：增加50%时值，营造"底色"稳重感
        return base_duration * 1.5
    else:  # 中音范围
        return base_duration

def add_breathing_pause(tempo=120):
    """
    添加呼吸暂停（2拍休止）
    返回应该跳过多少时间
    """
    # 计算2拍的时间
    quarter_note_duration = 60 / tempo
    return 2 * quarter_note_duration  # 2拍

def get_pitch_aware_temperature(pitch, base_temperature):
    """
    根据音高调整温度，实现'旋律优先'机制
    - 高音区（>=70）：降低温度到0.6，保持稳健性
    - 中音区（48-70）：使用基础温度
    - 低音区（<48）：提高温度，增加随机性和厚度感
    """
    if pitch >= HIGH_PITCH_THRESHOLD:
        # 高音区：更保守，避免跳跃过大
        return min(0.6, base_temperature)
    elif pitch < LOW_PITCH_THRESHOLD:
        # 低音区：更灵活，为低音增加表现力
        return base_temperature * 1.1
    else:
        # 中音区：使用基础温度
        return base_temperature

def check_note_density(pitches_in_measure, max_density=8):
    """
    检查当前小节的音符密度
    返回是否应该继续生成（False = 超过密度限制）
    
    Args:
        pitches_in_measure: 当前小节内的pitch列表
        max_density: 每小节最多音符数
    
    Returns:
        True = 可以继续，False = 达到密度上限
    """
    # 计算当前小节的合法音符数（velocity >= 30）
    # 这里简化为直接检查长度
    return len(pitches_in_measure) < max_density

def compute_intervalic_similarity(section_a_pitches, section_aprime_pitches):
    """
    计算A段和A'段的音程相似度
    
    Args:
        section_a_pitches: A段的pitch列表
        section_aprime_pitches: A'段的pitch列表
    
    Returns:
        dict: 包含多个相似度指标
            - interval_match_ratio: 共同音程的比例
            - pitch_range_diff: 音域范围的差异
            - contour_similarity: 轮廓相似度（上升/下降的一致性）
    """
    if len(section_a_pitches) < 2 or len(section_aprime_pitches) < 2:
        return {
            "interval_match_ratio": 0.0,
            "pitch_range_diff": 0,
            "contour_similarity": 0.0,
            "mean_pitch_a": np.mean(section_a_pitches) if section_a_pitches else 0,
            "mean_pitch_aprime": np.mean(section_aprime_pitches) if section_aprime_pitches else 0,
        }
    
    # 计算音程
    a_intervals = np.diff(section_a_pitches)
    aprime_intervals = np.diff(section_aprime_pitches)
    
    # 1. 音程相似度：共同音程的比例
    common_intervals = len(set(a_intervals) & set(aprime_intervals))
    total_intervals = max(len(set(a_intervals)), len(set(aprime_intervals)))
    interval_match_ratio = common_intervals / total_intervals if total_intervals > 0 else 0.0
    
    # 2. 音域范围差异
    a_range = max(section_a_pitches) - min(section_a_pitches)
    aprime_range = max(section_aprime_pitches) - min(section_aprime_pitches)
    pitch_range_diff = abs(a_range - aprime_range)
    
    # 3. 轮廓相似度（上升/下降方向的一致性）
    a_contour = np.sign(a_intervals)  # 1=上升, -1=下降, 0=相同
    aprime_contour = np.sign(aprime_intervals)
    
    # 取最小长度进行比较
    min_len = min(len(a_contour), len(aprime_contour))
    contour_matches = np.sum(a_contour[:min_len] == aprime_contour[:min_len])
    contour_similarity = contour_matches / min_len if min_len > 0 else 0.0
    
    return {
        "interval_match_ratio": interval_match_ratio,
        "pitch_range_diff": pitch_range_diff,
        "contour_similarity": contour_similarity,
        "mean_pitch_a": np.mean(section_a_pitches),
        "mean_pitch_aprime": np.mean(section_aprime_pitches),
    }

# =============================================================================
# 印象派织体 (Impressionist Texture)
# =============================================================================

def get_duration_with_impressionist_texture(pitch, base_duration, high_pitch_threshold=HIGH_PITCH_THRESHOLD, low_pitch_threshold=LOW_PITCH_THRESHOLD):
    """
    根据音高调整时值，创造'远近错落'的印象派空间感
    - 高音（>70）：较短的时值（Duration × 0.8）→ 飘动感
    - 低音（<48）：较长的时值（Duration × 1.6）→ 沉稳感
    - 中音：保持基础时值
    
    Args:
        pitch: MIDI pitch 值
        base_duration: 基础时值（秒）
        high_pitch_threshold: 高音阈值
        low_pitch_threshold: 低音阈值
    
    Returns:
        调整后的时值
    """
    if pitch > high_pitch_threshold:
        # 高音：短促、飘动
        return base_duration * 0.8
    elif pitch < low_pitch_threshold:
        # 低音：深邃、延伸
        return base_duration * 1.6
    else:
        # 中音：自然
        return base_duration

def enforce_high_low_alternation(recent_pitches, current_pitch, high_threshold=HIGH_PITCH_THRESHOLD, low_threshold=LOW_PITCH_THRESHOLD, max_consecutive_high=3):
    """
    印象派高低音分工：连续 3 个高音后强制引入低音
    
    Args:
        recent_pitches: 最近的 pitch 历史（最后 3-4 个）
        current_pitch: 当前生成的 pitch
        high_threshold: 高音定义（>= 70）
        low_threshold: 低音定义（< 48）
        max_consecutive_high: 最多连续高音数
    
    Returns:
        tuple: (adjusted_pitch, was_enforced)
            - adjusted_pitch: 可能被调整后的 pitch
            - was_enforced: 是否实施了强制低音
    """
    high_count = 0
    for pitch in recent_pitches[-max_consecutive_high:]:
        if pitch > high_threshold:
            high_count += 1
        else:
            high_count = 0
    
    # 如果连续高音数达到上限，强制生成低音
    if high_count >= max_consecutive_high:
        # 生成一个在低音区的音符
        # 使用约束到调式的低音
        low_pitch = current_pitch % 12 + 36  # 在低音区（36-47）
        return low_pitch, True
    
    return current_pitch, False

def limit_interval_by_mode(current_pitch, previous_pitch, section, major_scale, minor_scale):
    """
    大小调音程限制：创造情感增强
    
    Args:
        current_pitch: 当前音高
        previous_pitch: 前一个音高
        section: 段落（A/B/A'）
        major_scale: 大调音阶
        minor_scale: 小调音阶
    
    Returns:
        音程限制后的 pitch
    """
    if previous_pitch is None or previous_pitch == 0:
        return current_pitch
    
    interval = abs(current_pitch - previous_pitch)
    
    if section in ["A", "A'"]:
        # A 段（大调）：允许宽大的音程跳跃（开阔感）
        # 不做限制，允许 > 8 度的跳跃
        return current_pitch
    elif section == "B":
        # B 段（小调）：限制音程跳跃在 4 度以内（内省感）
        max_interval = 4  # 4 个半音
        if interval > max_interval:
            # 找到一个距离更近的、在调式内的音符
            if current_pitch > previous_pitch:
                # 向上贴近，最多跳跃 4 个半音
                adjusted_pitch = previous_pitch + max_interval
            else:
                # 向下贴近
                adjusted_pitch = previous_pitch - max_interval
            return adjusted_pitch
    
    return current_pitch

def apply_ritardando_duration(base_duration, global_step, total_steps, ritardando_length=8):
    """
    渐缓（Ritardando）处理：最后 N 个音符逐渐放慢
    
    Args:
        base_duration: 基础时值
        global_step: 全局步数
        total_steps: 总步数
        ritardando_length: Ritardando 的长度（步数）
    
    Returns:
        调整后的时值
    """
    if global_step > total_steps - ritardando_length:
        # 在最后 N 个步骤中逐渐放慢
        ritardando_progress = (global_step - (total_steps - ritardando_length)) / ritardando_length
        # 逐渐增加时值（放慢）从 1.0 到 1.5
        duration_multiplier = 1.0 + ritardando_progress * 0.5
        return base_duration * duration_multiplier
    return base_duration


def get_generation_step_context(step, num_notes, temperature_strategy, aba_controller):
    if aba_controller:
        section, transition_state, fade_progress = aba_controller.get_section_and_transition(step)
        rhythm_pattern = aba_controller.get_rhythm_pattern_for_section(step)
        temperature = aba_controller.get_temperature_for_section(step, temperature_strategy)
        section_length = aba_controller.section_length
        fade_length = aba_controller.fade_length
        if section == "A'":
            step_in_aprime = step - section_length * 2
        else:
            step_in_aprime = 0
        return (
            section,
            transition_state,
            fade_progress,
            rhythm_pattern,
            temperature,
            section_length,
            fade_length,
            step_in_aprime,
        )

    section = None
    transition_state = None
    fade_progress = 0.0
    rhythm_pattern = "steady"
    temperature = get_dynamic_temperature(step, num_notes, temperature_strategy)
    section_length = num_notes
    fade_length = 0
    step_in_aprime = 0
    return (
        section,
        transition_state,
        fade_progress,
        rhythm_pattern,
        temperature,
        section_length,
        fade_length,
        step_in_aprime,
    )


def apply_density_temperature(temperature, recent_step_pitches):
    notes_in_bar = sum(1 for p in recent_step_pitches if p != PAD_PITCH)
    if notes_in_bar >= MAX_NOTES_PER_BAR:
        return min(temperature, 0.3)
    return temperature


def run_generation_loop(
    model,
    seed_pitches,
    num_notes,
    key,
    temperature_strategy,
    use_top_p,
    top_p,
    top_k,
    use_affective_layer,
    use_theme_anchoring,
    use_pitch_aware_temp,
    use_density_control,
    use_metric_grid,
    use_impressionist_texture,
    use_mode_emotion_enhance,
    use_ritardando,
    use_velocity_humanization,
    use_deep_sea_bass,
    use_art_silence,
    motif_cache,
    aba_controller,
    deep_sea_bass_controller,
    virtual_sustain_pedal,
    velocity_momentum_tracker,
):
    generated_pitches = list(seed_pitches)
    generated_durations = []
    generated_velocities = []

    current_time = 0.0
    section_a_pitches = []
    section_aprime_pitches = []

    high_low_deque = deque(maxlen=4)
    recent_step_pitches = deque(generated_pitches[-BAR_LENGTH_STEPS:], maxlen=BAR_LENGTH_STEPS)
    last_generated_pitches = deque(maxlen=2)

    with torch.no_grad():
        for step in range(num_notes):
            (
                section,
                transition_state,
                fade_progress,
                rhythm_pattern,
                temperature,
                section_length,
                fade_length,
                step_in_aprime,
            ) = get_generation_step_context(step, num_notes, temperature_strategy, aba_controller)

            if use_density_control:
                temperature = apply_density_temperature(temperature, recent_step_pitches)

            duration = get_duration_from_template(step, rhythm_pattern)

            input_seq = torch.tensor([generated_pitches[-100:]])
            logits = model(input_seq)
            next_token_logits = logits[0, -1, :]

            if use_deep_sea_bass and deep_sea_bass_controller:
                if deep_sea_bass_controller.should_force_bass(step, bar_length=BAR_LENGTH_STEPS):
                    next_token_logits = boost_low_pitch_probability(next_token_logits, boost_factor=2.5)
                    bass_indicator = " [FORCE_BASS]"
                else:
                    bass_indicator = ""
            else:
                bass_indicator = ""

            if use_pitch_aware_temp and len(generated_pitches) > 0:
                predicted_pitch_class = next_token_logits.argmax().item() % 12
                temperature = get_pitch_aware_temperature(predicted_pitch_class, temperature)

            next_token_logits = next_token_logits / temperature

            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, 130))[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)

            if use_affective_layer and len(generated_pitches) > 0:
                probs = apply_harmonic_bias(probs, generated_pitches[-1])

            if use_top_p:
                next_pitch = top_p_sampling(probs, p=top_p)
            else:
                next_pitch = torch.multinomial(probs, 1).item()

            pad_indicator = ""
            if next_pitch == PAD_PITCH:
                if step > 64:
                    print(f"[Generation Complete] Stopped at PAD token")
                    break
                next_pitch = random.choice([40, 43, 47, 48])
                pad_indicator = f" [PAD_PREVENT:{next_pitch}]"

            if (
                use_theme_anchoring
                and aba_controller
                and section == "A'"
                and transition_state
                and not transition_state.startswith("fade")
            ):
                if aba_controller.should_use_anchor(step_in_aprime):
                    next_pitch = aba_controller.get_theme_anchor_pitch(step_in_aprime)
                    anchor_indicator = " [ANCHOR]"
                else:
                    anchor_indicator = ""
            else:
                anchor_indicator = ""

            if aba_controller:
                next_pitch = aba_controller.apply_pitch_transition(
                    next_pitch, step, section, transition_state, fade_progress
                )

                flow_pitch = aba_controller.maybe_apply_scalic_flow(
                    step,
                    generated_pitches[-1] if generated_pitches else None,
                    section,
                    transition_state,
                )
                if flow_pitch is not None and flow_pitch != PAD_PITCH:
                    next_pitch = flow_pitch

                if use_mode_emotion_enhance:
                    next_pitch = limit_interval_by_mode(
                        next_pitch,
                        generated_pitches[-1] if generated_pitches else None,
                        section,
                        aba_controller.major_scale,
                        aba_controller.minor_scale,
                    )

                if use_impressionist_texture and len(generated_pitches) > 0:
                    high_low_deque.append(next_pitch)
                    next_pitch, was_enforced = enforce_high_low_alternation(
                        list(high_low_deque),
                        next_pitch,
                        high_threshold=HIGH_PITCH_THRESHOLD,
                        low_threshold=LOW_PITCH_THRESHOLD,
                        max_consecutive_high=3,
                    )
                    if was_enforced:
                        high_low_indicator = " [LOW_ENFORCE]"
                    else:
                        high_low_indicator = ""
                else:
                    high_low_indicator = ""

                if section == "A" and transition_state == "pure":
                    aba_controller.record_section_a(next_pitch, duration)
                    aba_controller.record_motif(next_pitch)
                    section_a_pitches.append(next_pitch)

                if section == "A'" and transition_state == "pure":
                    section_aprime_pitches.append(next_pitch)

                fade_indicator = ""
                if transition_state != "pure":
                    fade_indicator = f" | 过渡: {transition_state[:7]:7s} ({fade_progress*100:5.1f}%)"

                log_str = (
                    f"[{section:2s}] Step {step:3d}: Pitch {next_pitch:3d} | Temp {temperature:.2f} | {rhythm_pattern:12s}"
                    f"{fade_indicator}{anchor_indicator}{high_low_indicator}"
                )
            else:
                next_pitch = apply_key_constraint(next_pitch, key=key)
                log_str = f"Step {step:3d}: Pitch {next_pitch:3d} | Temp {temperature:.2f}"

            log_str += pad_indicator

            should_silence, silence_type = should_add_selective_silence(
                step,
                next_pitch,
                bar_length=BAR_LENGTH_STEPS,
                silence_probability=0.30,
                preserve_bass=True,
            )

            if use_art_silence and should_silence:
                if silence_type == 'high_only':
                    rest_pitch, rest_duration, rest_velocity = create_selective_rest_note(
                        silence_type='high_only',
                        duration=0.25,
                        bass_pitch=next_pitch if next_pitch < 50 else BREATHING_PITCH,
                    )
                    silence_indicator = " [SILENCE:HIGH_ONLY]"
                else:
                    rest_pitch, rest_duration, rest_velocity = create_selective_rest_note(
                        silence_type='full',
                        duration=0.25,
                    )
                    silence_indicator = " [SILENCE:FULL]"

                generated_pitches.append(rest_pitch)
                generated_durations.append(rest_duration)
                generated_velocities.append(rest_velocity)

                log_str += silence_indicator
                print(log_str)
                last_generated_pitches.append(rest_pitch)
                recent_step_pitches.append(rest_pitch)
                current_time += rest_duration
                continue

            if use_affective_layer:
                velocity = calculate_dynamic_velocity(
                    next_pitch,
                    section,
                    transition_state,
                    generated_pitches,
                    step,
                    section_length,
                    use_metric_grid=use_metric_grid,
                    global_step=step,
                    total_steps=num_notes,
                    use_velocity_humanization=use_velocity_humanization,
                    velocity_momentum_tracker=velocity_momentum_tracker,
                )

                adjusted_duration = virtual_sustain_pedal.update_bass_sustain(
                    next_pitch, duration, current_time
                )

                if len(last_generated_pitches) > 0:
                    legato_duration = calculate_legato_adjusted_duration(
                        adjusted_duration, list(last_generated_pitches), next_pitch
                    )
                else:
                    legato_duration = adjusted_duration

                if use_impressionist_texture:
                    legato_duration = get_duration_with_impressionist_texture(next_pitch, legato_duration)
                else:
                    legato_duration = get_staccato_legato_duration(next_pitch, legato_duration)

                if use_ritardando:
                    legato_duration = apply_ritardando_duration(
                        legato_duration, step, num_notes, ritardando_length=8
                    )

                adjusted_duration = legato_duration

                if next_pitch != PAD_PITCH:
                    if next_pitch > HIGH_VOICE_PITCH_SPLIT:
                        adjusted_duration = random.uniform(HIGH_VOICE_DUR_MIN, HIGH_VOICE_DUR_MAX)
                    elif next_pitch < LOW_VOICE_PITCH_SPLIT:
                        adjusted_duration = random.uniform(LOW_VOICE_DUR_MIN, LOW_VOICE_DUR_MAX)
                        velocity = int(velocity * LOW_VOICE_VELOCITY_MULT)
                        velocity = max(0, min(127, velocity))

                breathing_space = should_add_breathing_space(
                    step, section_length, fade_length, section, transition_state
                )
                if breathing_space:
                    breathing_indicator = " [BREATH]"
                else:
                    breathing_indicator = ""

                log_str += f" | Vel {velocity:3d}{breathing_indicator}{bass_indicator}"
            else:
                velocity = 100
                adjusted_duration = duration
                breathing_space = False

            print(log_str)

            generated_pitches.append(next_pitch)
            generated_durations.append(adjusted_duration)
            generated_velocities.append(velocity)
            recent_step_pitches.append(next_pitch)
            current_time += adjusted_duration

            last_generated_pitches.append(next_pitch)

            if breathing_space:
                breathing_pause = add_breathing_pause(tempo=120)
                generated_pitches.append(BREATHING_PITCH)
                generated_durations.append(breathing_pause)
                generated_velocities.append(BREATHING_VELOCITY)
                current_time += breathing_pause

            if motif_cache:
                motif_cache.add(next_pitch)

    return (
        generated_pitches,
        generated_durations,
        generated_velocities,
        section_a_pitches,
        section_aprime_pitches,
    )

def generate_music(
    num_notes=96,
    key="C_major",
    use_aba_structure=True,
    temperature_strategy="arch",
    use_top_p=True,
    top_p=0.9,
    top_k=10,
    use_motif_repeat=False,
    use_affective_layer=True,
    use_theme_anchoring=True,
    use_pitch_aware_temp=True,
    use_density_control=True,
    use_metric_grid=True,              # 新增：4/4 拍节拍器
    use_impressionist_texture=True,    # 新增：印象派织体
    use_mode_emotion_enhance=True,     # 新增：大小调情感增强
    use_ritardando=True,               # 新增：渐缓结尾
    use_velocity_humanization=True,    # 新增：力度呼吸化
    use_deep_sea_bass=True,            # 新增：深海低音织体
    use_art_silence=True,              # 新增：艺术留白
):
    """
    超高级音乐生成，集成艺术性的音乐优化
    
    Previous Args:
        use_metric_grid: 4/4 拍节拍器（强弱律动）
        use_impressionist_texture: 印象派织体（高低音分工）
        use_mode_emotion_enhance: 大小调情感增强
        use_ritardando: 渐缓处理（最后 8 音符）
    
    New Args:
        use_velocity_humanization: 力度呼吸化（随机抖动+力度曲线）
        use_deep_sea_bass: 深海低音织体（每2小节强制长低音）
        use_art_silence: 艺术留白（每小节第4拍30%概率休止）
    """
    
    model = build_gen_model()
    model.load_state_dict(torch.load("best_gen_model.pth", map_location=device))
    model.eval()

    global _last_velocity_jitter
    _last_velocity_jitter = 0.0
    
    seed = torch.tensor([[60, 62, 64, 65]])
    seed_pitches = seed.squeeze().tolist()

    motif_cache = MotifCache(motif_len=4) if use_motif_repeat else None
    aba_controller = (
        ABASectionControllerV2(section_length=num_notes // 3, key=key, fade_length=4)
        if use_aba_structure
        else None
    )
    deep_sea_bass_controller = DeepSeaBassController() if use_deep_sea_bass else None

    virtual_sustain_pedal = VirtualSustainPedalController()
    velocity_momentum_tracker = VelocityMomentumTracker(window_size=8)
    
    print(f"\n[Music Generation (Artistic Enhanced)]...")
    print(f"   Main Key: {key}")
    if use_aba_structure:
        print(f"   Structure: ABA Three-Part Form (32 notes each)")
    if use_metric_grid:
        print(f"   4/4 Metric Grid: ON - Bar=16 steps, Strong Beat at Step 0 (+20 vel)")
    if use_impressionist_texture:
        print(f"   Impressionist Texture: ON - High/Low alternation, Duration variation")
    if use_mode_emotion_enhance:
        print(f"   Mode Emotion: ON - A: Wide intervals, B: ≤4 semitones")
    if use_ritardando:
        print(f"   Ritardando: ON - Last 8 notes gradually slow down")
    if use_velocity_humanization:
        print(f"   Velocity Humanization: ON - Random jitter + pitch curve")
    if use_deep_sea_bass:
        print(f"   Deep Sea Bass Texture: ON - Mandatory bass every 2 bars")
    if use_art_silence:
        print(f"   Art of Silence: ON - 30% rest probability at beat 4")
    print(f"\n[v2.1 Legato Soul Enhancements - Enabled]")
    print(f"   Virtual Sustain Pedal: ON - Bass extends to next bass note")
    print(f"   Velocity Momentum: ON - Smooth force curves across 8-step phrases")
    print(f"   Selective Silence: ON - High pitches rest, bass sustains")
    print(f"   Legato Overlap: ON - 8% note overlap for smooth transitions")
    print(f"\n")
    
    (
        generated_pitches,
        generated_durations,
        generated_velocities,
        section_a_pitches,
        section_aprime_pitches,
    ) = run_generation_loop(
        model=model,
        seed_pitches=seed_pitches,
        num_notes=num_notes,
        key=key,
        temperature_strategy=temperature_strategy,
        use_top_p=use_top_p,
        top_p=top_p,
        top_k=top_k,
        use_affective_layer=use_affective_layer,
        use_theme_anchoring=use_theme_anchoring,
        use_pitch_aware_temp=use_pitch_aware_temp,
        use_density_control=use_density_control,
        use_metric_grid=use_metric_grid,
        use_impressionist_texture=use_impressionist_texture,
        use_mode_emotion_enhance=use_mode_emotion_enhance,
        use_ritardando=use_ritardando,
        use_velocity_humanization=use_velocity_humanization,
        use_deep_sea_bass=use_deep_sea_bass,
        use_art_silence=use_art_silence,
        motif_cache=motif_cache,
        aba_controller=aba_controller,
        deep_sea_bass_controller=deep_sea_bass_controller,
        virtual_sustain_pedal=virtual_sustain_pedal,
        velocity_momentum_tracker=velocity_momentum_tracker,
    )

    if False:
        for step in range(num_notes):
            (
                section,
                transition_state,
                fade_progress,
                rhythm_pattern,
                temperature,
                section_length,
                fade_length,
                step_in_aprime,
            ) = get_generation_step_context(step, num_notes, temperature_strategy, aba_controller)
            
            if use_density_control:
                temperature = apply_density_temperature(temperature, recent_step_pitches)

            
            duration = get_duration_from_template(step, rhythm_pattern)
            
            # ===== 前向传播 =====
            input_seq = torch.tensor([generated_pitches[-100:]])
            logits = model(input_seq)
            next_token_logits = logits[0, -1, :]
            
            # ===== 新增：深海低音织体（Deep Sea Bass）=====
            if use_deep_sea_bass and deep_sea_bass_controller:
                if deep_sea_bass_controller.should_force_bass(step, bar_length=BAR_LENGTH_STEPS):
                    # 强制生成低音
                    next_token_logits = boost_low_pitch_probability(next_token_logits, boost_factor=2.5)
                    bass_indicator = " [FORCE_BASS]"
                else:
                    bass_indicator = ""
            else:
                bass_indicator = ""
            
            # ===== 应用音高敏感的温度 =====
            if use_pitch_aware_temp and len(generated_pitches) > 0:
                predicted_pitch_class = next_token_logits.argmax().item() % 12
                temperature = get_pitch_aware_temperature(predicted_pitch_class, temperature)
            
            # ===== 应用温度 =====
            next_token_logits = next_token_logits / temperature
            
            # ===== Top-K 过滤 =====
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, 130))[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # ===== 采样 =====
            probs = F.softmax(next_token_logits, dim=-1)
            
            # ===== 新增：和谐音程偏好 =====
            if use_affective_layer and len(generated_pitches) > 0:
                probs = apply_harmonic_bias(probs, generated_pitches[-1])
            
            if use_top_p:
                next_pitch = top_p_sampling(probs, p=top_p)
            else:
                next_pitch = torch.multinomial(probs, 1).item()

            pad_indicator = ""
            if next_pitch == PAD_PITCH:
                if step > 64:
                    print(f"[Generation Complete] Stopped at PAD token")
                    break
                next_pitch = random.choice([40, 43, 47, 48])
                pad_indicator = f" [PAD_PREVENT:{next_pitch}]"
            
            # ===== 新增：主题锚定逻辑（A'段特性）=====
            if use_theme_anchoring and aba_controller and section == "A'" and not transition_state.startswith("fade"):
                if aba_controller.should_use_anchor(step_in_aprime):
                    # 在锚点位置，强制使用A段的原始pitch
                    anchor_pitch = aba_controller.get_theme_anchor_pitch(step_in_aprime)
                    next_pitch = anchor_pitch
                    anchor_indicator = " [ANCHOR]"
                else:
                    # 在非锚点位置让模型自由变奏
                    anchor_indicator = ""
            else:
                anchor_indicator = ""
            
            # ===== 应用ABA三段式逻辑（包含灰度过渡） =====
            if aba_controller:
                next_pitch = aba_controller.apply_pitch_transition(
                    next_pitch, step, section, transition_state, fade_progress
                )
                
                # ===== 新增：大小调情感增强 - 限制 B 段音程跳跃 =====
                if use_mode_emotion_enhance:
                    next_pitch = limit_interval_by_mode(
                        next_pitch, generated_pitches[-1] if generated_pitches else None,
                        section, aba_controller.major_scale, aba_controller.minor_scale
                    )
                
                # ===== 新增：印象派织体 - 强制高低音分工 =====
                if use_impressionist_texture and len(generated_pitches) > 0:
                    high_low_deque.append(next_pitch)
                    next_pitch, was_enforced = enforce_high_low_alternation(
                        list(high_low_deque), next_pitch, 
                        high_threshold=HIGH_PITCH_THRESHOLD, low_threshold=LOW_PITCH_THRESHOLD, max_consecutive_high=3
                    )
                    if was_enforced:
                        high_low_indicator = " [LOW_ENFORCE]"
                    else:
                        high_low_indicator = ""
                else:
                    high_low_indicator = ""
                
                # ===== 记录A段的pitch和duration，用于A'段的锚定和节奏共享 =====
                if section == "A" and transition_state == "pure":
                    aba_controller.record_section_a(next_pitch, duration)
                    aba_controller.record_motif(next_pitch)
                    section_a_pitches.append(next_pitch)
                
                # 在A'段记录pitch用于最终的相似度计算
                if section == "A'" and transition_state == "pure":
                    section_aprime_pitches.append(next_pitch)
                
                # 格式化输出
                fade_indicator = ""
                if transition_state != "pure":
                    fade_indicator = f" | 过渡: {transition_state[:7]:7s} ({fade_progress*100:5.1f}%)"
                
                log_str = f"[{section:2s}] Step {step:3d}: Pitch {next_pitch:3d} | Temp {temperature:.2f} | {rhythm_pattern:12s}{fade_indicator}{anchor_indicator}{high_low_indicator}"
            else:
                next_pitch = apply_key_constraint(next_pitch, key=key)
                log_str = f"Step {step:3d}: Pitch {next_pitch:3d} | Temp {temperature:.2f}"
            log_str += pad_indicator
            
            # ===== v2.1 选择性艺术留白（Selective Silence）=====
            should_silence, silence_type = should_add_selective_silence(
                step, next_pitch, bar_length=BAR_LENGTH_STEPS, silence_probability=0.30, preserve_bass=True
            )
            
            if use_art_silence and should_silence:
                # 选择性添加休止符（高音休止，保留低音）
                if silence_type == 'high_only':
                    # 保留低音，高音休止
                    rest_pitch, rest_duration, rest_velocity = create_selective_rest_note(
                        silence_type='high_only', duration=0.25, bass_pitch=next_pitch if next_pitch < 50 else BREATHING_PITCH
                    )
                    silence_indicator = " [SILENCE:HIGH_ONLY]"
                else:
                    # 完全休止
                    rest_pitch, rest_duration, rest_velocity = create_selective_rest_note(
                        silence_type='full', duration=0.25
                    )
                    silence_indicator = " [SILENCE:FULL]"
                
                generated_pitches.append(rest_pitch)
                generated_durations.append(rest_duration)
                generated_velocities.append(rest_velocity)
                
                log_str += silence_indicator
                print(log_str)
                last_generated_pitches.append(rest_pitch)
                recent_step_pitches.append(rest_pitch)
                current_time += rest_duration
                continue
            
            # ===== 新增：情感表达层（含 4/4 拍节拍器、Ritardando 和力度呼吸化）=====
            if use_affective_layer:
                # 计算动态力度（增加 Metric Grid、Ritardando、Velocity Humanization 和 Momentum 参数）
                velocity = calculate_dynamic_velocity(
                    next_pitch, section, transition_state, 
                    generated_pitches, step, section_length,
                    use_metric_grid=use_metric_grid,
                    global_step=step,
                    total_steps=num_notes,
                    use_velocity_humanization=use_velocity_humanization,
                    velocity_momentum_tracker=velocity_momentum_tracker  # v2.1 添加
                )
                
                # ===== v2.1 虚拟延音踏板（Virtual Sustain Pedal）=====
                # 对低音应用虚拟踏板（延长到下一个低音为止）
                adjusted_duration = virtual_sustain_pedal.update_bass_sustain(next_pitch, duration, current_time)
                
                # ===== v2.1 连奏重叠（Legato Note Overlap）=====
                # 添加8%的音符重叠，使音符流畅连接
                if len(last_generated_pitches) > 0:
                    legato_duration = calculate_legato_adjusted_duration(
                        adjusted_duration, list(last_generated_pitches), next_pitch
                    )
                else:
                    legato_duration = adjusted_duration
                
                # ===== 新增：印象派织体 - 高低音时值调整 =====
                if use_impressionist_texture:
                    legato_duration = get_duration_with_impressionist_texture(next_pitch, legato_duration)
                else:
                    legato_duration = get_staccato_legato_duration(next_pitch, legato_duration)
                
                # ===== 新增：Ritardando - 时值调整 =====
                if use_ritardando:
                    legato_duration = apply_ritardando_duration(legato_duration, step, num_notes, ritardando_length=8)
                
                adjusted_duration = legato_duration
                
                # 检查呼吸空间
                breathing_space = should_add_breathing_space(step, section_length, fade_length, section, transition_state)
                if breathing_space:
                    breathing_indicator = " [BREATH]"
                else:
                    breathing_indicator = ""
                
                log_str += f" | Vel {velocity:3d}{breathing_indicator}{bass_indicator}"
            else:
                velocity = 100
                adjusted_duration = duration
                breathing_space = False
            
            print(log_str)
            
            generated_pitches.append(next_pitch)
            generated_durations.append(adjusted_duration)
            generated_velocities.append(velocity)
            recent_step_pitches.append(next_pitch)
            current_time += adjusted_duration
            
            # v2.1 更新pitch历史以用于legato计算
            last_generated_pitches.append(next_pitch)
            
            # 在呼吸空间处加入休止
            if breathing_space:
                breathing_pause = add_breathing_pause(tempo=120)
                generated_pitches.append(BREATHING_PITCH)
                generated_durations.append(breathing_pause)
                generated_velocities.append(BREATHING_VELOCITY)
                current_time += breathing_pause
            
            # 缓存到动机缓存
            if motif_cache:
                motif_cache.add(next_pitch)
            
            
    
    # ===== 计算并输出相似度指标 =====
    if use_theme_anchoring and aba_controller and len(section_a_pitches) > 0 and len(section_aprime_pitches) > 0:
        print("\n" + "="*80)
        print("[THEME ANALYSIS REPORT]")
        print("="*80)
        
        similarity_metrics = compute_intervalic_similarity(section_a_pitches, section_aprime_pitches)
        
        print(f"\nSection A Analysis:")
        print(f"  Pitch Range: {min(section_a_pitches)} - {max(section_a_pitches)} (Range: {max(section_a_pitches) - min(section_a_pitches)})")
        print(f"  Mean Pitch: {similarity_metrics['mean_pitch_a']:.1f}")
        print(f"  Total Notes: {len(section_a_pitches)}")
        
        print(f"\nSection A' Analysis:")
        print(f"  Pitch Range: {min(section_aprime_pitches)} - {max(section_aprime_pitches)} (Range: {max(section_aprime_pitches) - min(section_aprime_pitches)})")
        print(f"  Mean Pitch: {similarity_metrics['mean_pitch_aprime']:.1f}")
        print(f"  Total Notes: {len(section_aprime_pitches)}")
        
        print(f"\nThematic Coherence Metrics:")
        print(f"  Interval Match Ratio: {similarity_metrics['interval_match_ratio']:.2%}")
        print(f"  Pitch Range Difference: {similarity_metrics['pitch_range_diff']} semitones")
        print(f"  Contour Similarity (Direction Consistency): {similarity_metrics['contour_similarity']:.2%}")
        
        # 整体评分
        overall_score = (
            similarity_metrics['interval_match_ratio'] * 0.4 +
            (1.0 - min(similarity_metrics['pitch_range_diff'] / 20.0, 1.0)) * 0.3 +
            similarity_metrics['contour_similarity'] * 0.3
        )
        print(f"\nOverall Thematic Coherence Score: {overall_score:.2%}")
        
        if overall_score >= 0.75:
            coherence_level = "[EXCELLENT] Strong thematic recapitulation"
        elif overall_score >= 0.60:
            coherence_level = "[GOOD] Clear A' as variation of A"
        elif overall_score >= 0.45:
            coherence_level = "[FAIR] Some thematic relationship detected"
        else:
            coherence_level = "[WEAK] Minimal thematic connection"
        
        print(f"Coherence Level: {coherence_level}")
        print("="*80 + "\n")
    elif use_theme_anchoring and len(section_a_pitches) > 0:
        print("\n" + "="*80)
        print("[THEME ANALYSIS REPORT]")
        print("="*80)
        print(f"\nGeneration completed early (stopped at PAD token)")
        print(f"Section A was fully generated ({len(section_a_pitches)} notes)")
        print(f"Section A' was not fully generated ({len(section_aprime_pitches)} notes)")
        print("="*80 + "\n")
    
    return generated_pitches, generated_durations, generated_velocities

# =============================================================================
# MIDI 转换与保存
# =============================================================================

def save_midi(pitches, durations, velocities=None, output_path="generated_music_advanced.mid"):
    """
    保存为MIDI文件，支持动态力度
    
    Args:
        pitches: 生成的pitch列表
        durations: 对应的duration列表
        velocities: 对应的velocity列表（如果为None则使用默认100）
        output_path: 输出文件路径
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Piano
    
    # 如果没有提供velocity，使用默认值
    if velocities is None:
        velocities = [100] * len(pitches)
    
    current_time = 0.0

    overlap_sec = max(
        LEGATO_OVERLAP_MIN_SEC,
        min(LEGATO_OVERLAP_MAX_SEC, LEGATO_OVERLAP_RATIO * LEGATO_OVERLAP_BASE_SEC),
    )
    
    for i, (pitch, duration, velocity) in enumerate(zip(pitches, durations, velocities)):
        original_pitch = int(pitch)
        
        # 确保velocity在有效范围
        velocity = int(velocity)
        velocity = max(0, min(127, velocity))

        if original_pitch == PAD_PITCH:
            current_time += duration
            continue

        pitch = max(21, min(108, original_pitch))
        
        # 跳过休止符（velocity<30），但依然推进时间
        if velocity < 30:
            current_time += duration
            continue
        
        note_duration = duration + overlap_sec
        if pitch < LOW_PITCH_THRESHOLD:
            note_duration *= LOW_PITCH_SUSTAIN_MULT

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + note_duration,
        )
        inst.notes.append(note)
        current_time += duration
    
    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"\n[MIDI Saved] {output_path}")
    print(f"  Total Duration: {current_time:.1f}s")
    print(f"  Notes (velocity >= 30): {sum(1 for v in velocities if v >= 30)}\n")
    
    return pm

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # ========================================================================
    # 完整配置：所有艺术性优化启用（推荐）
    # ========================================================================
    CONFIG_ARTISTIC = {
        "num_notes": 96,
        "key": "C_major",
        "use_aba_structure": True,
        "temperature_strategy": "arch",
        "use_top_p": True,
        "top_p": 0.90,
        "top_k": 10,
        "use_motif_repeat": False,
        "use_affective_layer": True,
        "use_theme_anchoring": True,
        "use_pitch_aware_temp": True,
        "use_density_control": True,
        "use_metric_grid": True,              # ✨ 4/4 拍节拍器
        "use_impressionist_texture": True,    # ✨ 印象派织体
        "use_mode_emotion_enhance": True,     # ✨ 大小调情感增强
        "use_ritardando": True,               # ✨ 渐缓结尾
        "use_velocity_humanization": True,    # ✨ 力度呼吸化
        "use_deep_sea_bass": True,            # ✨ 深海低音织体
        "use_art_silence": True,              # ✨ 艺术留白
    }
    
    # 生成音乐
    pitches, durations, velocities = generate_music(**CONFIG_ARTISTIC)
    
    # 保存MIDI
    save_midi(pitches, durations, velocities=velocities, output_path="generated_music_artistic_v2.mid")
    
    # ========================================================================
    # 方案2：可选 - ABA三段式，不带高级功能
    # ========================================================================
    # CONFIG_ABA_SIMPLE = {
    #     "num_notes": 96,
    #     "key": "C_major",
    #     "use_aba_structure": True,
    #     "use_affective_layer": True,
    #     "use_theme_anchoring": False,
    #     "use_pitch_aware_temp": False,
    #     "use_density_control": False,
    # }
    # pitches, durations, velocities = generate_music(**CONFIG_ABA_SIMPLE)
    # save_midi(pitches, durations, velocities=velocities, output_path="generated_music_aba_simple.mid")
