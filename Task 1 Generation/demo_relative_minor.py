"""
ABA三段式 + 关系小调灰度过渡 演示脚本
展示从大调（明亮）→ 关系小调（忧郁）→ 大调（明亮）的情感转换
"""
import torch
from generate import generate_music, save_midi

print("=" * 80)
print("🎼 ABA三段式 + 关系小调灰度过渡 - 音乐生成演示")
print("=" * 80)

test_configs = [
    {
        "name": "C大调 → A小调 (经典大小调关系)",
        "key": "C_major",
        "output": "aba_c_major_to_a_minor.mid",
        "description": "从C大调的灿烂明亮过渡到A小调的深沉忧郁，再回到C大调"
    },
    {
        "name": "G大调 → E小调 (明快→暗沉)",
        "key": "G_major",
        "output": "aba_g_major_to_e_minor.mid",
        "description": "G大调的明快感过渡到E小调的introspective"
    },
    {
        "name": "D大调 → B小调 (光明→忧思)",
        "key": "D_major",
        "output": "aba_d_major_to_b_minor.mid",
        "description": "D大调的喜悦感转变为B小调的思考"
    },
]

for i, test in enumerate(test_configs, 1):
    print(f"\n{'='*80}")
    print(f"测试 {i}/3: {test['name']}")
    print(f"{'='*80}")
    print(f"说明: {test['description']}\n")
    
    try:
        CONFIG = {
            "num_notes": 96,
            "key": test["key"],
            "use_aba_structure": True,
            "temperature_strategy": "arch",
            "use_top_p": True,
            "top_p": 0.90,
            "top_k": 10,
            "use_motif_repeat": False,
        }
        
        pitches, durations = generate_music(**CONFIG)
        save_midi(pitches, durations, output_path=test["output"])
        
        # 分析结构
        section_len = 32
        # 跳过初始seed (前4个)
        a_pitches = pitches[4:4+section_len]
        b_pitches = pitches[4+section_len:4+section_len*2]
        a_prime_pitches = pitches[4+section_len*2:4+section_len*3]
        
        print(f"\n📊 段落分析:")
        print(f"  A段 (大调/明亮)      - 音高范围: {min(a_pitches):3d}-{max(a_pitches):3d} | 平均: {sum(a_pitches)/len(a_pitches):.1f} | 多样性: {sum((x - sum(a_pitches)/len(a_pitches))**2 for x in a_pitches)/len(a_pitches):.1f}")
        print(f"  B段 (小调/忧郁)      - 音高范围: {min(b_pitches):3d}-{max(b_pitches):3d} | 平均: {sum(b_pitches)/len(b_pitches):.1f} | 多样性: {sum((x - sum(b_pitches)/len(b_pitches))**2 for x in b_pitches)/len(b_pitches):.1f}")
        print(f"  A'段 (大调/回归)     - 音高范围: {min(a_prime_pitches):3d}-{max(a_prime_pitches):3d} | 平均: {sum(a_prime_pitches)/len(a_prime_pitches):.1f} | 多样性: {sum((x - sum(a_prime_pitches)/len(a_prime_pitches))**2 for x in a_prime_pitches)/len(a_prime_pitches):.1f}")
        
        # 统计过渡区间的音符特征
        fade_len = 4
        a_to_b_fade = pitches[4 + section_len - fade_len:4 + section_len + fade_len]
        b_to_a_fade = pitches[4 + section_len * 2 - fade_len:4 + section_len * 2 + fade_len]
        
        print(f"\n🎨 灰度过渡分析:")
        print(f"  A→B过渡区间 (步 {section_len-fade_len}-{section_len+fade_len-1}): {a_to_b_fade}")
        print(f"  B→A过渡区间 (步 {section_len*2-fade_len}-{section_len*2+fade_len-1}): {b_to_a_fade}")
        
        print(f"\n✓ 生成成功! 输出: {test['output']}")
        
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("演示完成!")
print("=" * 80)
print("""
关系小调灰度过渡的音乐学原理：

1. 大调 vs 小调：
   - C大调: C D E F G A B (0 2 4 5 7 9 11mod 12)
   - A小调: A B C D E F G (相同音符，不同心理中心)
   
2. 灰度过渡 (Fade Transition)：
   - 前4步 (A→B)：逐渐从大调过渡到小调，增加情感的"雾蒙蒙"感
   - B段纯粹：完全进入小调，深沉、内省、忧郁
   - 后4步 (B→A')：逐渐从小调回到大调，温暖的"解冻"感
   
3. 温度调节：
   - 过渡区间温度提升，增加采样的随机性和音乐的"模糊感"
   - A/B/A'纯粹段落温度相对稳定
   
4. 节奏对比：
   - A/A'段：dotted（附点节奏，舒缓、稳定）
   - B段：swing（摇摆节奏、密集快速）
   
这种设计让音乐在情感上有明确的"起承转合"！
""")
