"""
ABA三段式结构演示脚本
用于验证生成的音乐是否遵循预期的结构和节奏变化
"""
import torch
from generate import generate_music, save_midi

print("=" * 80)
print("ABA三段式音乐生成测试")
print("=" * 80)

# 测试不同的ABA组合
test_configs = [
    {
        "name": "经典C大调ABA",
        "config": {
            "num_notes": 96,
            "key": "C_major",
            "use_aba_structure": True,
            "temperature_strategy": "arch",
            "use_top_p": True,
            "top_p": 0.90,
            "top_k": 10,
        },
        "output": "aba_c_major.mid"
    },
    {
        "name": "A小调ABA（忧伤风格）",
        "config": {
            "num_notes": 96,
            "key": "A_minor",
            "use_aba_structure": True,
            "temperature_strategy": "arch",
            "use_top_p": True,
            "top_p": 0.90,
            "top_k": 10,
        },
        "output": "aba_a_minor.mid"
    },
    {
        "name": "G大调ABA（明亮风格）",
        "config": {
            "num_notes": 96,
            "key": "G_major",
            "use_aba_structure": True,
            "temperature_strategy": "arch",
            "use_top_p": True,
            "top_p": 0.90,
            "top_k": 10,
        },
        "output": "aba_g_major.mid"
    },
]

for test in test_configs:
    print(f"\n{'='*80}")
    print(f"生成: {test['name']}")
    print(f"{'='*80}\n")
    
    try:
        pitches, durations = generate_music(**test["config"])
        save_midi(pitches, durations, output_path=test["output"])
        
        # 分析结构
        section_len = 32
        a_section_pitches = pitches[4:4+section_len]  # 跳过初始seed
        b_section_pitches = pitches[4+section_len:4+section_len*2]
        a_prime_section_pitches = pitches[4+section_len*2:4+section_len*3]
        
        print(f"\n📊 结构分析:")
        print(f"  A段  音诺范围: {min(a_section_pitches)}-{max(a_section_pitches)}")
        print(f"  B段  音诺范围: {min(b_section_pitches)}-{max(b_section_pitches)}")
        print(f"  A'段 音诺范围: {min(a_prime_section_pitches)}-{max(a_prime_section_pitches)}")
        
        # 计算平均音高
        avg_a = sum(a_section_pitches) / len(a_section_pitches)
        avg_b = sum(b_section_pitches) / len(b_section_pitches)
        avg_a_prime = sum(a_prime_section_pitches) / len(a_prime_section_pitches)
        
        print(f"\n  A段  平均音高: {avg_a:.1f}")
        print(f"  B段  平均音高: {avg_b:.1f}")
        print(f"  A'段 平均音高: {avg_a_prime:.1f}")
        
        # 计算音高变化（方差作为多样性指标）
        var_a = sum((x - avg_a) ** 2 for x in a_section_pitches) / len(a_section_pitches)
        var_b = sum((x - avg_b) ** 2 for x in b_section_pitches) / len(b_section_pitches)
        var_a_prime = sum((x - avg_a_prime) ** 2 for x in a_prime_section_pitches) / len(a_prime_section_pitches)
        
        print(f"\n  A段  音高多样性: {var_a:.1f}")
        print(f"  B段  音高多样性: {var_b:.1f}")
        print(f"  A'段 音高多样性: {var_a_prime:.1f}")
        
        print(f"\n✓ 生成成功!")
        
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("ABA生成测试完成！")
print("=" * 80)
