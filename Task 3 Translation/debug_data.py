"""快速诊断数据和词表大小的脚本"""

from pathlib import Path
from data import read_tsv_pairs, build_datasets
from text_processor import tokenize_en, tokenize_zh
from config import MAX_SEQ_LEN, MIN_FREQ, TRAIN_SPLIT

# 检查数据文件
data_path = Path("../data/tatoeba_en_zh.tsv")
if not data_path.exists():
    print(f"❌ 数据文件不存在: {data_path.resolve()}")
    print("请运行: python prepare_tatoeba.py 来下载数据")
    exit(1)

print("✅ 数据文件存在")

# 读取所有数据
print("\n📊 读取数据中...")
pairs = read_tsv_pairs(data_path, max_rows=None)
print(f"✅ 加载了 {len(pairs)} 对句子")

# 查看样本
print("\n📝 样本数据（前3对）：")
for i, (en, zh) in enumerate(pairs[:3]):
    print(f"  {i+1}. EN: {en}")
    print(f"     ZH: {zh}")
    en_tokens = tokenize_en(en)
    zh_tokens = tokenize_zh(zh)
    print(f"     分词: EN={en_tokens} ZH={zh_tokens}")

# 构建词表
print("\n🔤 构建词表中...")
bundle = build_datasets(
    pairs,
    max_seq_len=MAX_SEQ_LEN,
    min_freq=MIN_FREQ,
    train_split=TRAIN_SPLIT,
)

print(f"\n✅ 词表统计：")
print(f"   源词表大小（英文）: {len(bundle.processor.src_vocab)}")
print(f"   目标词表大小（中文）: {len(bundle.processor.tgt_vocab)}")
print(f"   训练集大小: {len(bundle.train_ds)}")
print(f"   验证集大小: {len(bundle.val_ds)}")

# 测试编码/解码
print(f"\n🧪 编码/解码测试：")
test_en = "I love you"
test_zh = "我爱你"

en_ids = bundle.processor.encode_src(test_en)
zh_ids = bundle.processor.encode_tgt(test_zh)
print(f"   '{test_en}' -> {en_ids}")
print(f"   '{test_zh}' -> {zh_ids}")

decoded_zh = bundle.processor.decode_tgt(zh_ids)
print(f"   解码结果: {decoded_zh}")

# 展示词表中的词
print(f"\n📚 英文词表样本（前20个）：")
print(f"   {bundle.processor.src_vocab.id_to_token[:20]}")

print(f"\n📚 中文词表样本（前20个）：")
print(f"   {bundle.processor.tgt_vocab.id_to_token[:20]}")

print("\n✅ 诊断完成！")
print("\n💡 分析：")
if len(bundle.processor.src_vocab) < 100:
    print("   ⚠️  英文词表太小(<100)，可能需要更多数据或调整min_freq")
if len(bundle.processor.tgt_vocab) < 100:
    print("   ⚠️  中文词表太小(<100)，可能需要更多数据或调整min_freq")
if len(bundle.train_ds) < 1000:
    print("   ⚠️  训练集太小(<1000)，可能导致欠拟合")

print("\n📌 后续步骤：")
print("   1. 运行 python train_mt.py 来训练模型（会使用全部数据）")
print("   2. 注意输出的词表大小和数据集大小")
print("   3. 查看第一个测试翻译的调试信息（Debug mode）")
