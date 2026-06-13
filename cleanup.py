#!/usr/bin/env python3
"""
cleanup.py — ทำความสะอาด Project Structure
ลบไฟล์ซ้ำซ้อนใน PREPARE/raw_data/ ที่ถูก sync ไป DATASET/ แล้ว

วิธีใช้:
  python3 cleanup.py          # ดูว่าจะลบอะไรบ้าง (Dry Run)
  python3 cleanup.py --delete # ลบจริง
"""

import os
import sys
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(BASE, "PREPARE", "raw_data")
DATASET_VIDEOS = os.path.join(BASE, "DATASET", "videos")

DRY_RUN = "--delete" not in sys.argv

if DRY_RUN:
    print("=" * 55)
    print("  DRY RUN — แสดงผลเท่านั้น ยังไม่ลบจริง")
    print("  ใช้ python3 cleanup.py --delete เพื่อลบจริง")
    print("=" * 55)

# ─────────────────────────────────────────────────────────
# 1. ลบ .mp4 ซ้ำใน PREPARE/raw_data/ ที่มีใน DATASET/videos/ แล้ว
# ─────────────────────────────────────────────────────────
print("\n📁 ตรวจสอบไฟล์ซ้ำใน PREPARE/raw_data/")

duplicates = []
for fname in os.listdir(RAW_DATA):
    if not fname.endswith(".mp4"):
        continue
    src = os.path.join(RAW_DATA, fname)
    dst = os.path.join(DATASET_VIDEOS, fname)
    if os.path.exists(dst):
        size_mb = os.path.getsize(src) / 1_000_000
        duplicates.append((src, size_mb))
        print(f"  🗑  {fname}  ({size_mb:.2f} MB)")

total_mb = sum(s for _, s in duplicates)
print(f"\n  รวมพื้นที่ที่จะได้คืน: {total_mb:.2f} MB ({len(duplicates)} ไฟล์)")

if not DRY_RUN:
    for src, _ in duplicates:
        os.remove(src)
    print("  ✅ ลบไฟล์ซ้ำเรียบร้อยแล้ว")

# ─────────────────────────────────────────────────────────
# 2. ลบ labels.json ซ้ำใน raw_data (ขนาดเท่ากับ DATASET/)
# ─────────────────────────────────────────────────────────
print("\n📄 ตรวจสอบ labels ซ้ำ")

raw_json = os.path.join(RAW_DATA, "labels.json")
ds_json  = os.path.join(BASE, "DATASET", "labels.json")

if os.path.exists(raw_json) and os.path.exists(ds_json):
    if os.path.getsize(raw_json) == os.path.getsize(ds_json):
        print(f"  🗑  PREPARE/raw_data/labels.json  (ซ้ำกับ DATASET/labels.json)")
        if not DRY_RUN:
            os.remove(raw_json)
            print("  ✅ ลบแล้ว")
    else:
        print(f"  ⚠️  labels.json ขนาดต่างกัน — ข้ามการลบ กรุณาตรวจสอบด้วยตัวเอง")

# ─────────────────────────────────────────────────────────
# 3. แจ้งเตือน labels.csv ที่ขนาดต่างกัน
# ─────────────────────────────────────────────────────────
raw_csv = os.path.join(RAW_DATA, "labels.csv")
ds_csv  = os.path.join(BASE, "DATASET", "labels.csv")

if os.path.exists(raw_csv) and os.path.exists(ds_csv):
    raw_sz = os.path.getsize(raw_csv)
    ds_sz  = os.path.getsize(ds_csv)
    print(f"\n⚠️  labels.csv ขนาดต่างกัน:")
    print(f"   PREPARE/raw_data/labels.csv : {raw_sz:,} bytes")
    print(f"   DATASET/labels.csv          : {ds_sz:,} bytes  ← (ใหม่กว่า ใช้อันนี้)")
    print(f"   → DATASET/labels.csv มีข้อมูลมากกว่า (รวม screen recordings แล้ว)")
    if not DRY_RUN:
        os.remove(raw_csv)
        print("   ✅ ลบ PREPARE/raw_data/labels.csv แล้ว (ใช้ DATASET/labels.csv เป็นหลัก)")

# ─────────────────────────────────────────────────────────
# 4. ลบ __pycache__ ที่ root
# ─────────────────────────────────────────────────────────
pycache = os.path.join(BASE, "__pycache__")
if os.path.exists(pycache):
    print(f"\n🐍 พบ __pycache__/ ที่ root")
    if not DRY_RUN:
        shutil.rmtree(pycache)
        print("  ✅ ลบแล้ว")
    else:
        print(f"  🗑  {pycache}")

# ─────────────────────────────────────────────────────────
# 5. แจ้งเตือนไฟล์ที่รอประมวลผลใน DATASET/test/
# ─────────────────────────────────────────────────────────
test_dir = os.path.join(BASE, "DATASET", "test")
test_files = os.listdir(test_dir) if os.path.exists(test_dir) else []
if test_files:
    print(f"\n📋 DATASET/test/ มีไฟล์รอประมวลผล ({len(test_files)} ไฟล์):")
    for f in test_files:
        print(f"  ⏳ {f}")
    print("  → ถ้าต้องการเพิ่มใน Dataset ให้รัน: python3 PREPARE/process_youtube.py <path>")
    print("     จากนั้นรัน: python3 PREPARE/sync_pipeline.py")

# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if DRY_RUN:
    print("  ✅ Dry Run เสร็จสิ้น — ใช้ --delete เพื่อลบจริง")
else:
    print("  ✅ Cleanup เสร็จสิ้น!")
print("=" * 55)
