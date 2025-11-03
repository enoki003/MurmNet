#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libzim.reader import Archive
from src.config.settings import config

zim_path = config.knowledge_base.zim_file_path
print(f"Opening: {zim_path}")

archive = Archive(zim_path)
print(f"Entry count: {archive.entry_count}")
print(f"All entry count: {archive.all_entry_count}")

# 最初の10個の記事を表示
count = 0
for idx in range(min(20, archive.all_entry_count)):
    try:
        entry = archive._get_entry_by_id(idx)
        if not entry.is_redirect:
            print(f"{count}. Path: {entry.path}, Title: {entry.title}")
            count += 1
            if count >= 10:
                break
    except:
        pass
