#!/usr/bin/env python
"""Test script for ZIM file API"""

import libzim.reader

a = libzim.reader.Archive('./ZIM/wikipedia_en_top_nopic_2025-09.zim')
print(f"Total entries: {a.entry_count}")
print(f"Article count: {a.article_count}")
print(f"Has fulltext index: {a.has_fulltext_index}")
print(f"Has title index: {a.has_title_index}")

# Test getting an entry
for i in range(min(10, a.entry_count)):
    try:
        e = a._get_entry_by_id(i)
        print(f"Entry {i}: title='{e.title}', path='{e.path}', is_redirect={e.is_redirect}")
        if not e.is_redirect:
            item = e.get_item()
            print(f"  -> Item mimetype: {item.mimetype}, size: {item.size}")
            print(f"  -> Item methods: {[m for m in dir(item) if not m.startswith('_')]}")
            content = bytes(item.content).decode('utf-8')[:500]
            print(f"  -> Content preview: {content}")
            break
    except Exception as ex:
        print(f"Entry {i}: Error - {ex}")
