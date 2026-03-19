# utils có nghĩa là tiện ích, chứa các hàm tiện ích được sử dụng trong toàn bộ ứng dụng. 
# Đây là nơi bạn có thể đặt các hàm chung mà nhiều phần của ứng dụng có thể sử dụng,
# giúp giảm thiểu sự trùng lặp mã và cải thiện khả năng bảo trì của ứng dụng.

# app/utils/__init__.py
"""
Utility functions and helpers.

WHY THIS SUBPACKAGE?
- Shared code used by multiple modules
- Logger, config, validators, etc.
- Keeps main modules clean

WHAT GOES HERE?
✅ Logger setup
✅ Config loading
✅ Data validation
✅ Performance metrics
❌ Business logic (goes to core/)
❌ UI code (goes to ui/)
"""

print("[Utils] Loaded")