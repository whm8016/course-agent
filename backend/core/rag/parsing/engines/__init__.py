"""解析引擎实现（core 不在此 import 重依赖，由 registry 惰性加载）。

- ``mineru_api``：默认，托管 API（云端不装 torch）
- ``docling``：可选自托管（装 parse-docling extra，第二期 deps-split 平移）
"""
