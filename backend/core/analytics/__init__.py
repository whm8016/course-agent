"""学情分析读模型层（学情分析四模块设计 §第二期+）。

事件层（L0）：learning_events 表 + record_learning_event 写入。
读模型层（L1）：course_daily_rollup / student_course_rollup（ARQ cron 增量聚合）、
course_faq（语义聚类）随各期落地。
"""
