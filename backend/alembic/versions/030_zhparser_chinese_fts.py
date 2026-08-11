r"""zhparser 中文分词：重建 data_kb_chunks.text_search_tsv 用 chinese 配置。

根因：PGVectorStore 配 ``text_search_config='simple'``，simple 配置对中文不分词（只按
空白/标点切），整段中文被压成一个 token；查询「怎么理解协程」整串也是一个 token，
必须精确等于文档某 token 才命中 -> sparse 路每次 0 召回。llama-index 库的 SPARSE 查询
（base.py:951）虽把查询标点转空格再 ``|`` 拼 OR，但 ``re.sub(r'\W+',' ',q)`` 的 ``\W``
对中文（Unicode 字母）不生效，纯中文查询不分词 -> 仍整串一个 token。

修复：装 zhparser 扩展（postgres/Dockerfile 编译 SCWS+zhparser），建 ``chinese`` 文本搜索
配置（PARSER=zhparser），把 ``text_search_tsv`` 列的 generated 表达式从
``to_tsvector('simple', text)`` 改成 ``to_tsvector('chinese', text)``。文档侧分词后每个
中文词是独立 token。查询侧由 ``_PgSparseStore.bm25_search`` 用 zhparser 分词 + OR
（见 retriever/llamaindex_pg.py）--库的 to_tsquery 不分词，必须绕开。

zhparser 默认 ``dict_in_memory=off``，词典走 mmap + OS page cache，多 backend 共享同一
物理页，per-connection 内存趋近 0--适配 2 核 4GB 部署。

SQLite 测试基座跳过（CREATE EXTENSION / tsvector / generated column 均为 PG 专有）。
greenfield PG 部署：本迁移先于 PGVectorStore.perform_setup 建表执行，data_kb_chunks 尚
不存在 -> 只建扩展+配置，列由 PGVectorStore 用 ``text_search_config='chinese'`` 建。
存量 PG 部署：表已存在（simple tsv）-> 重建列为 chinese（DROP 级联删 GIN，重算 527 行）。

Revision ID: 030
Revises: 029
Create Date: 2026-08-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "030"
down_revision: Union[str, None] = "029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# zhparser 词性 -> simple dictionary（不进一步词干化，保留原词形）。
# n名词 v动词 a形容词 i习语 e叹词 l惯用语 x非语素字，覆盖中文实词主体。
_ZH_MAPPING_TOKENS = "n,v,a,i,e,l,x"

# 与 PGVectorStore 建表同源（base.py:134 indexname = "%s_idx" % index_name，index_name=kb_chunks），
# 重建用同名避免 greenfield perform_setup 再建一个重复 GIN。
_TSV_GIN_INDEX = "kb_chunks_idx"
_TABLE = "data_kb_chunks"


def _is_postgres() -> bool:
    bind = op.get_bind()
    return bind.dialect.name == "postgresql"


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def _create_chinese_config() -> None:
    """CREATE EXTENSION + chinese 文本搜索配置（幂等）。"""
    op.execute("CREATE EXTENSION IF NOT EXISTS zhparser")
    # CREATE TEXT SEARCH CONFIGURATION 无 IF NOT EXISTS，DO 块 catch duplicate_object。
    op.execute(
        """
        DO $$ BEGIN
            CREATE TEXT SEARCH CONFIGURATION chinese (PARSER = zhparser);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
        """
    )
    # DROP MAPPING IF EXISTS 清旧映射再 ADD，保证幂等且终态正确（部分映射时不漏）。
    op.execute(
        f"ALTER TEXT SEARCH CONFIGURATION chinese "
        f"DROP MAPPING IF EXISTS FOR {_ZH_MAPPING_TOKENS}"
    )
    op.execute(
        f"ALTER TEXT SEARCH CONFIGURATION chinese "
        f"ADD MAPPING FOR {_ZH_MAPPING_TOKENS} WITH simple"
    )


def _rebuild_tsv_column(config: str) -> None:
    """重建 text_search_tsv 列的 generated 表达式为指定 config（DROP 级联删 GIN，重算全表）。"""
    if not _table_exists(_TABLE):
        return
    op.execute(f"ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS text_search_tsv")
    op.execute(
        f"ALTER TABLE {_TABLE} ADD COLUMN text_search_tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('{config}'::regconfig, text)) STORED"
    )
    op.execute(
        f"CREATE INDEX {_TSV_GIN_INDEX} ON {_TABLE} USING gin (text_search_tsv)"
    )


def upgrade() -> None:
    if not _is_postgres():
        # SQLite 测试基座跑全链 upgrade head（xfail）时安全跳过。
        return
    _create_chinese_config()
    _rebuild_tsv_column("chinese")


def downgrade() -> None:
    if not _is_postgres():
        return
    # 回退列到 simple（恢复改动前状态）；保留 zhparser 扩展与 chinese 配置（无依赖、无害）。
    _rebuild_tsv_column("simple")
