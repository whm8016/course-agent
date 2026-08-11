# pgvector + zhparser：在官方 pgvector/pgvector:pg16 基础上编译 SCWS + zhparser，
# 让 PG tsvector 对中文按词切分（simple 配置不分词，整句压成一个 token -> sparse 0 命中）。
#
# zhparser 默认 dict_in_memory=off，词典走 mmap + OS page cache，多个 backend 进程共享
# 同一份物理内存页，per-connection 内存增量趋近 0--适配 2 核 4GB 部署（见 alembic 030）。
FROM pgvector/pgvector:pg16

# 编译工具链 + PG16 开发头文件（pg_config 在 /usr/lib/postgresql/16/bin/）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# SCWS 1.2.3：zhparser 依赖的中文分词核心库（装到 /usr/local）
RUN wget -q -O /tmp/scws.tar.bz2 http://www.xunsearch.com/scws/down/scws-1.2.3.tar.bz2 \
    && tar xf /tmp/scws.tar.bz2 -C /tmp \
    && cd /tmp/scws-1.2.3 && ./configure && make install \
    && ldconfig \
    && rm -rf /tmp/scws*

# zhparser v2.3：PG 中文分词扩展。PG_CONFIG 指向 16，make install 装到 PG extension 目录
ENV PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
RUN wget -q -O /tmp/zhparser.tar.gz https://github.com/amutu/zhparser/tarball/v2.3 \
    && mkdir -p /tmp/zhparser \
    && tar xf /tmp/zhparser.tar.gz -C /tmp/zhparser --strip-components=1 \
    && cd /tmp/zhparser && make && make install \
    && rm -rf /tmp/zhparser*
