import asyncio
from rag_llama.embedding_bridge import get_embedding_client

async def test():
    client = get_embedding_client()
    vecs = await client.embed(["你好", "LlamaIndex 测试"])
    print(f"向量数: {len(vecs)}, 维度: {len(vecs[0])}")

asyncio.run(test())