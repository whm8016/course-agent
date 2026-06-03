import { useEffect, useState, useRef, useCallback } from 'react'
import { fetchGraphs, deleteGraphNode, type GraphNode, type GraphData, type GraphsResponse } from '../../services/api'

interface CanvasNode extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
}

function getColor(node: GraphNode): string {
  if (node.type === 'error_pattern') {
    const severity = node.severity ?? 0.5
    const r = Math.round(200 + severity * 55)
    return `rgb(${r}, ${Math.round(80 - severity * 40)}, ${Math.round(80 - severity * 40)})`
  }
  const mastery = node.mastery ?? 0.5
  const risk = node.risk ?? 0.5
  if (risk > 0.7) return '#ef4444'
  if (mastery > 0.7) return '#22c55e'
  return '#3b82f6'
}

function getRadius(node: GraphNode): number {
  if (node.type === 'error_pattern') {
    return 12 + (node.error_count ?? 1) * 3
  }
  const importance = node.importance ?? 0.5
  return 14 + importance * 12
}

export default function GraphPage({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<GraphsResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'knowledge' | 'error'>('knowledge')
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [loading, setLoading] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const nodesRef = useRef<CanvasNode[]>([])
  const edgesRef = useRef<{ source: string; target: string }[]>([])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fetchGraphs()
      setData(result)
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { void loadData() }, [loadData])

  const currentGraph: GraphData = data
    ? (activeTab === 'knowledge' ? data.knowledge_graph : data.error_graph)
    : { nodes: [], edges: [] }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !currentGraph.nodes.length) {
      nodesRef.current = []
      return
    }
    canvas.width = canvas.offsetWidth * devicePixelRatio
    canvas.height = canvas.offsetHeight * devicePixelRatio
    const ctx = canvas.getContext('2d')!
    ctx.scale(devicePixelRatio, devicePixelRatio)
    const cw = canvas.offsetWidth
    const ch = canvas.offsetHeight

    const nodes: CanvasNode[] = currentGraph.nodes.map((n) => ({
      ...n,
      x: cw / 2 + (Math.random() - 0.5) * cw * 0.6,
      y: ch / 2 + (Math.random() - 0.5) * ch * 0.6,
      vx: 0,
      vy: 0,
      radius: getRadius(n),
    }))
    nodesRef.current = nodes
    edgesRef.current = currentGraph.edges

    const nodeMap = new Map(nodes.map(n => [n.id, n]))

    let frame = 0
    function simulate() {
      frame++
      for (const node of nodes) {
        node.vx *= 0.9
        node.vy *= 0.9
        const dx = cw / 2 - node.x
        const dy = ch / 2 - node.y
        node.vx += dx * 0.0005
        node.vy += dy * 0.0005
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j]
          const dx = b.x - a.x
          const dy = b.y - a.y
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
          const force = 800 / (dist * dist)
          const fx = (dx / dist) * force
          const fy = (dy / dist) * force
          a.vx -= fx; a.vy -= fy
          b.vx += fx; b.vy += fy
        }
      }

      for (const edge of edgesRef.current) {
        const a = nodeMap.get(edge.source)
        const b = nodeMap.get(edge.target)
        if (!a || !b) continue
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        const force = (dist - 120) * 0.01
        const fx = (dx / Math.max(dist, 1)) * force
        const fy = (dy / Math.max(dist, 1)) * force
        a.vx += fx; a.vy += fy
        b.vx -= fx; b.vy -= fy
      }

      for (const node of nodes) {
        node.x += node.vx
        node.y += node.vy
        node.x = Math.max(node.radius, Math.min(cw - node.radius, node.x))
        node.y = Math.max(node.radius, Math.min(ch - node.radius, node.y))
      }

      ctx.clearRect(0, 0, cw, ch)

      ctx.strokeStyle = '#e2e8f0'
      ctx.lineWidth = 1
      for (const edge of edgesRef.current) {
        const a = nodeMap.get(edge.source)
        const b = nodeMap.get(edge.target)
        if (!a || !b) continue
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }

      for (const node of nodes) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        ctx.fillStyle = getColor(node)
        ctx.globalAlpha = node.status === 'candidate' ? 0.5 : 0.9
        ctx.fill()
        ctx.globalAlpha = 1
        ctx.strokeStyle = selected?.id === node.id ? '#1e293b' : '#fff'
        ctx.lineWidth = selected?.id === node.id ? 3 : 2
        ctx.stroke()

        ctx.fillStyle = '#1e293b'
        ctx.font = '11px sans-serif'
        ctx.textAlign = 'center'
        const label = node.label.length > 6 ? node.label.slice(0, 6) + '…' : node.label
        ctx.fillText(label, node.x, node.y + node.radius + 14)
      }

      if (frame < 200) {
        animRef.current = requestAnimationFrame(simulate)
      }
    }

    animRef.current = requestAnimationFrame(simulate)
    return () => cancelAnimationFrame(animRef.current)
  }, [currentGraph, selected])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const hit = nodesRef.current.find(n => {
      const dx = n.x - x, dy = n.y - y
      return dx * dx + dy * dy <= n.radius * n.radius
    })
    setSelected(hit || null)
  }

  const handleDelete = async () => {
    if (!selected) return
    try {
      const result = await deleteGraphNode(selected.id)
      setData(result)
      setSelected(null)
    } catch { /* ignore */ }
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <header className="flex items-center gap-4 px-6 py-3 bg-white border-b border-slate-200">
        <button onClick={onBack} className="text-sm text-slate-500 hover:text-slate-800">&larr; 返回</button>
        <h1 className="text-lg font-semibold">学习图谱</h1>
        <div className="flex gap-2 ml-auto">
          <button
            onClick={() => { setActiveTab('knowledge'); setSelected(null) }}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'knowledge' ? 'bg-blue-600 text-white' : 'bg-slate-200'}`}
          >知识点图谱</button>
          <button
            onClick={() => { setActiveTab('error'); setSelected(null) }}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'error' ? 'bg-red-600 text-white' : 'bg-slate-200'}`}
          >错题图谱</button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 relative">
          {loading ? (
            <div className="flex items-center justify-center h-full text-slate-400">加载中...</div>
          ) : currentGraph.nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-400">
              暂无数据，开始对话后将自动生成图谱
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              className="w-full h-full cursor-pointer"
              onClick={handleCanvasClick}
            />
          )}
        </div>

        {selected && (
          <aside className="w-72 bg-white border-l border-slate-200 p-4 overflow-y-auto">
            <h3 className="font-semibold text-base mb-2">{selected.label}</h3>
            <div className="text-xs text-slate-500 mb-3">{selected.type === 'knowledge_point' ? '知识点' : '错误模式'}</div>

            {selected.type === 'knowledge_point' && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span>风险度</span><span className="font-mono">{((selected.risk ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>掌握度</span><span className="font-mono">{((selected.mastery ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>重要性</span><span className="font-mono">{((selected.importance ?? 0) * 100).toFixed(0)}%</span></div>
              </div>
            )}

            {selected.type === 'error_pattern' && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span>严重度</span><span className="font-mono">{((selected.severity ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>出错次数</span><span className="font-mono">{selected.error_count ?? 1}</span></div>
              </div>
            )}

            {selected.notes && <p className="mt-3 text-sm text-slate-600">{selected.notes}</p>}

            {selected.examples && selected.examples.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-slate-500 mb-1">示例</div>
                <ul className="text-sm space-y-1">
                  {selected.examples.map((ex, i) => <li key={i} className="text-slate-600">• {ex}</li>)}
                </ul>
              </div>
            )}

            {selected.correction_suggestions && selected.correction_suggestions.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-slate-500 mb-1">纠正建议</div>
                <ul className="text-sm space-y-1">
                  {selected.correction_suggestions.map((s, i) => <li key={i} className="text-slate-600">• {s}</li>)}
                </ul>
              </div>
            )}

            <button
              onClick={handleDelete}
              className="mt-4 w-full px-3 py-1.5 bg-red-50 text-red-600 rounded text-sm hover:bg-red-100"
            >删除此节点</button>
          </aside>
        )}
      </div>
    </div>
  )
}
