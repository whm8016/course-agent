import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
import {
  fetchCourses, fetchGraphs, deleteGraphNode, fetchStudentTopicGraph,
  type GraphNode, type GraphData, type GraphsResponse,
  type StudentTopic, type StudentTopicEdge, type StudentTopicGraph,
} from '../../services/api'

interface CanvasNode extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
}

// 节点填色与设计系统 token 对齐（去饱和的 muted pastel 语义色）
function getColor(node: GraphNode): string {
  if (node.type === 'error_pattern') {
    const severity = node.severity ?? 0.5
    const r = Math.round(200 + severity * 55)
    return `rgb(${r}, ${Math.round(80 - severity * 40)}, ${Math.round(80 - severity * 40)})`
  }
  const mastery = node.mastery ?? 0.5
  const risk = node.risk ?? 0.5
  if (risk > 0.7) return '#9F2F2D' // danger-fg
  if (mastery > 0.7) return '#346538' // ok-fg
  return '#1F6C9F' // info-fg
}

function getRadius(node: GraphNode): number {
  if (node.type === 'error_pattern') {
    return 12 + (node.error_count ?? 1) * 3
  }
  const importance = node.importance ?? 0.5
  return 14 + importance * 12
}

// ── 文件内共享 helper（力导向 / course 分层 DAG 两套绘制共用）────────────────

interface HiDpiCanvas {
  ctx: CanvasRenderingContext2D
  cw: number
  ch: number
}

function setupHiDpiCanvas(canvas: HTMLCanvasElement): HiDpiCanvas {
  canvas.width = canvas.offsetWidth * devicePixelRatio
  canvas.height = canvas.offsetHeight * devicePixelRatio
  const ctx = canvas.getContext('2d')!
  ctx.scale(devicePixelRatio, devicePixelRatio)
  return { ctx, cw: canvas.offsetWidth, ch: canvas.offsetHeight }
}

function hitTest<N extends { x: number; y: number; radius: number }>(
  nodes: N[], x: number, y: number,
): N | undefined {
  return nodes.find(n => {
    const dx = n.x - x, dy = n.y - y
    return dx * dx + dy * dy <= n.radius * n.radius
  })
}

function drawNodeLabel(ctx: CanvasRenderingContext2D, label: string, x: number, y: number, radius: number) {
  ctx.fillStyle = '#5C615F' // ink-soft
  ctx.font = '11px sans-serif'
  ctx.textAlign = 'center'
  const text = label.length > 6 ? label.slice(0, 6) + '…' : label
  ctx.fillText(text, x, y + radius + 14)
}

// ── 课程先修图谱（第三 tab）：静态分层 DAG，不跑力导向 ────────────────────────

interface CourseNode extends StudentTopic {
  x: number
  y: number
  radius: number
}

/** 拓扑分层：layer(n) = 1 + max(layer(前置))，无前置归第 0 层（学习起点）。
    推断边（闭包补的）与直接边一起参与分层——推断边是可达性的重复表达，不改变层号。 */
function assignLayers(topics: StudentTopic[], edges: StudentTopicEdge[]): Map<string, number> {
  const preds = new Map<string, string[]>()
  for (const e of edges) {
    if (!preds.has(e.dst)) preds.set(e.dst, [])
    preds.get(e.dst)!.push(e.src)
  }
  const layer = new Map<string, number>()
  const visiting = new Set<string>()
  const getLayer = (tid: string): number => {
    const cached = layer.get(tid)
    if (cached !== undefined) return cached
    if (visiting.has(tid)) return 0 // 防御：异常数据成环时按第 0 层兜底
    visiting.add(tid)
    const ps = preds.get(tid) ?? []
    const l = ps.length === 0 ? 0 : 1 + Math.max(...ps.map(getLayer))
    visiting.delete(tid)
    layer.set(tid, l)
    return l
  }
  for (const t of topics) getLayer(t.topic_id)
  return layer
}

/** 分层布局：x = 层列均分（左→右即学习顺序），y = 层内堆叠。 */
function layoutCourseGraph(
  topics: StudentTopic[], edges: StudentTopicEdge[], cw: number, ch: number,
): CourseNode[] {
  const layer = assignLayers(topics, edges)
  const byLayer = new Map<number, StudentTopic[]>()
  let maxLayer = 0
  for (const t of topics) {
    const l = layer.get(t.topic_id) ?? 0
    maxLayer = Math.max(maxLayer, l)
    const g = byLayer.get(l)
    if (g) g.push(t)
    else byLayer.set(l, [t])
  }
  const nodes: CourseNode[] = []
  for (const [l, group] of byLayer) {
    group.forEach((t, i) => {
      nodes.push({
        ...t,
        x: (l + 0.5) * (cw / (maxLayer + 1)),
        y: (i + 0.5) * (ch / group.length),
        radius: 16,
      })
    })
  }
  return nodes
}

/** 着色（Khan/LeetCode/Duolingo 公约数语义）：灰=未开始、红=薄弱（与学情预警
    门控同口径，后端算好 weak）、绿=已掌握、蓝渐变=学习中（越深越熟）。 */
function courseNodeColor(my: StudentTopic['my']): string {
  if (!my) return '#8A8F94'
  if (my.weak) return '#9F2F2D'
  if (my.mastery > 0.7) return '#346538'
  return `rgba(31, 108, 159, ${(0.35 + my.mastery * 0.6).toFixed(2)})`
}

export default function GraphPage({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<GraphsResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'knowledge' | 'error' | 'course'>('knowledge')
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [loading, setLoading] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const nodesRef = useRef<CanvasNode[]>([])
  const edgesRef = useRef<{ source: string; target: string }[]>([])

  // 课程先修图谱（course tab）：课程选择 + 学生视角主题图 + 选中节点 id
  // （选中主题对象由 topicById 派生——state 不存布局快照，图重绘后无 stale）
  const [courses, setCourses] = useState<{ id: string; name: string }[]>([])
  const [courseId, setCourseId] = useState('')
  const [courseGraph, setCourseGraph] = useState<StudentTopicGraph | null>(null)
  const [courseLoading, setCourseLoading] = useState(false)
  const [selectedTopicId, setSelectedTopicId] = useState<string | null>(null)
  const courseNodesRef = useRef<CourseNode[]>([])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fetchGraphs()
      setData(result)
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { void loadData() }, [loadData])

  // useMemo 稳定引用，避免每次渲染都触发下方绘制 effect
  const currentGraph = useMemo<GraphData>(() => data
    ? (activeTab === 'knowledge' ? data.knowledge_graph : data.error_graph)
    : { nodes: [], edges: [] }, [data, activeTab])

  useEffect(() => {
    const canvas = canvasRef.current
    // course tab 用独立的静态分层绘制 effect，共用同一 canvas——此处须让位
    if (activeTab === 'course' || !canvas || !currentGraph.nodes.length) {
      nodesRef.current = []
      return
    }
    const { ctx, cw, ch } = setupHiDpiCanvas(canvas)

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

      ctx.strokeStyle = '#EAEAEA' // line
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
        ctx.strokeStyle = selected?.id === node.id ? '#2F3437' : '#FFFFFF' // ink / surface
        ctx.lineWidth = selected?.id === node.id ? 3 : 2
        ctx.stroke()

        drawNodeLabel(ctx, node.label, node.x, node.y, node.radius)
      }

      if (frame < 200) {
        animRef.current = requestAnimationFrame(simulate)
      }
    }

    animRef.current = requestAnimationFrame(simulate)
    return () => cancelAnimationFrame(animRef.current)
  }, [activeTab, currentGraph, selected])

  // ── course tab：课程列表 lazy 加载（学生只看到自己选的课） ──────────────────
  useEffect(() => {
    if (activeTab !== 'course' || courses.length > 0) return
    fetchCourses()
      .then(cs => {
        setCourses(cs.map(c => ({ id: c.id, name: c.name })))
        if (cs.length > 0) setCourseId(prev => prev || cs[0].id)
      })
      .catch(() => { /* 选课列表加载失败时保持空态 */ })
  }, [activeTab, courses.length])

  useEffect(() => {
    // 同课程数据不重拉（数据只在教师重建后变化）；切 tab 往返零请求零闪烁
    if (activeTab !== 'course' || !courseId || courseGraph?.course_id === courseId) return
    let cancelled = false
    setCourseLoading(true)
    fetchStudentTopicGraph(courseId)
      .then(g => { if (!cancelled) setCourseGraph(g) })
      .catch(() => { if (!cancelled) setCourseGraph(null) })
      .finally(() => { if (!cancelled) setCourseLoading(false) })
    return () => { cancelled = true }
  }, [activeTab, courseId, courseGraph?.course_id])

  // course tab：静态分层 DAG 单次绘制（选中节点变化时重绘高亮邻接）
  useEffect(() => {
    const canvas = canvasRef.current
    if (activeTab !== 'course' || !canvas || !courseGraph || courseGraph.topic_count === 0) {
      courseNodesRef.current = []
      return
    }
    const { ctx, cw, ch } = setupHiDpiCanvas(canvas)
    const nodes = layoutCourseGraph(courseGraph.topics, courseGraph.edges, cw, ch)
    courseNodesRef.current = nodes
    const nodeMap = new Map(nodes.map(n => [n.topic_id, n]))
    // 只画 LLM 直接判定的边（推断边是可达性重复表达，画出来徒增交叉线；
    // 分层仍用全部边——层号不受推断边影响）
    const directEdges = courseGraph.edges.filter(e => e.relation !== 'prerequisite_inferred')

    const selId = selectedTopicId
    const adjIds = new Set<string>()
    if (selId) {
      for (const e of directEdges) {
        if (e.dst === selId) adjIds.add(e.src)
        if (e.src === selId) adjIds.add(e.dst)
      }
    }

    ctx.clearRect(0, 0, cw, ch)
    // 先修边：src（前置，左层）→ dst（被前置，右层），dst 端箭头
    for (const e of directEdges) {
      const a = nodeMap.get(e.src), b = nodeMap.get(e.dst)
      if (!a || !b) continue
      const hl = selId != null && (e.src === selId || e.dst === selId)
      ctx.strokeStyle = hl ? '#1F6C9F' : '#D8DCDB'
      ctx.lineWidth = hl ? 2 : 1.2
      const ang = Math.atan2(b.y - a.y, b.x - a.x)
      const ax = a.x + Math.cos(ang) * (a.radius + 3)
      const ay = a.y + Math.sin(ang) * (a.radius + 3)
      const bx = b.x - Math.cos(ang) * (b.radius + 9)
      const by = b.y - Math.sin(ang) * (b.radius + 9)
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.stroke()
      // 箭头（指向被前置主题）
      const wing = 0.45
      ctx.beginPath()
      ctx.moveTo(bx + Math.cos(ang) * 6, by + Math.sin(ang) * 6)
      ctx.lineTo(bx - Math.cos(ang - wing) * 8, by - Math.sin(ang - wing) * 8)
      ctx.lineTo(bx - Math.cos(ang + wing) * 8, by - Math.sin(ang + wing) * 8)
      ctx.closePath()
      ctx.fillStyle = hl ? '#1F6C9F' : '#B7BCBB'
      ctx.fill()
    }
    for (const n of nodes) {
      ctx.beginPath()
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2)
      ctx.fillStyle = courseNodeColor(n.my)
      ctx.globalAlpha = 0.9
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = selId === n.topic_id ? '#2F3437' : (adjIds.has(n.topic_id) ? '#1F6C9F' : '#FFFFFF')
      ctx.lineWidth = selId === n.topic_id ? 3 : 2
      ctx.stroke()
      drawNodeLabel(ctx, n.label, n.x, n.y, n.radius)
    }
  }, [activeTab, courseGraph, selectedTopicId])

  // 侧栏选中主题（由 id 派生）与「直接前置」列表（含各自薄弱红点）
  const topicById = useMemo(
    () => new Map((courseGraph?.topics ?? []).map(t => [t.topic_id, t])),
    [courseGraph],
  )
  const selectedTopic = selectedTopicId ? (topicById.get(selectedTopicId) ?? null) : null
  const directPrereqs = useMemo(() => {
    if (!courseGraph || !selectedTopic) return []
    return courseGraph.edges
      .filter(e => e.dst === selectedTopic.topic_id && e.relation !== 'prerequisite_inferred')
      .map(e => topicById.get(e.src))
      .filter((t): t is StudentTopic => !!t)
  }, [courseGraph, selectedTopic, topicById])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    if (activeTab === 'course') {
      setSelectedTopicId(hitTest(courseNodesRef.current, x, y)?.topic_id ?? null)
      return
    }
    setSelected(hitTest(nodesRef.current, x, y) || null)
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
    <div className="h-screen flex flex-col bg-canvas">
      <header className="flex items-center gap-4 px-6 py-3 bg-surface border-b border-line">
        <button onClick={onBack} className="text-sm text-ink-soft hover:text-ink transition">&larr; 返回</button>
        <h1 className="text-lg font-serif text-ink">学习图谱</h1>
        <div className="flex gap-2 ml-auto items-center">
          {activeTab === 'course' && courses.length > 0 && (
            <select
              value={courseId}
              onChange={e => { setCourseId(e.target.value); setSelectedTopicId(null) }}
              className="text-sm bg-surface-2 text-ink rounded-[var(--radius)] px-2 py-1 border border-line max-w-[180px]"
            >
              {courses.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          )}
          <button
            onClick={() => { setActiveTab('knowledge'); setSelected(null) }}
            className={`px-3 py-1 rounded-[var(--radius)] text-sm transition ${activeTab === 'knowledge' ? 'bg-accent text-white' : 'bg-surface-2 text-ink-soft hover:text-ink'}`}
          >知识点图谱</button>
          <button
            onClick={() => { setActiveTab('error'); setSelected(null) }}
            className={`px-3 py-1 rounded-[var(--radius)] text-sm transition ${activeTab === 'error' ? 'bg-danger-fg text-white' : 'bg-surface-2 text-ink-soft hover:text-ink'}`}
          >错题图谱</button>
          <button
            onClick={() => { setActiveTab('course'); setSelected(null); setSelectedTopicId(null) }}
            className={`px-3 py-1 rounded-[var(--radius)] text-sm transition ${activeTab === 'course' ? 'bg-info-fg text-white' : 'bg-surface-2 text-ink-soft hover:text-ink'}`}
          >课程先修图谱</button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 relative">
          {activeTab === 'course' ? (
            courseLoading ? (
              <div className="flex items-center justify-center h-full text-muted">加载中...</div>
            ) : !courseGraph || courseGraph.topic_count === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-muted gap-1.5">
                <span>教师尚未生成课程图谱</span>
                <span className="text-xs">（需教师在知识库详情中「生成主题图」后此处可见）</span>
              </div>
            ) : (
              <canvas
                ref={canvasRef}
                className="w-full h-full cursor-pointer"
                onClick={handleCanvasClick}
              />
            )
          ) : loading ? (
            <div className="flex items-center justify-center h-full text-muted">加载中...</div>
          ) : currentGraph.nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted">
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

        {activeTab === 'course' && selectedTopic && (
          <aside className="w-72 bg-surface border-l border-line p-4 overflow-y-auto">
            <h3 className="font-serif text-base text-ink mb-1">{selectedTopic.label}</h3>
            <div className="text-xs text-muted mb-3">{selectedTopic.source_section}</div>
            {selectedTopic.definition && (
              <p className="text-sm text-ink-soft mb-3">{selectedTopic.definition}</p>
            )}
            {selectedTopic.my ? (
              <div className="space-y-2 text-sm text-ink-soft">
                {selectedTopic.my.weak && (
                  <div className="text-xs px-2 py-1 rounded bg-danger-bg text-danger-fg">
                    薄弱环节（与学情预警同口径）
                  </div>
                )}
                <div className="flex justify-between">
                  <span>掌握度</span>
                  <span className="font-mono text-ink">{(selectedTopic.my.mastery * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>风险度</span>
                  <span className="font-mono text-ink">{(selectedTopic.my.risk * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>观测次数</span>
                  <span className="font-mono text-ink">{selectedTopic.my.observation_count}</span>
                </div>
              </div>
            ) : (
              <div className="text-sm text-muted">尚未学习此知识点</div>
            )}
            {directPrereqs.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-muted mb-1">直接前置（应先掌握）</div>
                <ul className="text-sm space-y-1">
                  {directPrereqs.map(p => (
                    <li key={p.topic_id} className="flex items-center gap-1.5 text-ink-soft">
                      {p.my?.weak && <span className="inline-block w-1.5 h-1.5 rounded-full bg-danger-fg shrink-0" />}
                      <span className="truncate">{p.label}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>
        )}

        {activeTab !== 'course' && selected && (
          <aside className="w-72 bg-surface border-l border-line p-4 overflow-y-auto">
            <h3 className="font-serif text-base text-ink mb-2">{selected.label}</h3>
            <div className="text-xs text-muted mb-3">{selected.type === 'knowledge_point' ? '知识点' : '错误模式'}</div>

            {selected.type === 'knowledge_point' && (
              <div className="space-y-2 text-sm text-ink-soft">
                <div className="flex justify-between"><span>风险度</span><span className="font-mono text-ink">{((selected.risk ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>掌握度</span><span className="font-mono text-ink">{((selected.mastery ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>重要性</span><span className="font-mono text-ink">{((selected.importance ?? 0) * 100).toFixed(0)}%</span></div>
              </div>
            )}

            {selected.type === 'error_pattern' && (
              <div className="space-y-2 text-sm text-ink-soft">
                <div className="flex justify-between"><span>严重度</span><span className="font-mono text-ink">{((selected.severity ?? 0) * 100).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span>出错次数</span><span className="font-mono text-ink">{selected.error_count ?? 1}</span></div>
              </div>
            )}

            {selected.notes && <p className="mt-3 text-sm text-ink-soft">{selected.notes}</p>}

            {selected.examples && selected.examples.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-muted mb-1">示例</div>
                <ul className="text-sm space-y-1">
                  {selected.examples.map((ex, i) => <li key={i} className="text-ink-soft">• {ex}</li>)}
                </ul>
              </div>
            )}

            {selected.correction_suggestions && selected.correction_suggestions.length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-muted mb-1">纠正建议</div>
                <ul className="text-sm space-y-1">
                  {selected.correction_suggestions.map((s, i) => <li key={i} className="text-ink-soft">• {s}</li>)}
                </ul>
              </div>
            )}

            <button
              onClick={handleDelete}
              className="mt-4 w-full px-3 py-1.5 bg-danger-bg text-danger-fg rounded-[var(--radius)] text-sm hover:opacity-80 transition"
            >删除此节点</button>
          </aside>
        )}
      </div>
    </div>
  )
}
