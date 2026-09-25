/** WS 测试共享桩：FakeWebSocket（含全 4 个 readyState 静态常量——
 *  services 层引用 WebSocket.OPEN 等，桩缺常量会让 close 语义静默失效）。 */
import { vi } from 'vitest'

export class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  readyState = 1
  close = vi.fn()
  send = vi.fn()
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
}

/** stubGlobal + 清实例；配 afterEach 里 vi.unstubAllGlobals() 使用。 */
export function installFakeWs(): void {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
}
