import { Chunk, Response, ModelInfo } from './types';

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    } catch (e) {
      reject(e);
    }
  });
}

export class LitertLocalClient {
  private baseUrl: string;
  private model: string;

  constructor(baseUrl = 'http://127.0.0.1:11434', model = '') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.model = model;
  }

  async chatSync(
    text: string,
    opts?: { images?: (string | File)[]; tools?: any[]; format?: string }
  ): Promise<Response> {
    const payload = await this.buildPayload(text, opts, false);
    const resp = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    return { text: data.message?.content || '', toolCalls: data.message?.tool_calls };
  }

  async *chat(
    text: string,
    opts?: { images?: (string | File)[]; tools?: any[]; format?: string }
  ): AsyncGenerator<Chunk> {
    const payload = await this.buildPayload(text, opts, true);
    const resp = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (!resp.body) throw new Error('Response body is null');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.done) {
            yield {
              text: '',
              done: true,
              doneReason: data.done_reason || 'stop',
              toolCalls: data.message?.tool_calls,
              evalCount: data.eval_count,
              totalDuration: data.total_duration,
            };
            return;
          }
          yield {
            text: data.message?.content || '',
            done: false,
            toolCalls: data.message?.tool_calls,
          };
        } catch (e) {
          console.error('Parse error:', e);
        }
      }
    }
  }

  async listModels(): Promise<ModelInfo[]> {
    const resp = await fetch(`${this.baseUrl}/api/tags`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    return (data.models || []).map((m: any) => ({ id: m.name, name: m.name }));
  }

  private async buildPayload(text: string, opts?: any, stream = true): Promise<any> {
    const msg: any = { role: 'user', content: text };
    if (opts?.images) {
      const b64Images: string[] = [];
      for (const img of opts.images) {
        if (img instanceof File) {
          const b64 = await fileToBase64(img);
          b64Images.push(b64);
        } else if (typeof img === 'string') {
          try {
            const resp = await fetch(img);
            const blob = await resp.blob();
            const file = new File([blob], 'image', { type: blob.type });
            const b64 = await fileToBase64(file);
            b64Images.push(b64);
          } catch {
            console.warn('Failed to load image, skipping');
          }
        }
      }
      if (b64Images.length > 0) msg.images = b64Images;
    }
    const payload: any = { model: this.model, messages: [msg], stream };
    if (opts?.tools) payload.tools = opts.tools;
    if (opts?.format) payload.format = opts.format;
    return payload;
  }
}
