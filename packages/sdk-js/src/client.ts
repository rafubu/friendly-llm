import {
  LitertClientOptions,
  Chunk,
  ModelInfo,
  SignalingMessage,
} from './types';

interface PendingRequest {
  chunks: { resolve: Function; reject: Function }[];
  finished: boolean;
}

export class LitertClient {
  private signalingUrl: string;
  private authToken: string;
  private model: string | null;
  private timeout: number;
  private verifyFp: boolean;
  private iceServers: RTCIceServer[];

  private ws: WebSocket | null = null;
  private connected = false;
  private userId: string | null = null;
  private roomId: string | null = null;
  private nodeId: string | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private pc: RTCPeerConnection | null = null;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private wsHandlers: Set<{ target: EventTarget; type: string; handler: EventListener }> = new Set();
  private timeoutIds: (number | NodeJS.Timeout)[] = [];
  private _wsOnMessage: EventListener | null = null;

  constructor(options: LitertClientOptions) {
    this.signalingUrl = options.signalingUrl;
    this.authToken = options.authToken;
    this.model = options.model || null;
    this.timeout = options.timeout || 300_000;
    this.verifyFp = options.verifyFingerprint !== false;
    this.iceServers = options.iceServers || [{ urls: 'stun:stun.l.google.com:19302' }];
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.signalingUrl);
      } catch (e) {
        reject(e);
        return;
      }

      this.ws.onopen = () => {
        this.ws?.send(JSON.stringify({ type: 'auth_jwt', token: this.authToken }));
      };

      const t = setTimeout(() => reject(new Error('Auth timeout')), 10000);
      this.timeoutIds.push(t);

      this._wsOnMessage = (event: Event) => {
        const msg = JSON.parse((event as MessageEvent).data);
        if (msg.type === 'auth_ok') {
          clearTimeout(t);
          this.userId = msg.user_id;
          this.connected = true;
          resolve();
          return;
        }
        if (msg.type === 'auth_error') {
          clearTimeout(t);
          reject(new Error(msg.error || 'Auth failed'));
          return;
        }
        this.handleMessage(msg);
      };

      this.ws.addEventListener('message', this._wsOnMessage);
    });
  }

  async listModels(): Promise<ModelInfo[]> {
    this.send({ type: 'list_nodes' });
    return new Promise((resolve, reject) => {
      const handler = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'node_list') {
            this.ws?.removeEventListener('message', handler);
            resolve(msg.nodes.map((n: any) => ({
              id: n.id,
              node: n.node,
              name: n.model,
              load: n.load,
              maxLoad: n.max_load,
              visibility: n.visibility,
            })));
          }
        } catch (e) {
          reject(e);
        }
      };
      const t = setTimeout(() => {
        this.ws?.removeEventListener('message', handler);
        reject(new Error('Timeout'));
      }, 10000);
      this.ws?.addEventListener('message', handler);
    });
  }

  async chat(text: string, opts?: { images?: (string | File)[]; tools?: any[]; format?: string }): Promise<AsyncIterator<Chunk>> {
    if (!this.connected) throw new Error('Not connected');

    const requestId = crypto.randomUUID();
    let finished = false;

    const request: PendingRequest = { chunks: [], finished: false };
    this.pendingRequests.set(requestId, request);

    const iterator: AsyncIterator<Chunk> = {
      next: (): Promise<IteratorResult<Chunk>> => {
        if (finished) {
          return Promise.resolve({ value: undefined as any, done: true });
        }
        if (request.chunks.length > 0) {
          const { resolve } = request.chunks.shift()!;
          return new Promise((r) => r(resolve()));
        }
        return new Promise((resolve, reject) => {
          request.chunks.push({ resolve, reject });
        });
      },
    };

    await this.ensureRoom();
    await this.ensureDataChannel();

    const payload: any = {
      model: this.model,
      messages: [{ role: 'user', content: text }],
      stream: true,
    };

    if (opts?.images) {
      payload.messages[0].images = [];
      for (const img of opts.images) {
        try {
          if (typeof img === 'string') {
            const response = await fetch(img);
            const blob = await response.blob();
            const b64 = await blobToBase64(blob);
            payload.messages[0].images.push(b64.split(',')[1]);
          } else {
            const b64 = await blobToBase64(img);
            payload.messages[0].images.push(b64.split(',')[1]);
          }
        } catch {
          console.warn('Failed to load image, skipping');
        }
      }
    }
    if (opts?.tools) payload.tools = opts.tools;
    if (opts?.format) payload.format = opts.format;

    this.dataChannel?.send(JSON.stringify({
      type: 'infer',
      request_id: requestId,
      endpoint: '/api/chat',
      payload,
    }));

    return iterator;

    const self = this;
    function resolveNext(result: Chunk | IteratorResult<Chunk>): void {
      if (request.chunks.length > 0) {
        const { resolve } = request.chunks.shift()!;
        resolve(result);
      } else {
        request.chunks.push({
          resolve: (r: any) => r(result),
          reject: () => {},
        });
      }
    }

    function handleData(data: any): void {
      if (data.type === 'done') {
        finished = true;
        request.finished = true;
        const chunk: Chunk = {
          text: '',
          done: true,
          doneReason: data.data?.done_reason || 'stop',
          evalCount: data.data?.eval_count,
          totalDuration: data.data?.total_duration,
        };
        resolveNext({ value: chunk, done: true });
        self.pendingRequests.delete(requestId);
      } else if (data.type === 'chunk') {
        const chunkData = data.data?.message || {};
        const chunk: Chunk = { text: chunkData.content || '', done: false };
        resolveNext({ value: chunk, done: false });
      } else if (data.type === 'error') {
        finished = true;
        request.finished = true;
        if (request.chunks.length > 0) {
          const { reject } = request.chunks.shift()!;
          reject(new Error(data.error));
        }
        self.pendingRequests.delete(requestId);
      }
    }
  }

  private async ensureRoom(): Promise<void> {
    if (this.roomId) return;

    return new Promise((resolve, reject) => {
      const handler = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'room_created') {
            this.roomId = msg.room_id;
            this.nodeId = msg.node;
            this.ws?.removeEventListener('message', handler);
            resolve();
          } else if (msg.type === 'error') {
            this.ws?.removeEventListener('message', handler);
            reject(new Error(msg.error));
          }
        } catch (e) { reject(e); }
      };
      this.ws?.addEventListener('message', handler);
      this.send({ type: 'create_room', model: this.model });
      setTimeout(() => {
        this.ws?.removeEventListener('message', handler);
        reject(new Error('Room creation timeout'));
      }, 15000);
    });
  }

  private async ensureDataChannel(): Promise<void> {
    if (this.dataChannel && this.dataChannel.readyState === 'open') return;

    return new Promise((resolve, reject) => {
      const self = this;
      const cleanup = () => {
        self.ws?.removeEventListener('message', sdpHandler);
      };

      const sdpHandler = async (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type !== 'sdp_offer') return;
          cleanup();

          const fingerprint = msg.fingerprint || '';
          if (self.verifyFp) {
            const expectedFp = self.getExpectedFingerprint();
            if (expectedFp && fingerprint !== expectedFp) {
              reject(new Error('DTLS fingerprint mismatch'));
              return;
            }
          }

          const pc = new RTCPeerConnection({
            iceServers: self.iceServers,
          });
          self.pc = pc;

          pc.ondatachannel = (event) => {
            const channel = event.channel;
            self.dataChannel = channel;
            channel.onmessage = (e) => {
              try {
                const data = JSON.parse(e.data);
                const reqId = data.request_id;
                const pending = self.pendingRequests.get(reqId);
                if (pending) {
                  if (data.type === 'done') {
                    pending.finished = true;
                    const chunk: Chunk = {
                      text: '', done: true,
                      doneReason: data.data?.done_reason || 'stop',
                      evalCount: data.data?.eval_count,
                      totalDuration: data.data?.total_duration,
                    };
                    self.deliverChunk(reqId, { value: chunk, done: true });
                    self.pendingRequests.delete(reqId);
                  } else if (data.type === 'chunk') {
                    const chunkData = data.data?.message || {};
                    const chunk: Chunk = { text: chunkData.content || '', done: false };
                    self.deliverChunk(reqId, { value: chunk, done: false });
                  } else if (data.type === 'error') {
                    pending.finished = true;
                    self.deliverError(reqId, new Error(data.error));
                    self.pendingRequests.delete(reqId);
                  }
                }
              } catch (err) {
                console.error('DataChannel message error:', err);
              }
            };
            channel.onopen = () => resolve();
          };

          pc.onicecandidate = (event) => {
            if (event.candidate) {
              self.ws?.send(JSON.stringify({
                type: 'ice_candidate',
                room_id: self.roomId,
                from: 'client',
                candidate: event.candidate.toJSON(),
              }));
            }
          };

          await pc.setRemoteDescription(
            new RTCSessionDescription(msg.sdp)
          );
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);

          if (!pc.localDescription) {
            reject(new Error('Failed to create local description'));
            return;
          }
          self.ws?.send(JSON.stringify({
            type: 'sdp_answer',
            room_id: self.roomId,
            sdp: { sdp: pc.localDescription.sdp, type: pc.localDescription.type },
          }));

          setTimeout(() => {
            if (self.dataChannel?.readyState !== 'open') {
              reject(new Error('DataChannel failed to open'));
            }
          }, 15000);
        } catch (e) { reject(e); }
      };

      this.ws?.addEventListener('message', sdpHandler);
    });
  }

  private deliverChunk(requestId: string, result: IteratorResult<Chunk>): void {
    const pending = this.pendingRequests.get(requestId);
    if (!pending) return;
    while (pending.chunks.length > 0) {
      const { resolve } = pending.chunks.shift()!;
      resolve(result);
      if (!result.done) return;
    }
    pending.chunks.push({ resolve: (r: any) => r(result), reject: () => {} });
  }

  private deliverError(requestId: string, error: Error): void {
    const pending = this.pendingRequests.get(requestId);
    if (!pending) return;
    while (pending.chunks.length > 0) {
      const { reject } = pending.chunks.shift()!;
      reject(error);
    }
  }

  private getExpectedFingerprint(): string {
    try {
      const b64 = this.authToken.split('.')[1]
        .replace(/-/g, '+')
        .replace(/_/g, '/');
      const payload = JSON.parse(atob(b64));
      return payload.dtls_fingerprint || '';
    } catch {
      return '';
    }
  }

  private handleMessage(msg: SignalingMessage): void {
    switch (msg.type) {
      case 'ice_candidate':
        if (this.pc && msg.candidate) {
          this.pc.addIceCandidate(new RTCIceCandidate(msg.candidate as any)).catch(() => {});
        }
        break;
      case 'room_closed':
        this.dataChannel?.close();
        this.dataChannel = null;
        this.roomId = null;
        this.pc?.close();
        this.pc = null;
        break;
      case 'sdp_answer':
        if (this.pc && msg.sdp) {
          this.pc.setRemoteDescription(new RTCSessionDescription(msg.sdp as any)).catch(() => {});
        }
        break;
    }
  }

  private send(msg: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    if (this.roomId) {
      this.send({ type: 'close_room', room_id: this.roomId });
    }
    this.dataChannel?.close();
    this.pc?.close();
    this.ws?.close();
    this.connected = false;
    this.pendingRequests.clear();
  }
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    } catch (e) {
      reject(e);
    }
  });
}
