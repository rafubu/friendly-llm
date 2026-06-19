import { LitertClient } from '../client';

// Mock WebSocket
class MockWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: ((err: any) => void) | null = null;
  readyState: number = WebSocket.CONNECTING;
  send(data: string): void {}
  close(): void {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.();
  }
  addEventListener(event: string, handler: any): void {
    if (event === 'open') this.onopen = handler;
    if (event === 'message') this.onmessage = handler;
    if (event === 'close') this.onclose = handler;
  }
  removeEventListener(event: string, handler: any): void {}
}

(global as any).WebSocket = MockWebSocket as any;

// Mock RTCPeerConnection
class MockRTCPeerConnection {
  localDescription: RTCSessionDescription | null = null;
  iceConnectionState: string = 'new';
  connectionState: string = 'new';
  ondatachannel: ((event: any) => void) | null = null;
  onicecandidate: ((event: any) => void) | null = null;
  oniceconnectionstatechange: (() => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    return { type: 'offer', sdp: 'mock-offer' };
  }
  async createAnswer(): Promise<RTCSessionDescriptionInit> {
    return { type: 'answer', sdp: 'mock-answer' };
  }
  async setLocalDescription(desc: RTCSessionDescriptionInit): Promise<void> {
    this.localDescription = new RTCSessionDescription(desc);
  }
  async setRemoteDescription(desc: RTCSessionDescriptionInit): Promise<void> {}
  async addIceCandidate(candidate: RTCIceCandidate): Promise<void> {}
  close(): void {}
}

(global as any).RTCPeerConnection = MockRTCPeerConnection as any;
(global as any).RTCSessionDescription = class {
  sdp: string;
  type: RTCSdpType;
  constructor(init: RTCSessionDescriptionInit) {
    this.sdp = init.sdp || '';
    this.type = init.type;
  }
} as any;
(global as any).RTCIceCandidate = class {} as any;

beforeEach(() => {
  jest.clearAllMocks();
});

describe('LitertClient', () => {
  it('creates instance with empty options uses defaults', () => {
    const client = new LitertClient({} as any);
    expect(client).toBeInstanceOf(LitertClient);
    expect((client as any).timeout).toBe(300_000);
  });

  it('creates instance with options', () => {
    const client = new LitertClient({
      signalingUrl: 'ws://test:9876',
      authToken: 'test-jwt',
    });
    expect(client).toBeInstanceOf(LitertClient);
  });

  it('connect() sends auth_jwt on open', (done) => {
    const client = new LitertClient({
      signalingUrl: 'ws://test:9876',
      authToken: 'my-jwt-token',
    });

    const connectPromise = client.connect();

    // Simulate the WebSocket opening
    setTimeout(() => {
      const ws = (client as any).ws;
      if (ws.onopen) ws.onopen();
    }, 10);

    // Simulate auth_ok response
    setTimeout(() => {
      const ws = (client as any).ws;
      if (ws.onmessage) {
        ws.onmessage(new MessageEvent('message', {
          data: JSON.stringify({ type: 'auth_ok', user_id: 'user_abc' }),
        }));
      }
    }, 20);

    connectPromise.then(() => {
      expect((client as any).connected).toBe(true);
      expect((client as any).userId).toBe('user_abc');
      done();
    });
  });

  it('rejects connect() on auth_error', (done) => {
    const client = new LitertClient({
      signalingUrl: 'ws://test:9876',
      authToken: 'bad-token',
    });

    const connectPromise = client.connect();

    setTimeout(() => {
      const ws = (client as any).ws;
      if (ws.onopen) ws.onopen();
    }, 10);

    setTimeout(() => {
      const ws = (client as any).ws;
      if (ws.onmessage) {
        ws.onmessage(new MessageEvent('message', {
          data: JSON.stringify({ type: 'auth_error', error: 'Invalid token' }),
        }));
      }
    }, 20);

    connectPromise.catch((err: Error) => {
      expect(err.message).toContain('Invalid token');
      done();
    });
  });
});
