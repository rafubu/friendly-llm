export interface Chunk {
  text: string;
  done: boolean;
  doneReason?: 'stop' | 'tool_calls' | 'error';
  toolCalls?: ToolCall[];
  evalCount?: number;
  totalDuration?: number;
}

export interface Response {
  text: string;
  toolCalls?: ToolCall[];
  usage?: { promptTokens: number; completionTokens: number };
}

export interface ToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

export interface ModelInfo {
  id: string;
  node: string;
  name: string;
  load: number;
  maxLoad: number;
  visibility: string;
}

export interface LitertClientOptions {
  signalingUrl: string;
  authToken: string;
  model?: string;
  iceServers?: RTCIceServer[];
  timeout?: number;
  verifyFingerprint?: boolean;
}

export interface SignalingMessage {
  type: string;
  [key: string]: unknown;
}
