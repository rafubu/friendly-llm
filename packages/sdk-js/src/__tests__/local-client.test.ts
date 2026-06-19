import { LitertLocalClient } from '../local-client';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

// Mock ReadableStream for NDJSON parsing
function createMockStream(lines: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index >= lines.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(lines[index] + '\n'));
      index++;
    },
  });
}

function mockResponse(status: number, body: any, isStream = false) {
  if (isStream) {
    return Promise.resolve({
      ok: status < 400,
      status,
      body: createMockStream(body),
      json: () => Promise.resolve(body),
    });
  }
  return Promise.resolve({
    ok: status < 400,
    status,
    json: () => Promise.resolve(body),
    body: null,
  });
}

beforeEach(() => {
  mockFetch.mockReset();
});

describe('LitertLocalClient', () => {
  describe('chatSync', () => {
    it('returns response text on success', async () => {
      mockFetch.mockResolvedValue(
        mockResponse(200, { message: { content: 'Hello!' } })
      );
      const client = new LitertLocalClient('http://test:11434', 'm');
      const result = await client.chatSync('hola');
      expect(result.text).toBe('Hello!');
    });

    it('throws on HTTP error', async () => {
      mockFetch.mockResolvedValue(mockResponse(500, {}));
      const client = new LitertLocalClient('http://test:11434');
      await expect(client.chatSync('hola')).rejects.toThrow('HTTP 500');
    });
  });

  describe('chat (streaming)', () => {
    it('yields content chunks and done', async () => {
      const lines = [
        JSON.stringify({ message: { content: 'Hello' }, done: false }),
        JSON.stringify({ message: { content: ' world' }, done: false }),
        JSON.stringify({ message: { content: '' }, done: true, done_reason: 'stop', eval_count: 5 }),
      ];
      mockFetch.mockResolvedValue(mockResponse(200, lines, true));

      const client = new LitertLocalClient('http://test:11434', 'm');
      const chunks: any[] = [];
      const gen = client.chat('hola');
      for await (const chunk of gen) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(3);
      expect(chunks[0].text).toBe('Hello');
      expect(chunks[0].done).toBe(false);
      expect(chunks[1].text).toBe(' world');
      expect(chunks[2].done).toBe(true);
      expect(chunks[2].doneReason).toBe('stop');
    });

    it('yields empty response on done without content', async () => {
      const lines = [
        JSON.stringify({ message: { content: '' }, done: true, done_reason: 'stop' }),
      ];
      mockFetch.mockResolvedValue(mockResponse(200, lines, true));

      const client = new LitertLocalClient('http://test:11434');
      const chunks: any[] = [];
      const gen = client.chat('hi');
      for await (const chunk of gen) {
        chunks.push(chunk);
      }
      expect(chunks).toHaveLength(1);
      expect(chunks[0].done).toBe(true);
    });

    it('handles HTTP error in stream', async () => {
      mockFetch.mockResolvedValue(mockResponse(500, {}));
      const client = new LitertLocalClient('http://test:11434');
      const gen = client.chat('hi');
      await expect(async () => {
        for await (const _ of gen) { /* noop */ }
      }).rejects.toThrow('HTTP 500');
    });
  });

  describe('listModels', () => {
    it('returns model list', async () => {
      mockFetch.mockResolvedValue(
        mockResponse(200, {
          models: [
            { name: 'gemma-4-12b' },
            { name: 'gemma-4-27b' },
          ],
        })
      );
      const client = new LitertLocalClient('http://test:11434');
      const models = await client.listModels();
      expect(models).toHaveLength(2);
      expect(models[0].id).toBe('gemma-4-12b');
    });
  });
});
