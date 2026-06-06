#!/usr/bin/env node
/**
 * MCP 工具调用脚本 - 命令行接口
 * 用法: node mcp_call_tool.js <tool_name> <json_args>
 */

// 兼容不同版本的 eventsource 导出方式
import * as eventsourceModule from 'eventsource';
const EventSource = eventsourceModule.default || eventsourceModule.EventSource || eventsourceModule;
// 使用 undici（5.x 兼容 Node 16）提供 web ReadableStream 兼容的 fetch
let fetchImpl, HeadersImpl, RequestImpl, ResponseImpl, ReadableStreamImpl, TransformStreamImpl;
try {
  const undici = await import('undici');
  fetchImpl = undici.fetch;
  HeadersImpl = undici.Headers;
  RequestImpl = undici.Request;
  ResponseImpl = undici.Response;

  // Node 16 提供 stream/web，但为稳妥仍动态导入
  const webStreams = await import('stream/web');
  ReadableStreamImpl = webStreams.ReadableStream;
  TransformStreamImpl = webStreams.TransformStream;
} catch (err) {
  console.error('❌ 未找到 undici，请先安装（npm install undici@5 --registry=https://registry.npmmirror.com）');
  process.exit(1);
}
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';

// 为 Node.js 环境提供浏览器 API（兼容 Node.js 16+）
global.EventSource = EventSource;
global.fetch = global.fetch || fetchImpl;
global.Headers = global.Headers || HeadersImpl;
global.Request = global.Request || RequestImpl;
global.Response = global.Response || ResponseImpl;
global.ReadableStream = global.ReadableStream || ReadableStreamImpl;
global.TransformStream = global.TransformStream || TransformStreamImpl;

async function callTool(toolName, args) {
  const serverUrl = 'https://api.example.com/sse';

  try {
    // 将 fetch 显式传给 SSEClientTransport，避免依赖默认 fetch
    const transport = new SSEClientTransport(new URL(serverUrl), {
      fetch: fetchImpl,
    });
    const client = new Client(
      {
        name: 'mcp-cli-client',
        version: '1.0.0',
      },
      {
        capabilities: {},
      }
    );

    await client.connect(transport);

    const result = await client.callTool({
      name: toolName,
      arguments: args
    });

    await client.close();

    // 输出 JSON 结果到 stdout
    console.log(JSON.stringify(result, null, 0));
  } catch (error) {
    // 错误输出到 stderr
    console.error(
      JSON.stringify({
        isError: true,
        error: error.message,
        stack: error.stack,
      })
    );
    process.exit(1);
  }
}

// 解析命令行参数
if (process.argv.length < 4) {
  console.error('用法: node mcp_call_tool.js <tool_name> <json_args>');
  console.error('示例: node mcp_call_tool.js meituan_search_v2 \'{"location":"杭州西湖","mode":"1","searchScene":"0","queries":["火锅"],"isComplex":false}\'');
  process.exit(1);
}

const toolName = process.argv[2];
const argsJson = process.argv[3];

let args;
try {
  args = JSON.parse(argsJson);
} catch (error) {
  console.error('参数解析错误:', error.message);
  process.exit(1);
}

// 执行调用
callTool(toolName, args);

