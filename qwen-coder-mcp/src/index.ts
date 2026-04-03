// qwen-coder-mcp - Expose Qwen3-Coder-Next as MCP tools (code_generation, code_review, debug_assistance)
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const QWEN_BASE_URL =
  process.env.QWEN_BASE_URL ?? "http://localhost:8080";
const QWEN_MODEL =
  process.env.QWEN_MODEL ?? "Qwen3-Coder-Next-Q5_K_M-00001-of-00002.gguf";
/** "openai" = llama.cpp /v1/completions; else Ollama /api/generate */
const QWEN_API = process.env.QWEN_API ?? "openai";
const TEMPERATURE = 0.2;
const MAX_TOKENS = 4096;
const REQUEST_TIMEOUT_MS = Number(process.env.QWEN_REQUEST_TIMEOUT_MS) || 120_000;

const server = new Server(
  { name: "qwen-coder-worker", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "code_generation",
      description:
        "Generate or refactor code using Qwen3-Coder-Next specialist model",
      inputSchema: {
        type: "object",
        properties: {
          task: {
            type: "string",
            description: "Coding task description",
          },
          language: {
            type: "string",
            description: "Programming language (optional)",
          },
          context: {
            type: "string",
            description: "Additional context or existing code (optional)",
          },
        },
        required: ["task"],
      },
    },
    {
      name: "code_review",
      description:
        "Review code for bugs, performance, security issues",
      inputSchema: {
        type: "object",
        properties: {
          code: { type: "string", description: "Code to review" },
          focus: {
            type: "string",
            description:
              "Specific focus area (security/performance/bugs/style)",
          },
        },
        required: ["code"],
      },
    },
    {
      name: "debug_assistance",
      description: "Help debug code issues or errors",
      inputSchema: {
        type: "object",
        properties: {
          code: {
            type: "string",
            description: "Problematic code",
          },
          error: {
            type: "string",
            description: "Error message or unexpected behaviour",
          },
        },
        required: ["code", "error"],
      },
    },
  ],
}));

async function callQwen(prompt: string): Promise<string> {
  const base = QWEN_BASE_URL.replace(/\/$/, "");
  const isOpenAI = QWEN_API === "openai";

  const url = isOpenAI
    ? `${base}/v1/completions`
    : `${base}/api/generate`;

  const body = isOpenAI
    ? {
        model: QWEN_MODEL,
        prompt,
        max_tokens: MAX_TOKENS,
        temperature: TEMPERATURE,
        top_p: 0.9,
      }
    : {
        model: QWEN_MODEL,
        prompt,
        stream: false,
        options: {
          temperature: TEMPERATURE,
          num_predict: MAX_TOKENS,
          top_p: 0.9,
        },
      };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Qwen API error ${response.status}: ${text}`);
  }

  const data = (await response.json()) as Record<string, unknown>;

  if (isOpenAI) {
    const choices = data.choices as Array<{ text?: string }> | undefined;
    const text = choices?.[0]?.text;
    if (typeof text === "string") return text;
  } else {
    const res = data.response;
    if (typeof res === "string") return res;
  }
  return JSON.stringify(data);
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, string | undefined>;

  switch (name) {
    case "code_generation": {
      if (typeof a.task !== "string" || !a.task.trim()) {
        throw new Error("code_generation requires non-empty task");
      }
      break;
    }
    case "code_review": {
      if (typeof a.code !== "string" || !a.code.trim()) {
        throw new Error("code_review requires non-empty code");
      }
      break;
    }
    case "debug_assistance": {
      if (typeof a.code !== "string" || !a.code.trim()) {
        throw new Error("debug_assistance requires non-empty code");
      }
      if (typeof a.error !== "string" || !a.error.trim()) {
        throw new Error("debug_assistance requires non-empty error");
      }
      break;
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }

  let prompt = "";

  switch (name) {
    case "code_generation": {
      prompt = `Task: ${a.task}\n`;
      if (a.language) prompt += `Language: ${a.language}\n`;
      if (a.context) prompt += `Context:\n${a.context}\n`;
      prompt += "\nGenerate the code:";
      break;
    }
    case "code_review": {
      const focus = a.focus
        ? ` focusing on ${a.focus}`
        : "";
      prompt = `Review this code${focus}:\n\n${a.code}\n\nProvide detailed review:`;
      break;
    }
    case "debug_assistance": {
      prompt = `Code:\n${a.code}\n\nError/Issue:\n${a.error}\n\nHelp debug this:`;
      break;
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }

  const responseText = await callQwen(prompt);

  return {
    content: [{ type: "text", text: responseText }],
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
