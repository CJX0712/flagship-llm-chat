/*
 * chat-repl.js · 可对话迷你 LLM 命令行
 * 用法:
 *   node chat-repl.js train [steps] [--char]   训练并保存 weights.json
 *   node chat-repl.js "你的问题"                加载 weights.json 后单轮回答（批处理）
 *   node chat-repl.js                          进入交互式对话（readline）
 *
 * 加载端依据 weights.json 内的 meta(mode/cfg/numMerges) 无歧义重建同构模型与分词器；
 * BPE 分词器确定性可复现，故无需序列化词表。
 */
const CHAT = require('./apex-chat.js');
const fs = require('fs');
const readline = require('readline');

const args = process.argv.slice(2);
const useChar = args.includes('--char');

function loadChat() {
  if (!fs.existsSync('./weights.json')) {
    console.log('未找到 weights.json，请先运行: node train.js');
    process.exit(1);
  }
  return CHAT.loadChatFrom(fs, './weights.json');
}

if (args[0] === 'train') {
  const steps = parseInt(args.find((a) => /^\d+$/.test(a)) || '6000', 10);
  const seed = 20261021;
  let chat, meta;
  const t0 = Date.now();
  if (useChar) {
    console.log(`训练 · ${steps} 步 · 字符级 ...`);
    chat = CHAT.trainChat(CHAT.DEFAULT_CORPUS, { steps, seed });
    meta = { mode: 'char', cfg: chat.cfg };
  } else {
    console.log(`训练 · ${steps} 步 · ⑥ BPE 子词 + 更大模型 ...`);
    chat = CHAT.trainChatBPE(CHAT.DEFAULT_CORPUS, { steps, seed, numMerges: 80 });
    meta = { mode: 'bpe', cfg: chat.cfg, numMerges: 80 };
  }
  CHAT.saveWeights(chat.model, fs, './weights.json', meta);
  console.log(`已保存 weights.json (${((Date.now() - t0) / 1000).toFixed(1)}s) · mode=${meta.mode} · 运行 node chat-repl.js "你的问题" 开启对话`);
} else if (args[0] && !args[0].startsWith('--')) {
  const chat = loadChat();
  const r = CHAT.respond(chat, args[0], { maxTokens: 20 });
  console.log('答：' + r.text);
} else {
  const chat = loadChat();
  console.log(`旗舰迷你 LLM · mode=${chat.bpe ? 'BPE' : '字符级'} · 输入问题回车对话，Ctrl+C 退出（玩具级，仅演示链路）`);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const history = [];
  const ask = () => {
    rl.question('你：', (q) => {
      if (!q.trim()) { rl.close(); return; }
      const r = CHAT.respondMulti(chat, q, history, { maxTokens: 20 });
      console.log('旗舰：' + r.text);
      history.push([q, r.text]);
      ask();
    });
  };
  ask();
}
