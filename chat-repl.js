/*
 * chat-repl.js · 可对话迷你 LLM 命令行
 * 用法:
 *   node chat-repl.js train           训练并保存 weights.json
 *   node chat-repl.js "你的问题"       加载 weights.json 后单轮回答（批处理）
 *   node chat-repl.js                 进入交互式对话（readline）
 */
const CHAT = require('./apex-chat.js');
const fs = require('fs');
const readline = require('readline');

const arg = process.argv[2];

function loadChat() {
  const chat = CHAT.trainChat(CHAT.DEFAULT_CORPUS, { steps: 1 }); // 仅取结构
  if (!fs.existsSync('./weights.json')) {
    console.log('未找到 weights.json，请先运行: node chat-repl.js train');
    process.exit(1);
  }
  CHAT.loadWeights(chat.model, fs, './weights.json');
  return chat;
}

if (arg === 'train') {
  const steps = parseInt(process.argv[3] || '4000', 10);
  const chat = CHAT.trainChat(CHAT.DEFAULT_CORPUS, { steps });
  CHAT.saveWeights(chat.model, fs, './weights.json');
  console.log('已保存 weights.json · 运行 node chat-repl.js "你的问题" 开启对话');
} else if (arg) {
  const chat = loadChat();
  const r = CHAT.respond(chat, arg, { maxTokens: 20 });
  console.log('答：' + r.text);
} else {
  const chat = loadChat();
  console.log('旗舰迷你 LLM · 输入问题回车对话，Ctrl+C 退出（玩具级，仅演示链路）');
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
