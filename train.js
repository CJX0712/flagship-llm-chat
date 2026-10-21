/*
 * train.js · 训练旗舰对话模型并保存权重
 * 用法: node train.js [steps] [--char]
 *   steps  默认 6000
 *   --char 用旧字符级分词（默认用 ⑥ BPE 子词分词 + 更大模型，命中率与 token 效率更优）
 * 产出: weights.json（含 mode/cfg/numMerges 元数据，加载端据此无歧义重建）
 */
const CHAT = require('./apex-chat.js');
const fs = require('fs');

const args = process.argv.slice(2);
const useChar = args.includes('--char');
const steps = parseInt(args.find((a) => /^\d+$/.test(a)) || '6000', 10);
const seed = 20261021;

let chat, meta;
const t0 = Date.now();
if (useChar) {
  console.log(`训练旗舰对话模型 · ${steps} 步 · 字符级因果 LM ...`);
  chat = CHAT.trainChat(CHAT.DEFAULT_CORPUS, { steps, seed });
  meta = { mode: 'char', cfg: chat.cfg };
} else {
  const numMerges = 80;
  console.log(`训练旗舰对话模型 · ${steps} 步 · ⑥ BPE 子词分词 + 更大模型(d64/L3) ...`);
  chat = CHAT.trainChatBPE(CHAT.DEFAULT_CORPUS, { steps, seed, numMerges });
  meta = { mode: 'bpe', cfg: chat.cfg, numMerges };
}
CHAT.saveWeights(chat.model, fs, './weights.json', meta);
console.log(`训练完成 (${((Date.now() - t0) / 1000).toFixed(1)}s) · 权重已存 weights.json · mode=${meta.mode} · 词表=${chat.tk.size}`);

console.log('\n=== 演示对话（greedy 解码）===');
const demo = ['什么是旗舰', 'RoPE 是什么', 'KV缓存有什么用', 'YaRN 做什么', '什么是外推'];
demo.forEach((q) => {
  const r = CHAT.respond(chat, q, { maxTokens: 16 });
  console.log(`问：${q}\n答：${r.text}\n`);
});
