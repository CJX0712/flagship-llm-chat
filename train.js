/*
 * train.js · 训练旗舰对话模型并保存权重
 * 用法: node train.js [steps]
 * 产出: weights.json（可被 load 反复用于对话，无需重训）
 */
const CHAT = require('./apex-chat.js');
const fs = require('fs');

const steps = parseInt(process.argv[2] || '4000', 10);
console.log(`训练旗舰对话模型 · ${steps} 步 · 字符级因果 LM ...`);

const t0 = Date.now();
const chat = CHAT.trainChat(CHAT.DEFAULT_CORPUS, { steps, seed: 20261021 });
CHAT.saveWeights(chat.model, fs, './weights.json');
console.log(`训练完成 (${((Date.now() - t0) / 1000).toFixed(1)}s) · 权重已存 weights.json`);

console.log('\n=== 演示对话（greedy 解码）===');
const demo = ['什么是旗舰', 'RoPE 是什么', 'KV缓存有什么用', 'YaRN 做什么', '什么是外推'];
demo.forEach((q) => {
  const r = CHAT.respond(chat, q, { maxTokens: 16 });
  console.log(`问：${q}\n答：${r.text}\n`);
});
