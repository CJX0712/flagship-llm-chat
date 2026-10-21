/*
 * bench-chat-bpe.js · ⑥ BPE vs 字符级 对话对照基准
 * ------------------------------------------------------------------
 * 同一语料、同一批问题，对比两种分词器驱动下的旗舰迷你 LLM：
 *   · 字符级（trainChat）：老路径，每字一 token
 *   · BPE（trainChatBPE）：子词分词 + 更大模型
 *
 * 诚实三把尺子：
 *   ① 答案命中率：问训练集内 23 问，答文是否覆盖标准答案（去空白后包含）
 *   ② token 效率：同一 prompt 的 token 数 BPE vs 字符级（越短越省上下文）
 *   ③ 词表规模：BPE 词表 vs 字符级词表
 *
 * 用法: node bench-chat-bpe.js [charSteps] [bpeSteps]
 */
const CHAT = require('./apex-chat.js');
const ENGINE = require('./apex-engine.js');

const charSteps = parseInt(process.argv[2] || '4000', 10);
const bpeSteps = parseInt(process.argv[3] || '6000', 10);
const corpus = CHAT.DEFAULT_CORPUS;
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

// 命中判定：生成文本去空白后包含标准答案去空白（子串命中即算对）
const norm = (s) => (s || '').replace(/\s+/g, '');
function hit(text, ans) { const t = norm(text), a = norm(ans); return a.length > 0 && t.indexOf(a) >= 0; }

function evalChat(chat, label) {
  let correct = 0;
  const rows = [];
  corpus.forEach(([q, a]) => {
    const r = CHAT.respond(chat, q, { maxTokens: 16 });
    const ok = hit(r.text, a);
    if (ok) correct++;
    rows.push({ q, a, got: r.text, ok });
  });
  return { label, acc: correct / corpus.length, rows };
}

console.log(`\n=== ⑥ BPE vs 字符级 对话对照 (charSteps=${charSteps}, bpeSteps=${bpeSteps}) ===\n`);

console.log(`[${el()}] 训练字符级 ...`);
const chatChar = CHAT.trainChat(corpus, { steps: charSteps, seed: 20261021 });
console.log(`[${el()}] 训练 BPE（更大模型）...`);
const chatBPE = CHAT.trainChatBPE(corpus, { steps: bpeSteps, seed: 20261021 });

const rc = evalChat(chatChar, '字符级');
const rb = evalChat(chatBPE, 'BPE');

// token 效率：对全部 prompt 求平均 token 数
let charTok = 0, bpeTok = 0;
corpus.forEach(([q]) => {
  charTok += chatChar.tk.encode(chatChar.fmtQ(q)).length;
  bpeTok += chatBPE.tk.encode(chatBPE.fmtQ(q)).length;
});
charTok /= corpus.length; bpeTok /= corpus.length;

console.log('\n=== 总览 ===');
console.log(`答案命中率   字符级=${(rc.acc * 100).toFixed(0)}%  BPE=${(rb.acc * 100).toFixed(0)}%`);
console.log(`平均prompt token  字符级=${charTok.toFixed(1)}  BPE=${bpeTok.toFixed(1)}  (省 ${(100 * (1 - bpeTok / charTok)).toFixed(0)}%)`);
console.log(`词表规模      字符级=${chatChar.tk.size}  BPE=${chatBPE.tk.size}（含 ${chatBPE.tk.merges.length} 合并）`);

console.log('\n=== 并排演示（前 6 问）===');
for (let i = 0; i < 6; i++) {
  const [q, a] = corpus[i];
  console.log(`问：${q}   [标准答案：${a}]`);
  console.log(`  字符级：${rc.rows[i].got}`);
  console.log(`  BPE   ：${rb.rows[i].got}`);
}

console.log(`\n耗时 ${el()}`);
process.exit(0);
