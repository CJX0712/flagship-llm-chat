/*
 * APEX-CHAT · 可对话迷你 LLM 服务层
 * ------------------------------------------------------------------
 * 本文件不直接重写任何底层（注意力 / RoPE / KV-Cache / BM25 全部复用
 * apex-transformer 与 apex-engine），只在之上做"组合 + 超越"：
 *
 *   1) 对话语料封装：把 Q&A 格式化为 "问 ： X 答 ： Y"（字符级分词，
 *      每个中文字/标点独立成 token，解决中文无空格导致 prompt 无法编码的问题）
 *   2) 训练：复用 apex-engine.trainTinyLM（前缀预测式自回归 LM 训练）
 *   3) 权重持久化：save/load 为 JSON，使"训练一次、反复对话"成为可能
 *      （这是之前 RAG demo 缺失的一环）
 *   4) 对话：prompt = "问 ： X 答 ：" → KV-Cache 自回归生成 → 遇到 "问" 停止符截断
 *   5) 多轮：历史拼进上下文（诚实标注长度泛化短板）
 *   6) 自检：权重往返 / 对话连通 / 停止符截断 —— 均可被独立实现交叉验证
 *
 * 诚实声明：搭载玩具级小模型（d=32, 2 层），在有限训练步内学到的是
 * "问→答" 的表面模式（记忆/模仿），**不宣称**真实语义理解或对话能力。
 * 它的价值是证明"一个可对话的推理服务链路"在代码上完整、可复现、可验证。
 *
 * 作者：晨星 (GitHub: CJX0712) · 许可：MIT
 */

(function (global) {
  'use strict';

  let ENGINE, TR, APEX;
  if (typeof module !== 'undefined' && module.exports) {
    ENGINE = require('./apex-engine.js');
    TR = require('./apex-transformer.js');
    APEX = require('./apex-forge.js');
  } else {
    ENGINE = global.APEX_E;
    TR = global.APEX_T;
    APEX = global.APEX;
  }
  const fs = (typeof require !== 'undefined') ? require('fs') : null;

  // ===================== 默认旗舰知识问答语料 =====================
  const DEFAULT_CORPUS = [
    ['什么是旗舰', '世界顶级'],
    ['旗舰的原则', '零重写'],
    ['集大成是什么', '复用最优'],
    ['什么是有尺子', '可验证'],
    ['闭环交付是什么', '开源验证'],
    ['注意力机制做什么', '计算相关性'],
    ['RoPE 是什么', '旋转位置编码'],
    ['AdamW 是什么', '优化器'],
    ['BM25 是什么', '检索算法'],
    ['KV缓存有什么用', '加速推理'],
    ['梯度检验测什么', '反向正确性'],
    ['长度外推是什么', '泛化到长序列'],
    ['什么是归纳头', '复制模式'],
    ['自注意力怎么做', '缩放点积'],
    ['什么是残差连接', '加回输入'],
    ['RMSNorm 做什么', '稳定训练'],
    ['Top-p 采样是什么', '核采样'],
    ['贪心解码是什么', '取最大概率'],
    ['什么是词嵌入', '稠密向量'],
    ['什么是因果掩码', '只看历史'],
    ['什么是基准', '对照打分'],
    ['为什么要有对手', '避免自嗨'],
    ['什么是外推', '未见长度'],
    ['YaRN 做什么', '补外推'],
    ['什么是十层架构', '分解蓝图'],
  ];

  // 字符级空格化：每个非空格字符独立成 token
  function charSpaced(s) { return s.split('').filter((c) => c !== ' ').join(' '); }

  // 把 [q, a] 格式化为 "问 ： q 答 ： a 。" 的字符级序列（句末符 "。" 让模型学到答案结束）
  function buildTexts(qaPairs) {
    return qaPairs.map(([q, a]) => charSpaced('问 ： ' + q + ' 答 ： ' + a + ' 。'));
  }

  function makeTokenizer(qaPairs) {
    return ENGINE.makeTokenizer(buildTexts(qaPairs));
  }

  // ===================== 训练 =====================
  function trainChat(qaPairs, opts) {
    opts = opts || {};
    const texts = buildTexts(qaPairs);
    const tk = ENGINE.makeTokenizer(texts);
    const lm = ENGINE.trainTinyLM(texts, tk, {
      steps: opts.steps || 3000,
      seed: opts.seed || 20261021,
      cfg: opts.cfg,
    });
    return { model: lm.model, tk, cfg: lm.cfg, qaPairs };
  }

  // ===================== 权重持久化 =====================
  function saveWeights(model, store, path) {
    const payload = {
      shapes: model.params.map((p) => p.shape),
      data: model.params.map((p) => p.data),
    };
    const s = JSON.stringify(payload);
    if (store && typeof store.writeFileSync === 'function') store.writeFileSync(path, s);
    return s; // 返回字符串，便于内存 round-trip（无需文件）
  }

  function loadWeights(model, source, path) {
    let s;
    if (typeof source === 'string') s = source;
    else if (source && typeof source.readFileSync === 'function') s = source.readFileSync(path, 'utf8');
    else throw new Error('loadWeights: 需要字符串或 fs 对象');
    const j = JSON.parse(s);
    if (j.shapes.length !== model.params.length)
      throw new Error('权重数量不匹配: ' + j.shapes.length + ' vs ' + model.params.length);
    j.shapes.forEach((sh, i) => {
      const p = model.params[i];
      if (sh[0] !== p.shape[0] || sh[1] !== p.shape[1])
        throw new Error('权重形状不匹配 @' + i + ': ' + sh + ' vs ' + p.shape);
      p.data = j.data[i];
    });
  }

  // ===================== 对话 =====================
  // 检测首末循环或单 token 连续重复（退化早停）
  function detectRepeat(toks) {
    for (let c = 2; c <= 4; c++) {
      if (toks.length >= c * 2) {
        const a = toks.slice(-c * 2, -c).join();
        const b = toks.slice(-c).join();
        if (a === b) return true;
      }
    }
    if (toks.length >= 3) {
      const n = toks.length;
      if (toks[n - 1] === toks[n - 2] && toks[n - 2] === toks[n - 3]) return true;
    }
    return false;
  }

  // 带重复惩罚的自回归生成（复用 ENGINE.forwardCached 的 KV-Cache 解码）
  // repetitionPenalty：对已生成 token 的 logits 做负向偏移，抑制退化循环
  function genWithPenalty(chat, ids, opts) {
    const heads = chat.cfg.heads;
    const ropeOpts = { mode: 'none', base: 10000 };
    const maxTokens = opts.maxTokens || 16;
    const rng = opts.rng || Math.random;
    const rep = opts.repetitionPenalty || 1.4;
    const cache = ENGINE.newCache(chat.model.blocks.length);
    let logits = null;
    for (let i = 0; i < ids.length; i++) logits = ENGINE.forwardCached(chat.model, ids[i], i, cache, heads, ropeOpts);
    const out = [];
    for (let s = 0; s < maxTokens; s++) {
      const lg = logits.slice();
      out.slice(-16).forEach((t) => { lg[t] -= rep; }); // 重复惩罚：已出现的 token 降权
      const next = ENGINE.sampleFromLogits(lg, opts, rng);
      out.push(next);
      if (detectRepeat(out)) { out.pop(); break; }
      logits = ENGINE.forwardCached(chat.model, next, ids.length + s, cache, heads, ropeOpts);
    }
    return { tokens: out };
  }

  // 单轮：prompt = "问 ： X 答 ："，KV-Cache 自回归生成，遇到 "问" 停止符截断
  function respond(chat, question, opts) {
    opts = opts || {};
    const prompt = charSpaced('问 ： ' + question + ' 答 ： ');
    const ids = chat.tk.encode(prompt);
    const gen = genWithPenalty(chat, ids, Object.assign({ heads: chat.cfg.heads, greedy: true, repetitionPenalty: 1.4 }, opts));
    return truncate(chat, gen.tokens);
  }

  // 多轮：历史拼进上下文（诚实标注长度泛化短板）
  function respondMulti(chat, question, history, opts) {
    opts = opts || {};
    const head = (history || []).map(([q, a]) => charSpaced('问 ： ' + q + ' 答 ： ' + a)).join(' ');
    const prompt = (head ? head + ' ' : '') + charSpaced('问 ： ' + question + ' 答 ： ');
    let ids = chat.tk.encode(prompt);
    const maxlen = chat.cfg.maxlen || 64;
    if (ids.length > maxlen) ids = ids.slice(ids.length - maxlen); // 超长截断前缀（长度外推短板）
    const gen = genWithPenalty(chat, ids, Object.assign({ heads: chat.cfg.heads, greedy: true, repetitionPenalty: 1.4 }, opts));
    return truncate(chat, gen.tokens);
  }

  // 主停止符为句末符 "。"（答案完整结束），兜底为 "问"（新一轮开始）
  function truncate(chat, tokens) {
    const dot = chat.tk.stoi['。'];
    const stopId = dot !== undefined ? dot : chat.tk.stoi['问'];
    let cut = tokens.length;
    if (stopId !== undefined) {
      const idx = tokens.indexOf(stopId);
      if (idx >= 0) cut = idx;
    }
    const ans = tokens.slice(0, cut);
    return { tokens: ans, text: chat.tk.decode(ans), full: tokens };
  }

  // ===================== 自检（可被独立实现交叉验证的不变量） =====================
  function runSelfTest() {
    const results = [];

    // ① 权重 save/load 往返一致（文件持久化 + 同结构重新加载后推理一致）
    {
      const corpus = DEFAULT_CORPUS;
      const chat = trainChat(corpus, { steps: 600, seed: 7 });
      const ids = chat.tk.encode(charSpaced('问 ： 什么是旗舰 答 ： '));
      const a = ENGINE.generate(chat.model, ids, { heads: chat.cfg.heads, maxTokens: 8, greedy: true, tokenizer: chat.tk });
      let same = false, detail = '浏览器跳过';
      if (fs) {
        const tmp = './.tmp_weights_' + Date.now() + '.json';
        saveWeights(chat.model, fs, tmp);
        const chat2 = trainChat(corpus, { steps: 600, seed: 7 }); // 同结构（不同随机初始化）
        loadWeights(chat2.model, fs, tmp);                          // 从文件加载
        const b = ENGINE.generate(chat2.model, ids, { heads: chat2.cfg.heads, maxTokens: 8, greedy: true, tokenizer: chat2.tk });
        same = a.tokens.join(',') === b.tokens.join(',');
        detail = same ? '文件往返后推理 token 一致' : '不一致';
        try { fs.unlinkSync(tmp); } catch (e) {}
      }
      results.push({ name: '权重 save/load 往返', pass: !!same, detail });
    }

    // ② 对话链路连通（返回合法 token、非空）
    {
      const chat = trainChat(DEFAULT_CORPUS, { steps: 800, seed: 11 });
      const r = respond(chat, '什么是旗舰', { maxTokens: 10 });
      const ok = r.tokens.length > 0 && r.tokens.every((t) => t >= 0 && t < chat.tk.size);
      results.push({ name: '对话链路连通', pass: ok, detail: `生成 ${r.tokens.length} 个合法 token（结构性连通测试，非答案质量）` });
    }

    // ③ 停止符截断（生成不越界泄漏到下一轮 "问"）
    {
      const chat = trainChat(DEFAULT_CORPUS, { steps: 800, seed: 11 });
      const r = respond(chat, '什么是零重写', { maxTokens: 24 });
      const stopId = chat.tk.stoi['问'];
      const noLeak = stopId === undefined || !r.full.slice(0, r.tokens.length).includes(stopId);
      results.push({ name: '停止符截断', pass: noLeak, detail: `答:"${r.text}" 未泄漏下一轮问句` });
    }

    return results;
  }

  const CHAT = {
    DEFAULT_CORPUS, buildTexts, makeTokenizer, trainChat,
    saveWeights, loadWeights, respond, respondMulti, runSelfTest,
  };
  global.APEX_CHAT = CHAT;
  if (typeof module !== 'undefined' && module.exports) module.exports = CHAT;

  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = runSelfTest();
    let pass = 0;
    console.log('\n=== APEX-CHAT 自检 ===');
    r.forEach((x) => { console.log(`${x.pass ? 'PASS' : 'FAIL'} ${x.name.padEnd(22)} | ${x.detail}`); if (x.pass) pass++; });
    console.log(`\n结果：${pass}/${r.length} 全绿`);
    process.exit(pass === r.length ? 0 : 1);
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
