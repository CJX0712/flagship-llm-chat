/*
 * APEX-BPE · 子词分词器（Byte Pair Encoding，词内 </w> 规范）
 * ------------------------------------------------------------------
 * 出处：Sennrich, Haddow & Birch 2016, "Neural Machine Translation of
 *       Rare Words with Subword Units" (arXiv:1508.07909)。
 *       —— BPE 子词的原始论文；GPT-2 用其 byte-level 变体，此处实现
 *       经典的"词内字符级 + 词尾 </w>"标准 BPE（词间不跨词合并）。
 *
 * 为什么需要它（相对 apex-engine 词级分词器的"组合 + 超越"）：
 *   引擎的分词器是【词级】——中文无空格时靠 charSpaced 把每字拆成独立 token，
 *   于是"旋转位置编码"这类答案占 6 个 token，词表随语料线性膨胀、长答案序列
 *   超长。BPE 让高频字组（如"位置""编码""优化器"）合并成子词 token，
 *   序列更短、词表更省泛化更好，且接口与词级分词器完全一致，可无缝替换。
 *
 * 接口契约（与 apex-engine.makeTokenizer 相同，故 trainTinyLM / 对话层零改动）：
 *   { vocab, stoi, itos, size, encode(s)->ids[], decode(ids)->text }
 *
 * 可被独立实现交叉验证的不变量（见文件底部自检）：
 *   ① 往返：decode(encode(t)) === 空白归一化后的 t
 *   ② 压缩：BPE 后 token 数 < 原字符数（numMerges>0 时）
 *   ③ 词表构成：size === 唯一基础符号数 + merges 数
 *   ④ 确定性：同语料两次训练得到完全相同的 merges
 *   ⑤ 正确性：在受控语料上，最高频二元组被优先合并
 *
 * 作者：晨星 (GitHub: CJX0712) · 许可：MIT
 */

(function (global) {
  'use strict';

  const END = '</w>'; // 词尾标记（Sennrich 2016 规范）：使解码可还原词边界

  // 词 → 初始符号序列：每个字符独立，末字符附词尾标记
  //   例："位置编码" → ['位','置','编','码</w>']
  function wordToSymbols(word) {
    const s = word.split('');
    if (s.length === 0) return s;
    s[s.length - 1] = s[s.length - 1] + END;
    return s;
  }

  // 在一条符号序列里，把所有相邻 (a,b) 就地合并为 merged（左到右，不重叠）
  function applyMerge(syms, a, b, merged) {
    const out = [];
    let i = 0;
    while (i < syms.length) {
      if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
        out.push(merged);
        i += 2;
      } else {
        out.push(syms[i]);
        i++;
      }
    }
    return out;
  }

  // 训练 BPE：迭代合并最高频相邻二元组，共 numMerges 轮
  // 并列时取字典序最小者（保证确定性 / 可复现）
  function trainBPE(texts, numMerges) {
    // 词频表（空白分词）
    const wf = new Map();
    (texts || []).forEach((t) => {
      String(t).split(/\s+/).filter(Boolean).forEach((w) => wf.set(w, (wf.get(w) || 0) + 1));
    });
    let words = Array.from(wf, ([w, freq]) => ({ syms: wordToSymbols(w), freq }));

    const merges = [];
    for (let m = 0; m < numMerges; m++) {
      // 统计所有相邻二元组（按词频加权）
      const pf = new Map();
      for (const { syms, freq } of words) {
        for (let i = 0; i < syms.length - 1; i++) {
          const k = syms[i] + '\t' + syms[i + 1];
          pf.set(k, (pf.get(k) || 0) + freq);
        }
      }
      if (pf.size === 0) break;
      let best = null, bestF = -1;
      for (const [k, f] of pf) {
        if (f > bestF || (f === bestF && k < best)) { best = k; bestF = f; }
      }
      const ai = best.indexOf('\t');
      const a = best.slice(0, ai), b = best.slice(ai + 1), merged = a + b;
      merges.push({ pair: [a, b], merged });
      for (const wd of words) wd.syms = applyMerge(wd.syms, a, b, merged);
    }
    return merges;
  }

  // 构建分词器：基础符号（字符 + 词尾变体）+ 全部合并产物
  function makeBPETokenizer(texts, numMerges) {
    numMerges = (numMerges === undefined) ? 200 : numMerges;
    texts = texts || [];
    const merges = trainBPE(texts, numMerges);

    const vocabSet = new Set();
    const baseSymbols = new Set();
    (function collect() {
      const wf = new Set();
      texts.forEach((t) => String(t).split(/\s+/).filter(Boolean).forEach((w) => wf.add(w)));
      wf.forEach((w) => wordToSymbols(w).forEach((s) => { vocabSet.add(s); baseSymbols.add(s); }));
    })();
    merges.forEach((mg) => vocabSet.add(mg.merged));

    const vocab = Array.from(vocabSet).sort();
    const stoi = {}, itos = {};
    vocab.forEach((w, i) => { stoi[w] = i; itos[i] = w; });

    function encodeWord(word) {
      let syms = wordToSymbols(word);
      for (const mg of merges) syms = applyMerge(syms, mg.pair[0], mg.pair[1], mg.merged);
      return syms;
    }
    return {
      vocab, stoi, itos, size: vocab.length, merges, baseCount: baseSymbols.size,
      encode(s) {
        return String(s).split(/\s+/).filter(Boolean)
          .reduce((acc, w) => acc.concat(encodeWord(w).map((x) => stoi[x])), [])
          .filter((i) => i !== undefined);
      },
      decode(ids) {
        let out = '';
        (ids || []).forEach((id) => {
          const tok = itos[id];
          if (tok === undefined) { out += '?'; return; }
          if (tok.length > END.length && tok.slice(-END.length) === END) out += tok.slice(0, -END.length) + ' ';
          else out += tok;
        });
        return out.trim();
      },
    };
  }

  const BPE = { makeBPETokenizer, trainBPE, wordToSymbols, applyMerge, END };
  global.APEX_BPE = BPE;
  if (typeof module !== 'undefined' && module.exports) module.exports = BPE;

  // ===================== 自检（纯算法，毫秒级，可被独立实现交叉验证） =====================
  function runSelfTest() {
    const results = [];
    const push = (name, pass, detail) => results.push({ name, pass: !!pass, detail });

    // 语料：词级（空白分隔）中文 Q&A —— 与 apex-chat 的 buildTextsBPE 输出同构
    const corpus = [
      '问 ： 什么是旗舰 答 ： 世界顶级 。',
      '问 ： RoPE 是什么 答 ： 旋转位置编码 。',
      '问 ： 优化器用什么 答 ： 优化器 AdamW 。',
      '问 ： 位置编码 答 ： 旋转位置编码 。',
      '问 ： 什么是编码 答 ： 位置编码 旋转位置编码 。',
    ];

    // ① 往返：decode(encode(t)) === 空白归一化后的 t
    {
      const tk = makeBPETokenizer(corpus, 120);
      const norm = (s) => s.split(/\s+/).filter(Boolean).join(' ');
      let ok = true, bad = '';
      corpus.forEach((t) => { const r = tk.decode(tk.encode(t)); if (r !== norm(t)) { ok = false; bad = t + ' -> ' + r; } });
      push('BPE 往返 decode(encode(t))==t', ok, ok ? '5/5 语料往返一致' : '失败: ' + bad);
    }

    // ② 压缩：BPE 后 token 数 < 原字符数（numMerges>0）
    {
      const tk = makeBPETokenizer(corpus, 120);
      const toks = tk.encode(corpus.join(' '));
      const chars = corpus.join(' ').split(/\s+/).filter(Boolean).join('').split('').length;
      push('BPE 压缩 token<字符', toks.length < chars, `token=${toks.length} < 字符=${chars}（压缩 ${(100 * (1 - toks.length / chars)).toFixed(0)}%）`);
    }

    // ③ 词表构成：size === 唯一基础符号数 + merges 数
    {
      const nm = 120;
      const tk = makeBPETokenizer(corpus, nm);
      const expect = tk.baseCount + tk.merges.length;
      push('BPE 词表构成 size=base+merges', tk.size === expect, `size=${tk.size} == base=${tk.baseCount}+merges=${tk.merges.length}`);
    }

    // ④ 确定性：同语料两次训练 → merges 完全一致
    {
      const a = makeBPETokenizer(corpus, 120).merges.map((m) => m.merged).join('|');
      const b = makeBPETokenizer(corpus, 120).merges.map((m) => m.merged).join('|');
      push('BPE 确定性（可复现）', a === b, a === b ? '两次训练 merges 完全一致' : '不一致');
    }

    // ⑤ 正确性：受控语料上，最高频二元组被优先合并
    {
      const c = ['甲甲甲甲甲甲', '乙丙', '乙丙']; // 词内二元组 "甲甲" 频次最高(=5)
      const tk = makeBPETokenizer(c, 1);
      push('BPE 优先合并最高频二元组', tk.merges[0] && tk.merges[0].merged === '甲甲', `top1 merge=${tk.merges[0] ? tk.merges[0].merged : '(空)'}（期望 甲甲）`);
    }

    return results;
  }

  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = runSelfTest();
    let pass = 0;
    console.log('\n=== APEX-BPE 自检 ===');
    r.forEach((x) => { console.log(`${x.pass ? 'PASS' : 'FAIL'} ${x.name.padEnd(30)} | ${x.detail}`); if (x.pass) pass++; });
    console.log(`\n结果：${pass}/${r.length} 全绿`);
    process.exit(pass === r.length ? 0 : 1);
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
