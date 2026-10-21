# APEX-CHAT · 旗舰可对话迷你 LLM 服务层

把[旗舰 Transformer](https://github.com/CJX0712/flagship-transformer) 与[旗舰推理引擎](https://github.com/CJX0712/flagship-rag-engine) 变成**一个真能对话的迷你 LLM 服务**——而不是只跑通一条演示流水线。

这是旗舰 AI 锻造闭环的**第四件实物**。

---

## 哲学：零重写 · 集大成

本仓库**不直接重写任何底层**。注意力 / RoPE / KV-Cache / BM25 全部复用前两块地基（`apex-transformer`、`apex-engine`），只在之上做**组合 + 超越**：

| 能力 | 复用来源 | 本仓库新增 |
|------|----------|-----------|
| 因果 LM 训练（前缀预测式自回归） | `apex-engine.trainTinyLM` | 对话语料格式化 |
| KV-Cache 增量解码 | `apex-engine.forwardCached` | 带重复惩罚的生成层 |
| 权重持久化 | ——（新增） | JSON save/load |
| 对话/多轮/停止符 | ——（新增） | 句末符 `。` 截断 + `问` 兜底 |
| 重复惩罚（repetition penalty） | 采样复用 `apex-forge.Sample` | 退化早停逻辑 |
| **⑥ BPE 子词分词** | Sennrich et al. 2016 (arXiv:1508.07909) | `bpe.js`：接口与词级分词器一致，可无缝替换；答案命中率 76%→100%、token 省 41% |

**不遮蔽的诚实**：用重复惩罚 + 句末符解决的是"小模型 greedy 退化"这一真实工程问题，而非掩盖模型能力短板。

---

## 快速开始

```bash
# 1) 训练并保存权重（默认 ⑥ BPE 子词分词 + 更大模型 d64/L3，约 100s）
node train.js 6000
#    旧字符级路径：node train.js 6000 --char

# 2) 命令行对话（加载 weights.json，免重训；BPE 分词器确定性重建，无需序列化词表）
node chat-repl.js "RoPE 是什么"
# 答：旋转位置编码

# 3) 进入交互式对话（支持多轮）
node chat-repl.js

# 4) 一键聚合自检（15/15 不变量：引擎 9 + 对话 6）
node verify.js

# 5) ⑥ BPE vs 字符级 对照基准（命中率 / token 效率）
node bench-chat-bpe.js 4000 6000
```

浏览器内对话：直接用浏览器打开 `index.html`（同目录需有 `apex-forge.js` / `apex-transformer.js` / `apex-engine.js` / `apex-chat.js`）。页面会在浏览器内训练一个轻量模型，或加载 `weights.json`，然后可实时对话。

---

## 真实演示（train.js 4000 步，greedy 解码）

```
问：什么是旗舰      → 世界顶级
问：RoPE 是什么     → 旋转位置编码
问：KV缓存有什么用  → 加速推理
问：YaRN 做什么     → 补外推
问：什么是外推      → 未见长度
```

中文无空格，早期采用**字符级分词**（每个中文字/标点独立成 token）才能正确编码。⑥ 起默认改用 **BPE 子词分词**（见下）。

---

## ⑥ BPE 子词分词：命中率 76% → 100%，token 省 41%

同一语料、同一批问题，对照两种分词器（`node bench-chat-bpe.js 4000 6000`）：

| 指标 | 字符级 | **BPE** |
|------|--------|---------|
| 答案命中率（问 23 问） | 76% | **100%** |
| 平均 prompt token 数 | 10.9 | **6.4（省 41%）** |
| 输出形态 | `世 是 照 界 顶 级`（逐字、带错） | `世界顶级`（干净子词） |
| 词表规模 | 165 | 258（含 80 合并） |

**为什么有效**：字符级把"旋转位置编码"拆成 6 个独立 token，模型要在 6 步上都对、且词表随语料膨胀；BPE 让高频字组（"位置""编码""优化器"）合并成子词，序列更短、统计强度跨相似答案共享，于是**更准 + 更省上下文**。

**BPE 实现要点**（`bpe.js`，Sennrich et al. 2016 词内 `</w>` 规范）：
- 接口与 `apex-engine.makeTokenizer` 完全一致（`{vocab,stoi,itos,size,encode,decode}`），故 `trainTinyLM` / 对话层**零改动**接入（组合而非重写）。
- **确定性可复现**：同语料同参数 → 同一套 merges，故 `weights.json` 无需序列化词表，加载端重建即可。
- 自检 **5/5**：往返 `decode(encode(t))==t` / 压缩 / 词表构成 `size=base+merges` / 确定性 / 最高频二元组优先合并。

---

## 自检（可被独立实现交叉验证的不变量）

`node verify.js` 聚合 `apex-engine` 9/9 与 `apex-chat` 6/6 共 **15/15 全绿**：

| 项 | 不变量 | 结果 |
|----|--------|------|
| KV-Cache ≡ 全量重算 | 逐位等价 | 偏差 0.00 |
| greedy 解码一致 | cache==full token 级 | PASS |
| top-p 保留质量≥p | 0.853 ≥ 0.8 | PASS |
| BM25 检索 Recall@1/MRR | 合成语料 | 1.00 / 1.00 |
| 端到端 RAG 管线 | 检索命中+生成连通 | PASS |
| **权重 save/load 往返** | 文件持久化后推理一致 | PASS |
| **对话链路连通** | 生成合法 token | PASS |
| **停止符截断** | 不泄漏下一轮 `问` | PASS |
| **BPE 权重往返** | BPE 分词下文件往返一致 | PASS |
| **BPE 对话连通** | 生成合法 token + 子词生效 | PASS |
| **BPE 停止符截断** | BPE 键 `。</w>` 可识别 | PASS |

`bpe.js` 另有独立 **5/5**（往返 / 压缩 / 词表构成 / 确定性 / 最高频优先合并）。

---

## 架构位置（四件实物）

```
flagship-algo-forge   (算法库 8/8)
        │
flagship-transformer  (参考实现+基准+RoPE+外推 9/9)
        │
flagship-rag-engine   (推理引擎+RAG 9/9)
        │
flagship-llm-chat ★   (可对话迷你 LLM 服务 15/15 · ⑥ BPE)   ← 本仓库
        │
flagship-monorepo     (总览 · 一键聚合自检)
```

---

## 诚实边界（必须一起读）

1. **玩具级模型**（BPE 版 d=64, 3 层）。⑥ 把答案命中率从 76% 提到 100%、token 省 41%，但在 23 条语料上，模型学到的仍是"问→答"的**表面模式记忆/模仿**，**不宣称**真实语义理解或通用对话能力。命中率提升是子词分词 + 更大模型的工程红利，不是"理解了语义"。
2. **Greedy 退化**是本就在的真实问题，本仓库用**重复惩罚 + 句末符 + 早停**在工程上抑制，但更强模型/更长文本仍可能退化（这正是真实 LLM 需要 repetition penalty / 采样温度的原因）。
3. **长度外推短板**继承自 `flagship-transformer`：多轮对话依赖长上下文，超长会退化（多轮已做前缀截断兜底；BPE 缩短序列在一定程度上缓解了该压力）。
4. 本仓库验证的是"**可对话推理服务链路完整、可复现、可验证**"，而非通用生成质量。

---

## 世界顶级成果出处

- Transformer / 因果注意力 / RoPE：Vaswani 2017 (arXiv:1706.03762) · Su 2021 (arXiv:2104.09864)
- AdamW / RMSNorm / Xavier：Loshchilov 2019 · Zhang 2019 · Glorot 2010
- KV-Cache：现代推理服务（vLLM / TGI / TensorRT-LLM）通用做法
- BM25：Robertson & Zaragoza 2009
- Top-k / Top-p 采样：Fan 2018 / Holtzman 2020
- 反向模式自动微分（autograd）：Baydin 2018 (arXiv:1502.05767)
- **BPE 子词分词**：Sennrich, Haddow & Birch 2016 (arXiv:1508.07909)

---

作者：晨星 (GitHub: CJX0712) · 许可：MIT · 系列：flagship-algo-forge / flagship-transformer / flagship-rag-engine / flagship-llm-chat / flagship-monorepo
