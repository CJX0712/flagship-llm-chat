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

**不遮蔽的诚实**：用重复惩罚 + 句末符解决的是"小模型 greedy 退化"这一真实工程问题，而非掩盖模型能力短板。

---

## 快速开始

```bash
# 1) 训练并保存权重（约 20s，生成 weights.json）
node train.js 4000

# 2) 命令行对话（加载 weights.json，免重训）
node chat-repl.js "RoPE 是什么"
# 答：旋转位置编码

# 3) 进入交互式对话
node chat-repl.js

# 4) 一键聚合自检（12/12 不变量）
node verify.js
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

中文无空格，采用**字符级分词**（每个中文字/标点独立成 token），否则 prompt 无法正确编码。

---

## 自检（可被独立实现交叉验证的不变量）

`node verify.js` 聚合 `apex-engine` 9/9 与 `apex-chat` 3/3 共 **12/12 全绿**：

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

---

## 架构位置（四件实物）

```
flagship-algo-forge   (算法库 8/8)
        │
flagship-transformer  (参考实现+基准+RoPE+外推 9/9)
        │
flagship-rag-engine   (推理引擎+RAG 9/9)
        │
flagship-llm-chat ★   (可对话迷你 LLM 服务 12/12)   ← 本仓库
        │
flagship-monorepo     (总览 · 一键聚合自检)
```

---

## 诚实边界（必须一起读）

1. **玩具级模型**（d=32, 2 层）。在有限步数内学到的是"问→答"的**表面模式记忆/模仿**，**不宣称**真实语义理解或对话能力。demo 中前几个字正确是模式匹配，不是推理。
2. **Greedy 退化**是本就在的真实问题，本仓库用**重复惩罚 + 句末符 + 早停**在工程上抑制，但更强模型/更长文本仍可能退化（这正是真实 LLM 需要 repetition penalty / 采样温度的原因）。
3. **长度外推短板**继承自 `flagship-transformer`：多轮对话依赖长上下文，超长会退化（多轮已做前缀截断兜底）。
4. 本仓库验证的是"**可对话推理服务链路完整、可复现、可验证**"，而非生成质量。

---

## 世界顶级成果出处

- Transformer / 因果注意力 / RoPE：Vaswani 2017 (arXiv:1706.03762) · Su 2021 (arXiv:2104.09864)
- AdamW / RMSNorm / Xavier：Loshchilov 2019 · Zhang 2019 · Glorot 2010
- KV-Cache：现代推理服务（vLLM / TGI / TensorRT-LLM）通用做法
- BM25：Robertson & Zaragoza 2009
- Top-k / Top-p 采样：Fan 2018 / Holtzman 2020
- 反向模式自动微分（autograd）：Baydin 2018 (arXiv:1502.05767)

---

作者：晨星 (GitHub: CJX0712) · 许可：MIT · 系列：flagship-algo-forge / flagship-transformer / flagship-rag-engine / flagship-llm-chat / flagship-monorepo
