---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url("./evidence-lab.css");
---

<!-- _paginate: false -->
<!-- _class: slide--title -->

<div class="slide-hero">
  <div class="eyebrow">QWEN MEETUP TOKYO · LOCAL LLM EXPERIMENTS</div>
  <h1>
    When Does a Local Qwen
    <br />
    Start to Break?
  </h1>
  <div class="rule"></div>
  <p class="lead">量子化して、長い文脈を入れて、それでも使えるのか？</p>
  <p class="subtle">
    Qwen3.8-27B / llama.cpp / Apple Silicon 64GB
    <br />
    Quantization × Context × Evaluation × Agent History
  </p>
</div>

---

## WHOAMI

<img width="100%" src="./img/linkedin.png">

---

## Qwen Model とは？

<img width="100%" src="./img/qwen-huggingface.png">

<p class="source">
  Sources: Qwen official blog / Qwen3.8-27B official Hugging Face model repository (checked 2026-09-03)
</p>

<!-- ---

## Qwen Model とは？

<div class="layout--split">
  <div class="card">
    <h3>Qwen family</h3>
    <ul>
      <li>Qwen Teamが公開する<strong>open-weight LLM family</strong></li>
      <li>reasoning / coding / multilingual / multimodalまで幅広く展開</li>
      <li>ローカル実行の選択肢が多く、実験しやすい</li>
    </ul>
    <div class="callout callout--info u-mt-24">
      APIで使うモデルだけでなく、<strong>自分のマシンで挙動を観察できるモデル</strong>。
    </div>
  </div>
  <div>
    <div class="stat">
      27B
      <small>
        <br />
        今回の対象モデル
        <br />
        Qwen3.8-27B
      </small>
    </div>
    <div class="layout--metrics u-mt-26">
      <div class="metric-card">
        <span class="metric-card__value">Apache-2.0</span>
        <span class="metric-card__label">official model license</span>
      </div>
      <div class="metric-card">
        <span class="metric-card__value">55.6 GB</span>
        <span class="metric-card__label">official HF repository footprint</span>
      </div>
    </div>
  </div>
</div>

<p class="source">
  Sources: Qwen official blog / Qwen3.8-27B official Hugging Face model repository (checked 2026-09-03)
</p> -->

---

## なぜローカルで動かす？ そして、なぜ量子化する？

<div class="layout--split">
  <div class="card">
    <h3>Local LLM の魅力</h3>
    <ul>
      <li>データを外に出さずに試せる</li>
      <li>APIコストやrate limitを気にせず反復できる</li>
      <li>モデル内部の条件を固定して、実験しやすい</li>
    </ul>
    <div class="quote u-mt-26">でも、27Bをそのまま載せるには重い。</div>
  </div>
  <div class="card">
    <h3>Quantization</h3>
    <div class="layout--metrics u-mt-16">
      <div class="metric-card">
        <span class="metric-card__value">Q8</span>
        <span class="metric-card__label">29.05 GB</span>
      </div>
      <div class="metric-card">
        <span class="metric-card__value">Q6</span>
        <span class="metric-card__label">22.43 GB</span>
      </div>
      <div class="metric-card">
        <span class="metric-card__value">Q4</span>
        <span class="metric-card__label">16.81 GB</span>
      </div>
    </div>
    <div class="callout callout--warning u-mt-24">
      重みを少ないbitで表現して軽くする。<br />
      では、<strong>軽くした代わりに何を失うのか？</strong>
    </div>
  </div>
</div>

<p class="source">Measured artifact size in exp_002 · Q8_0 / Q6_K / Q4_K_M</p>

---

## 　量子化とは

<div class="quantization-explainer">
  <div class="quantization-main">
    <div class="quantization-step">
      <div class="quantization-label">元の重み</div>
      <div class="value-rack value-rack--fine">
        <span>−0.93</span>
        <span>−0.62</span>
        <span>−0.11</span>
        <span>＋0.37</span>
        <span>＋0.71</span>
        <span>＋0.94</span>
      </div>
      <p>細かい値を持つ</p>
    </div>
    <div class="quantization-arrow">→</div>
    <div class="quantization-step quantization-step--accent">
      <div class="quantization-label">量子化</div>
      <div class="quantization-round">丸める</div>
      <p>少ないbitで表す</p>
    </div>
    <div class="quantization-arrow">→</div>
    <div class="quantization-step">
      <div class="quantization-label">量子化後</div>
      <div class="value-rack value-rack--coarse">
        <span>−1.0</span>
        <span>−0.5</span>
        <span>0</span>
        <span>＋0.5</span>
        <span>＋1.0</span>
      </div>
      <p>使う値を減らす</p>
    </div>
  </div>
  <div class="quantization-meaning">
    <div class="quantization-conclusion">
      <span class="quantization-conclusion__lead">モデルが軽くなる</span>
      <span class="quantization-conclusion__sub">容量・メモリを節約</span>
    </div>
    <div class="quantization-cost">
      <span>トレードオフ</span>
      <strong>少し誤差が入る</strong>
    </div>
  </div>
</div>

<p class="source">概念図：重みの表現を粗くして、モデルを軽量化する</p>

---

<!-- 入力 \(X\) と重み \(W\) のうち、普通の値は INT8 で行列積
精度を壊しやすい outlier（黄色）だけ FP16 のまま計算
最後に両者を足して FP16 の出力を作る -->

<img width="90%" src="./img/quantization.png">

<p class="source">
  Sources: Qwen official blog / Qwen3.8-27B official Hugging Face model repository (checked 2026-09-03)
</p>

---

## 今回、知りたかったこと

<div class="layout--three">
  <div class="axis-card">
    <h3>RQ1 · Context</h3>
    <p>
      長い文脈を入れたとき、
      <br />
      情報を最後まで正しく使えるか？
    </p>
  </div>
  <div class="axis-card axis-card--teal">
    <h3>RQ2 · Quantization</h3>
    <p>
      Q8 → Q4で、
      <br />
      回答能力はどこまで残るか？
    </p>
  </div>
  <div class="axis-card axis-card--amber">
    <h3>RQ3 · Interaction</h3>
    <p>
      長い文脈 × 強い量子化で、
      <br />
      劣化は増幅するか？
    </p>
  </div>
</div>

<div class="callout callout--info u-mt-34">
  さらに、agent historyまで伸ばしたとき一度見つけた事実を最後まで使えるかも確認した。
</div>

---

## “Break” をどう測ったか

<div class="layout--split">
  <div>
    <div class="card">
      <h3>Task</h3>
      <pre><code>Context
┌────────────────────────┐
│ distractor / noise     │
│                        │
│  KEY = ZX-4817         │
│                        │
│ distractor / noise     │
└────────────────────────┘

Q: What is the key?</code></pre>
    </div>
  </div>
  <div>
    <div class="layout--three">
      <div class="axis-card">
        <h3>literal</h3>
        <p>そのまま拾う</p>
      </div>
      <div class="axis-card axis-card--teal">
        <h3>semantic</h3>
        <p>意味を理解して拾う</p>
      </div>
      <div class="axis-card axis-card--amber">
        <h3>multi-hop</h3>
        <p>複数情報をつなぐ</p>
      </div>
    </div>
    <div class="callout callout--warning u-mt-24">
      exact matchだけでなく、<strong>answer-bearing / format-valid / end-to-end</strong>を分離して採点。
    </div>
  </div>
</div>

<p class="source">calibrated.v1 · independent tasks · greedy decoding · p50 evidence position unless noted</p>

---

## Context Window は使える長さではない

<div class="layout--split">
  <div>
    <div class="stat">
      75s → 339.5s
      <small>
        <br />
        median stream-derived TTFT
        <br />
        8K → 32K
      </small>
    </div>
    <div class="callout callout--info u-mt-24">
      8K / 32Kでは、answer-bearing correctnessは<strong>全セル 10/10</strong>。
    </div>
    <div class="callout callout--warning u-mt-18">
      しかし正答したことと指定形式まで満たしたことは別だった。
    </div>
  </div>
  <div>
    <table class="data-table">
      <thead>
        <tr>
          <th>target</th>
          <th>result</th>
          <th>observed boundary</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>64K</td>
          <td><span class="badge badge--measured">3/3 complete</span></td>
          <td>TTFT 783–786s</td>
        </tr>
        <tr>
          <td>128K</td>
          <td><span class="badge badge--pilot">3/3 timeout</span></td>
          <td>900s · RSS ≈ 37.6GB</td>
        </tr>
        <tr>
          <td>262K</td>
          <td><span class="badge badge--pilot">3/3 timeout</span></td>
          <td>900s · RSS ≈ 46.2GB</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="quote u-mt-30">入るより先に、待てないが実用上の限界になった。</div>
<p class="source">Q8_0 · baseline 60/60 + feasibility probe 9 attempts · 900s timeout</p>

---

## Feasibility Boundary

<img width="100%" src="../experiments/exp_001-context_measurement/results/figures/presentation-baseline-by-task.png">

<p class="source">Bounded feasibility result — not a model hard limit · 900s timeout per attempt</p>

---

## Q4で −42.1%。では、賢さも42%落ちる？

<div class="layout--split">
  <div class="card">
    <h3>Artifact footprint</h3>
    <div class="bar-chart">
      <div class="bar-chart__row">
        <b>Q8</b>
        <div class="bar-chart__track"><div class="bar-chart__fill" style="--bar-size: 100%"></div></div>
        <span class="bar-chart__value">29.05 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q6</b>
        <div class="bar-chart__track"><div class="bar-chart__fill" style="--bar-size: 77.2%"></div></div>
        <span class="bar-chart__value">22.43 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q5</b>
        <div class="bar-chart__track"><div class="bar-chart__fill" style="--bar-size: 67.3%"></div></div>
        <span class="bar-chart__value">19.54 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q4</b>
        <div class="bar-chart__track"><div class="bar-chart__fill" style="--bar-size: 57.9%"></div></div>
        <span class="bar-chart__value">16.81 GB</span>
      </div>
    </div>
  </div>
  <div>
    <table class="data-table">
      <thead>
        <tr>
          <th>variant</th>
          <th>end-to-end</th>
          <th>answer-bearing</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Q8</td><td>32/60</td><td>60/60</td></tr>
        <tr><td>Q6</td><td>32/60</td><td>60/60</td></tr>
        <tr><td>Q5</td><td>32/60</td><td>60/60</td></tr>
        <tr><td>Q4</td><td>27/60</td><td>59/60</td></tr>
      </tbody>
    </table>
    <div class="callout callout--info u-mt-20">
      answer-bearingではQ4もほぼ維持。<br />
      <strong>量子化 = 一律に大きく劣化ではなかった。</strong>
    </div>
    <div class="callout callout--warning u-mt-16">
      exact / format / end-to-endの同等性は未確定。
    </div>
  </div>
</div>

<p class="source">240/240 completed · Q8_0/Q6_K/Q5_K_M/Q4_K_M · 8K/32K · p50</p>

---

## Footprint vs. Quality, per Metric

<img width="100%" src="../experiments/exp_002-quantization_llama_cpp_gguf/results/figures/presentation-size-quality.png">

<p class="source">Qwen3.8-27B · llama.cpp · 240 capability trials · 30 independent tasks · p50 evidence position</p>

---

## Q4/Q8同等性はメトリック次第

<img width="80%" src="../experiments/exp_002-quantization_llama_cpp_gguf/results/figures/presentation-equivalence-by-metric.png">

<div class="quote u-mt-16">Answer-bearingは同等。End-to-end / Format-validはまだ<strong>inconclusive</strong>。</div>
<p class="source">Matched pairs · 60 pairs · 95% paired bootstrap CI · practical margin ±10pt</p>

---

## Context × Quantization はタスク依存

<table class="data-table u-mt-28">
  <thead>
    <tr>
      <th>task family</th>
      <th>Q8 8K → 32K</th>
      <th>Q4 8K → 32K</th>
      <th>observation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>literal</td>
      <td>2/10 → 6/10</td>
      <td>2/10 → 4/10</td>
      <td><span class="badge badge--pilot">context-dependent</span></td>
    </tr>
    <tr>
      <td>semantic</td>
      <td>4/10 → 3/10</td>
      <td>3/10 → 3/10</td>
      <td><span class="badge badge--measured">≈ constant</span></td>
    </tr>
    <tr>
      <td>multi-hop</td>
      <td>8/10 → 9/10</td>
      <td>7/10 → 8/10</td>
      <td><span class="badge badge--measured">≈ constant</span></td>
    </tr>
  </tbody>
</table>

<div class="layout--metrics u-mt-30">
  <div class="metric-card">
    <span class="metric-card__value">120/120</span>
    <span class="metric-card__label">matched trials</span>
  </div>
  <div class="metric-card">
    <span class="metric-card__value">p50</span>
    <span class="metric-card__label">evidence position only</span>
  </div>
  <div class="metric-card">
    <span class="metric-card__value">Not yet</span>
    <span class="metric-card__label">full position sweep</span>
  </div>
</div>

<p class="source">Calibrated matched pilot · descriptive interaction, not significance test</p>

---

## Task ごとの Success Rate

<img width="88%" src="../experiments/exp_003-context_x_quantization/results/figures/presentation-context-task-heatmap.png">

<p class="source">120 matched trials · all completed · values are end-to-end success counts</p>

---

## Q4のハンデが伸びるのは Literal だけ

<img width="88%" src="../experiments/exp_003-context_x_quantization/results/figures/presentation-context-gap.png">

<div class="quote u-mt-16">Q4が常に悪いわけでも、Q4とQ8が常に同じわけでもない。</div>
<p class="source">Matched pilot · p50 evidence position · n=10 per task family/cell · descriptive, not a significance test</p>

---

## Agent History：長い履歴そのものは壊れなかった

<div class="layout--split">
  <div>
    <div class="stat">
      300/300
      <small>
        <br />
        final task success
        <br />
        trajectory 1 → 32 turns
      </small>
    </div>
    <div class="layout--metrics u-mt-24">
      <div class="metric-card">
        <span class="metric-card__value">300/300</span>
        <span class="metric-card__label">critical-fact reuse</span>
      </div>
      <div class="metric-card">
        <span class="metric-card__value">0</span>
        <span class="metric-card__label">planning errors</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h3>むしろ壊れていたのは output protocol</h3>
    <ul>
      <li>以前のpilot：64-token limit → 30件 invalid output</li>
      <li>recheck：128-token JSON + 3 action attempts</li>
      <li>最大completion 109 tokens → 全件成功</li>
    </ul>
    <div class="callout callout--warning u-mt-18">
      履歴長による失敗に見えても、実際には<strong>出力制約</strong>が原因かもしれない。
    </div>
  </div>
</div>

<p class="source">Q8_0/Q4_K_M · trajectory 1/4/8/16/32 · 10 tasks × 3 deterministic repeats</p>

---

## Output Protocol を直すと失敗が消えた

<img width="100%" src="../experiments/exp_004-agent_context_growth/results/figures/presentation-agent-protocol-diagnosis.png">

<p class="source">Descriptive comparison: the recheck changed the output budget, JSON policy, and retry policy; this is not a causal ablation</p>

---

## 履歴が伸びても Reliability は落ちない

<img width="100%" src="../experiments/exp_004-agent_context_growth/results/figures/presentation-agent-recheck-stability.png">

<p class="source">Q8_0/Q4_K_M · 10 independent tasks · 3 greedy repeats per cell · one critical position (50%)</p>

---

## 一番大きな学び：LLMより先に、評価器が壊れる

<div class="layout--split">
  <div class="card">
    <h3>Example</h3>
    <pre><code>Expected:
ZX-4817

Model output:
ZX-4817.659</code></pre>
    <div class="callout callout--warning u-mt-22">
      exact matchでは失敗。<br />
      でも答えを含んでいるか？では別の判定になる。
    </div>
  </div>
  <div>
    <div class="layout--three">
      <div class="axis-card">
        <h3>exact</h3>
        <p>文字列が完全一致？</p>
      </div>
      <div class="axis-card axis-card--teal">
        <h3>answer-bearing</h3>
        <p>必要な答えを含む？</p>
      </div>
      <div class="axis-card axis-card--amber">
        <h3>format</h3>
        <p>指定形式を守る？</p>
      </div>
    </div>
    <div class="quote u-mt-30">モデルを測る前に、測定器を校正する。</div>
  </div>
</div>

<p class="source">Scorer calibration changed the interpretation of exp_001–exp_004</p>

---

<!-- _class: slide--closing -->
<div class="slide-closing">
  <div class="eyebrow">TAKEAWAY</div>
  <h1>Fits ≠ Useful</h1>
  <div class="rule"></div>
  <p class="lead">
    Local Qwenは動くか？より、
    <br />
    <strong>どこで・どう壊れるか</strong>を測ると面白い。
  </p>
  <div class="layout--next">
    <div class="card">
      <h3>1 · Quantization</h3>
      <p>Q4でもanswer-bearingはかなり残った</p>
    </div>
    <div class="card">
      <h3>2 · Context</h3>
      <p>window sizeより運用コストが先に効く</p>
    </div>
    <div class="card">
      <h3>3 · Evaluation</h3>
      <p>scorer / format / protocolを分離して見る</p>
    </div>
  </div>
  <p class="source">
    Next: full position sweep → repository-level validation (exp_005)
  </p>
</div>

---

## 宣伝
