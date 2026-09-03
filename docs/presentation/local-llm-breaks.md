---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  :root {
    --ink: #10233f;
    --muted: #64748b;
    --line: #d7e0ec;
    --paper: #f7f9fc;
    --navy: #163a63;
    --blue: #1d70b8;
    --teal: #168b8b;
    --amber: #c47a11;
    --red: #b23b3b;
    --green: #287a54;
  }
  section {
    background: var(--paper);
    color: var(--ink);
    font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    padding: 54px 68px 48px;
  }
  section::after {
    color: #8a9ab0;
    font-size: 13px;
    right: 58px;
    bottom: 24px;
  }
  h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
  h1 { font-size: 46px; line-height: 1.12; margin: 0 0 18px; }
  h2 { font-size: 34px; line-height: 1.16; margin: 0 0 18px; }
  h3 { font-size: 22px; margin: 0 0 8px; }
  p, li { font-size: 21px; line-height: 1.42; }
  ul { margin-top: 10px; }
  .eyebrow { color: var(--blue); font-size: 16px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
  .lead { font-size: 30px; line-height: 1.3; font-weight: 700; max-width: 920px; }
  .subtle { color: var(--muted); font-size: 17px; }
  .source { color: #73839a; font-size: 13px; margin-top: 18px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 34px; align-items: start; }
  .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; align-items: stretch; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 22px 24px; box-shadow: 0 5px 18px rgba(16,35,63,.05); }
  .card p { margin: 8px 0 0; font-size: 18px; }
  .hero { display: flex; height: 74%; flex-direction: column; justify-content: center; }
  .rule { height: 5px; width: 98px; background: var(--blue); border-radius: 4px; margin: 18px 0 22px; }
  .quote { border-left: 6px solid var(--blue); padding: 4px 0 4px 20px; font-size: 28px; line-height: 1.32; font-weight: 700; }
  .axis { border-top: 4px solid var(--blue); padding-top: 14px; }
  .axis.teal { border-color: var(--teal); }
  .axis.amber { border-color: var(--amber); }
  .big { font-size: 42px; font-weight: 800; line-height: 1.05; }
  .big small { font-size: 18px; font-weight: 600; color: var(--muted); }
  .metric-row { display: flex; gap: 16px; margin: 16px 0 0; }
  .metric { flex: 1; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .metric .value { display: block; font-size: 28px; font-weight: 800; }
  .metric .label { display: block; color: var(--muted); font-size: 15px; margin-top: 5px; }
  .map { display: grid; grid-template-columns: repeat(5, 1fr); gap: 13px; align-items: center; margin-top: 28px; }
  .map-node { position: relative; min-height: 192px; background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 17px 15px 14px; box-shadow: 0 5px 15px rgba(16,35,63,.05); }
  .map-node::after { content: "→"; position: absolute; right: -22px; top: 78px; color: #91a3b9; font-size: 28px; font-weight: 700; }
  .map-node:last-child::after { content: ""; }
  .map-node.future { border: 2px dashed #b5c0cf; background: #f1f4f8; }
  .map-code { font-size: 14px; color: var(--blue); font-weight: 800; letter-spacing: .06em; }
  .map-title { font-size: 21px; font-weight: 800; margin: 12px 0 12px; line-height: 1.18; }
  .map-kpi { font-size: 18px; font-weight: 800; }
  .map-note { color: var(--muted); font-size: 14px; line-height: 1.35; margin-top: 8px; }
  .status { display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 13px; font-weight: 800; letter-spacing: .04em; }
  .measured { color: #1b633f; background: #dff2e8; }
  .pilot { color: #8d5b0a; background: #fff0ce; }
  .pending { color: #5d6d82; background: #e7edf4; }
  .table { display: table; width: 100%; table-layout: fixed; border-collapse: collapse; background: #fff; border: 1px solid var(--line); font-size: 17px; }
  .table th, .table td { padding: 11px 13px; border-bottom: 1px solid var(--line); text-align: right; }
  .table th:first-child, .table td:first-child { text-align: left; }
  .table th { color: var(--muted); font-size: 14px; text-transform: uppercase; letter-spacing: .04em; }
  .table tr:last-child td { border-bottom: 0; }
  .highlight { background: #eaf3fb; border-left: 5px solid var(--blue); border-radius: 9px; padding: 15px 18px; font-size: 20px; font-weight: 700; }
  .warning { background: #fff5df; border-left: 5px solid var(--amber); border-radius: 9px; padding: 14px 18px; font-size: 17px; }
  .bars { margin-top: 8px; }
  .bar-line { display: grid; grid-template-columns: 74px 1fr 78px; gap: 12px; align-items: center; margin: 15px 0; font-size: 16px; }
  .bar-track { height: 20px; background: #e6edf5; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #1d70b8, #4fa2db); border-radius: 999px; }
  .bar-value { font-weight: 800; text-align: right; }
  .claim-list { display: grid; grid-template-columns: 1fr 1fr; gap: 13px 20px; margin-top: 20px; }
  .claim { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .claim strong { display: block; font-size: 19px; margin-bottom: 6px; }
  .claim span { font-size: 16px; line-height: 1.35; color: #42536a; }
  .closing { display: flex; flex-direction: column; justify-content: center; height: 77%; }
  .next { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 22px; }
  .next .card { min-height: 120px; }
  .next .card h3 { font-size: 19px; }
  .next .card p { font-size: 16px; }
---

<!-- _paginate: false -->

<div class="hero">
<div class="eyebrow">QWEN MEETUP TOKYO · INTERMEDIATE REPORT</div>
<h1>When Does a Local LLM<br>Start to Break?</h1>
<div class="rule"></div>
<p class="lead">「入る文脈」と「使える能力」は、同じではない。</p>
<p class="subtle">Qwen3.8-27B / llama.cpp / calibrated scorer<br>Context → quantization → interaction → agent history</p>
</div>

---

## “Break”を3つに分けて測る

<div class="grid3">
<div class="axis"><h3>Capability</h3><p>答えを含むか。<br>正確に答えられるか。<br>形式も満たすか。</p></div>
<div class="axis teal"><h3>Systems cost</h3><p>モデルは小さくなるか。<br>長い入力を現実的な時間で<br>処理できるか。</p></div>
<div class="axis amber"><h3>Trajectory reliability</h3><p>発見した事実を、<br>長い履歴のあとでも<br>最終状態まで使えるか。</p></div>
</div>

<div class="quote" style="margin-top:54px;">スペック上のcontext windowではなく、能力が崩れる地点を測る。</div>
<p class="source">Scope: calibrated real-model measurements; fixture and incomplete matrices are kept separate.</p>

---

## 4つの実験を、1本の測定ストーリーにする

<div class="map">
<div class="map-node">
  <div class="map-code">EXP_001</div><div class="map-title">Context<br>baseline</div>
  <span class="status measured">MEASURED</span><div class="map-kpi" style="margin-top:10px;">60/60 + 9</div><div class="map-note">baseline + bounded feasibility probe<br>Q8 · p50</div>
</div>
<div class="map-node">
  <div class="map-code">EXP_002</div><div class="map-title">Quantization<br>trade-off</div>
  <span class="status measured">MEASURED</span><div class="map-kpi" style="margin-top:10px;">240/240</div><div class="map-note">Q8 → Q4<br>artifact −42.1%</div>
</div>
<div class="map-node">
  <div class="map-code">EXP_003</div><div class="map-title">Context ×<br>quantization</div>
  <span class="status pilot">PILOT</span><div class="map-kpi" style="margin-top:10px;">120/120</div><div class="map-note">matched 8K / 32K<br>descriptive interaction</div>
</div>
<div class="map-node">
  <div class="map-code">EXP_004</div><div class="map-title">Agent<br>history</div>
  <span class="status measured">RECHECK</span><div class="map-kpi" style="margin-top:10px;">300/300</div><div class="map-note">1 → 32 turns<br>fixed output policy</div>
</div>
<div class="map-node future">
  <div class="map-code">NEXT</div><div class="map-title">Repository<br>validation</div>
  <span class="status pending">UNMEASURED</span><div class="map-kpi" style="margin-top:10px;">exp_005</div><div class="map-note">curated ↔ broad<br>machine-checked outcome</div>
</div>
</div>

<div class="highlight" style="margin-top:30px;">観測された失敗を、モデル・評価器・出力プロトコルに分解していく。</div>
<p class="source">Source: docs/findings.md; raw manifests and processed summaries for exp_001–exp_004.</p>

---

## exp_001 — 長い文脈を「使える」とは限らない

<div class="grid2">
<div>
<div class="big">75s → 339.5s<small><br>median stream-derived TTFT<br>8K → 32K</small></div>
<div class="highlight" style="margin-top:26px;">answer-bearing correctnessは全セル10/10。<br>しかしend-to-end successは形式条件に左右された。</div>
</div>
<div>
<table class="table"><thead><tr><th>task family</th><th>8K</th><th>32K</th></tr></thead><tbody>
<tr><td>literal</td><td>2/10</td><td>6/10</td></tr>
<tr><td>semantic</td><td>4/10</td><td>3/10</td></tr>
<tr><td>multi-hop</td><td>8/10</td><td>9/10</td></tr>
</tbody></table>
<div class="warning" style="margin-top:18px;">baseline matrixはp50のみ。別のfeasibility probeは固定3タスク・p50で、position biasはまだ測っていない。</div>
</div>
</div>
<p class="source">60/60 completed · Q8_0 · 30 independent tasks · calibrated.v1</p>

---

## exp_001 feasibility — 262Kは「入る」前に時間で壊れる

<table class="table" style="margin-top:22px;"><thead><tr><th>target context</th><th>attempts</th><th>classification</th><th>observed boundary</th></tr></thead><tbody>
<tr><td>64K</td><td>3/3 completed</td><td><span class="status measured">accepted + useful</span></td><td>answer-bearing 3/3 · TTFT 783–786s</td></tr>
<tr><td>128K</td><td>3/3 attempted</td><td><span class="status pilot">operational failure</span></td><td>3/3 timeout at 900s · RSS ≈ 37.6GB</td></tr>
<tr><td>262K</td><td>3/3 attempted</td><td><span class="status pilot">operational failure</span></td><td>3/3 timeout at 900s · RSS ≈ 46.2GB</td></tr>
</tbody></table>

<div class="grid2" style="margin-top:28px;">
<div class="highlight">64Kでは、exact/formatではなく<strong>answer-bearing correctness</strong>で3タスクとも通過。</div>
<div class="warning">Q8・p50・固定3タスク・各1回・900秒timeoutの結果。262Kの一般的なeffective contextを意味しない。</div>
</div>
<p class="source">exp_001 feasibility manifest + raw JSONL · 9/9 attempted · source revision 67941cd · calibrated.v1</p>

---

## exp_002 — 量子化で小さくなるが、品質の問いは単純ではない

<div class="grid2">
<div class="card">
<h3>Artifact footprint</h3>
<div class="bars">
<div class="bar-line"><b>Q8</b><div class="bar-track"><div class="bar-fill" style="width:100%"></div></div><span class="bar-value">29.05 GB</span></div>
<div class="bar-line"><b>Q6</b><div class="bar-track"><div class="bar-fill" style="width:77.2%"></div></div><span class="bar-value">22.43 GB</span></div>
<div class="bar-line"><b>Q5</b><div class="bar-track"><div class="bar-fill" style="width:67.3%"></div></div><span class="bar-value">19.54 GB</span></div>
<div class="bar-line"><b>Q4</b><div class="bar-track"><div class="bar-fill" style="width:57.9%"></div></div><span class="bar-value">16.81 GB</span></div>
</div>
<div class="big" style="margin-top:20px;">−42.1%<small><br>Q8 → Q4 artifact size</small></div>
</div>
<div>
<table class="table"><thead><tr><th>variant</th><th>end-to-end</th><th>answer-bearing</th></tr></thead><tbody>
<tr><td>Q8</td><td>32/60</td><td>60/60</td></tr>
<tr><td>Q6</td><td>32/60</td><td>60/60</td></tr>
<tr><td>Q5</td><td>32/60</td><td>60/60</td></tr>
<tr><td>Q4</td><td>27/60</td><td>59/60</td></tr>
</tbody></table>
<div class="warning" style="margin-top:18px;">paired equivalence（95% CI、±10pp）：answer-bearingは同等範囲内。exact/end-to-endとformat-validは同等性未確定。timing probesは474/1,200。</div>
</div>
</div>
<p class="source">240/240 completed · Q8_0/Q6_K/Q5_K_M/Q4_K_M · 8K/32K · p50</p>

---

## exp_003 — 量子化×文脈のinteractionは「タスク依存」

<table class="table" style="margin-top:28px;"><thead><tr><th>task family</th><th>Q8 8K → 32K</th><th>Q4 8K → 32K</th><th>interaction</th></tr></thead><tbody>
<tr><td>literal</td><td>2/10 → 6/10</td><td>2/10 → 4/10</td><td><span class="status pilot">context-dependent</span></td></tr>
<tr><td>semantic</td><td>4/10 → 3/10</td><td>3/10 → 3/10</td><td><span class="status measured">≈ constant</span></td></tr>
<tr><td>multi-hop</td><td>8/10 → 9/10</td><td>7/10 → 8/10</td><td><span class="status measured">≈ constant</span></td></tr>
</tbody></table>

<div class="metric-row" style="margin-top:34px;">
<div class="metric"><span class="value">120/120</span><span class="label">completed matched trials</span></div>
<div class="metric"><span class="value">p50</span><span class="label">evidence position only</span></div>
<div class="metric"><span class="value">64K/128K</span><span class="label">not measured</span></div>
</div>
<div class="highlight" style="margin-top:28px;">「Q4は常に悪い」でも「Q4とQ8は常に同じ」でもない。</div>
<p class="source">Calibrated matched pilot · one greedy run per independent task · descriptive labels, not significance tests.</p>

---

## exp_004 — agentの失敗は、履歴長だけでは説明できない

<div class="grid2">
<div>
<div class="big">300/300<small><br>final task success<br>trajectory 1 → 32</small></div>
<div class="metric-row">
<div class="metric"><span class="value">300/300</span><span class="label">critical-fact reuse</span></div>
<div class="metric"><span class="value">0</span><span class="label">planning errors</span></div>
</div>
</div>
<div class="card">
<h3>Protocol changed the observation</h3>
<ul>
<li>以前のpilot：64-token limitで30件がinvalid output</li>
<li>recheck：128-token JSON + 3 action attempts</li>
<li>最大completionは109 tokens、全件成功</li>
</ul>
<div class="warning">これは因果ablationではない。出力予算とretry policyが同時に変わっている。</div>
</div>
</div>
<div class="quote" style="margin-top:34px;">固定ポリシー下では、履歴長による劣化は観測されなかった。</div>
<p class="source">Q8_0/Q4_K_M · trajectory 1/4/8/16/32 · p50 · 10 tasks × 3 deterministic repeats</p>

---

## 5つの強い主張を、証拠の強さに合わせて言い換える

<div class="claim-list">
<div class="claim"><strong><span class="status pilot">境界を観測</span> 262Kまで有効に使える？</strong><span>このQ8環境では64Kは通過、128K/262Kは900秒timeout。一般的なeffective contextではない。</span></div>
<div class="claim"><strong><span class="status pilot">指標限定</span> Q4とQ8は同等？</strong><span>answer-bearingは−1.7pp（95% CI −5.0〜0.0pp）で±10pp内。exact/formatは同等性未確定。</span></div>
<div class="claim"><strong><span class="status pending">未確認</span> position bias？</strong><span>p50のみ測定。全position sweepが必要。</span></div>
<div class="claim"><strong><span class="status measured">支持せず</span> 履歴が長いほど悪化？</strong><span>固定ポリシーのrecheckでは300/300。普遍則ではない。</span></div>
<div class="claim"><strong><span class="status pending">未測定</span> repositoryでも再現？</strong><span>exp_005のcurated vs broad pilotが次の検証。</span></div>
</div>

<p class="source">Claim boundary follows the experiment manifests, scorer version, task catalog, and raw-result provenance.</p>

---

<div class="closing">
<div class="eyebrow">TAKEAWAY</div>
<h1>Fits ≠ Useful</h1>
<div class="rule"></div>
<p class="lead">モデルを評価する前に、<br>評価器と出力プロトコルを校正する。</p>
<div class="next">
<div class="card"><h3>1 · Scorer</h3><p>exact / answer-bearing / formatを分離</p></div>
<div class="card"><h3>2 · Position</h3><p>matched evidence sweep</p></div>
<div class="card"><h3>3 · Transfer</h3><p>repository task validation</p></div>
</div>
<p class="source">Current status: exp_001 baseline + bounded feasibility probe; exp_002–exp_004 measured or pilot; exp_005 remains unmeasured.</p>
</div>
