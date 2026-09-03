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
  <div class="eyebrow">QWEN MEETUP TOKYO · INTERMEDIATE REPORT</div>
  <h1>
    When Does a Local LLM
    <br />
    Start to Break?
  </h1>
  <div class="rule"></div>
  <p class="lead">「入る文脈」と「使える能力」は、同じではない。</p>
  <p class="subtle">
    Qwen3.8-27B / llama.cpp / calibrated scorer
    <br />
    Context → quantization → interaction → agent history
  </p>
</div>

---

## WHOAMI

<img width="100%" src="./img/linkedin.png">

---

---

## “Break”を3つに分けて測る

<div class="layout--three">
  <div class="axis-card">
    <h3>Capability</h3>
    <p>
      答えを含むか。
      <br />
      正確に答えられるか。
      <br />
      形式も満たすか。
    </p>
  </div>
  <div class="axis-card axis-card--teal">
    <h3>Systems cost</h3>
    <p>
      モデルは小さくなるか。
      <br />
      長い入力を現実的な時間で
      <br />
      処理できるか。
    </p>
  </div>
  <div class="axis-card axis-card--amber">
    <h3>Trajectory reliability</h3>
    <p>
      発見した事実を、
      <br />
      長い履歴のあとでも
      <br />
      最終状態まで使えるか。
    </p>
  </div>
</div>

<div class="quote u-mt-54">スペック上のcontext windowではなく、能力が崩れる地点を測る。</div>
<p class="source">
  Scope: calibrated real-model measurements; fixture and incomplete matrices are kept separate.
</p>

---

## 4つの実験を、1本の測定ストーリーにする

<div class="experiment-flow">
  <div class="experiment-card">
    <div class="experiment-card__code">EXP_001</div>
    <div class="experiment-card__title">
      Context
      <br />
      baseline
    </div>
    <span class="badge badge--measured">MEASURED</span>
    <div class="experiment-card__kpi u-mt-18">60/60 + 9</div>
    <div class="experiment-card__note">
      baseline + bounded feasibility probe
      <br />
      Q8 · p50
    </div>
  </div>
  <div class="experiment-card">
    <div class="experiment-card__code">EXP_002</div>
    <div class="experiment-card__title">
      Quantization
      <br />
      trade-off
    </div>
    <span class="badge badge--measured">MEASURED</span>
    <div class="experiment-card__kpi u-mt-18">240/240</div>
    <div class="experiment-card__note">
      Q8 → Q4
      <br />
      artifact −42.1%
    </div>
  </div>
  <div class="experiment-card">
    <div class="experiment-card__code">EXP_003</div>
    <div class="experiment-card__title">
      Context ×
      <br />
      quantization
    </div>
    <span class="badge badge--pilot">PILOT</span>
    <div class="experiment-card__kpi u-mt-18">120/120</div>
    <div class="experiment-card__note">
      matched 8K / 32K
      <br />
      descriptive interaction
    </div>
  </div>
  <div class="experiment-card">
    <div class="experiment-card__code">EXP_004</div>
    <div class="experiment-card__title">
      Agent
      <br />
      history
    </div>
    <span class="badge badge--measured">RECHECK</span>
    <div class="experiment-card__kpi u-mt-18">300/300</div>
    <div class="experiment-card__note">
      1 → 32 turns
      <br />
      fixed output policy
    </div>
  </div>
  <div class="experiment-card experiment-card--future">
    <div class="experiment-card__code">NEXT</div>
    <div class="experiment-card__title">
      Repository
      <br />
      validation
    </div>
    <span class="badge badge--pending">UNMEASURED</span>
    <div class="experiment-card__kpi u-mt-18">exp_005</div>
    <div class="experiment-card__note">
      curated ↔ broad
      <br />
      machine-checked outcome
    </div>
  </div>
</div>

<div class="callout callout--info u-mt-30">
  観測された失敗を、モデル・評価器・出力プロトコルに分解していく。
</div>
<p class="source">
  Source: docs/findings.md; raw manifests and processed summaries for exp_001–exp_004.
</p>

---

## exp_001 — 長い文脈を「使える」とは限らない

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
    <div class="callout callout--info u-mt-26">
      answer-bearing correctnessは全セル10/10。
      <br />
      しかしend-to-end successは形式条件に左右された。
    </div>
  </div>
  <div>
    <table class="data-table">
      <thead>
        <tr>
          <th>task family</th>
          <th>8K</th>
          <th>32K</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>literal</td>
          <td>2/10</td>
          <td>6/10</td>
        </tr>
        <tr>
          <td>semantic</td>
          <td>4/10</td>
          <td>3/10</td>
        </tr>
        <tr>
          <td>multi-hop</td>
          <td>8/10</td>
          <td>9/10</td>
        </tr>
      </tbody>
    </table>
    <div class="callout callout--warning u-mt-18">
      baseline matrixはp50のみ。別のfeasibility probeは固定3タスク・p50で、position
      biasはまだ測っていない。
    </div>
  </div>
</div>
<p class="source">60/60 completed · Q8_0 · 30 independent tasks · calibrated.v1</p>

---

## exp_001 feasibility — 262Kは「入る」前に時間で壊れる

<table class="data-table u-mt-22">
  <thead>
    <tr>
      <th>target context</th>
      <th>attempts</th>
      <th>classification</th>
      <th>observed boundary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>64K</td>
      <td>3/3 completed</td>
      <td><span class="badge badge--measured">accepted + useful</span></td>
      <td>answer-bearing 3/3 · TTFT 783–786s</td>
    </tr>
    <tr>
      <td>128K</td>
      <td>3/3 attempted</td>
      <td><span class="badge badge--pilot">operational failure</span></td>
      <td>3/3 timeout at 900s · RSS ≈ 37.6GB</td>
    </tr>
    <tr>
      <td>262K</td>
      <td>3/3 attempted</td>
      <td><span class="badge badge--pilot">operational failure</span></td>
      <td>3/3 timeout at 900s · RSS ≈ 46.2GB</td>
    </tr>
  </tbody>
</table>

<div class="layout--split u-mt-28">
  <div class="callout callout--info">
    64Kでは、exact/formatではなく
    <strong>answer-bearing correctness</strong>
    で3タスクとも通過。
  </div>
  <div class="callout callout--warning">
    Q8・p50・固定3タスク・各1回・900秒timeoutの結果。262Kの一般的なeffective contextを意味しない。
  </div>
</div>
<div class="callout callout--warning u-mt-18">
  探索的予測：この単調な処理コストを仮定すると、900秒以内の実用境界は64Kと128Kの間。モデル固有のhard
  limitではない。
</div>
<p class="source">
  exp_001 feasibility manifest + raw JSONL · 9/9 attempted · source revision 67941cd · calibrated.v1
</p>

---

## exp_002 — 量子化で小さくなるが、品質の問いは単純ではない

<div class="layout--split">
  <div class="card">
    <h3>Artifact footprint</h3>
    <div class="bar-chart">
      <div class="bar-chart__row">
        <b>Q8</b>
        <div class="bar-chart__track">
          <div class="bar-chart__fill" style="--bar-size: 100%"></div>
        </div>
        <span class="bar-chart__value">29.05 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q6</b>
        <div class="bar-chart__track">
          <div class="bar-chart__fill" style="--bar-size: 77.2%"></div>
        </div>
        <span class="bar-chart__value">22.43 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q5</b>
        <div class="bar-chart__track">
          <div class="bar-chart__fill" style="--bar-size: 67.3%"></div>
        </div>
        <span class="bar-chart__value">19.54 GB</span>
      </div>
      <div class="bar-chart__row">
        <b>Q4</b>
        <div class="bar-chart__track">
          <div class="bar-chart__fill" style="--bar-size: 57.9%"></div>
        </div>
        <span class="bar-chart__value">16.81 GB</span>
      </div>
    </div>
    <div class="stat u-mt-20">
      −42.1%
      <small>
        <br />
        Q8 → Q4 artifact size
      </small>
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
        <tr>
          <td>Q8</td>
          <td>32/60</td>
          <td>60/60</td>
        </tr>
        <tr>
          <td>Q6</td>
          <td>32/60</td>
          <td>60/60</td>
        </tr>
        <tr>
          <td>Q5</td>
          <td>32/60</td>
          <td>60/60</td>
        </tr>
        <tr>
          <td>Q4</td>
          <td>27/60</td>
          <td>59/60</td>
        </tr>
      </tbody>
    </table>
    <div class="callout callout--warning u-mt-18">
      paired equivalence（95%
      CI、±10pp）：answer-bearingは同等範囲内。exact/end-to-endとformat-validは同等性未確定。timing
      probesは474/1,200。
    </div>
  </div>
</div>
<p class="source">240/240 completed · Q8_0/Q6_K/Q5_K_M/Q4_K_M · 8K/32K · p50</p>

---

## exp_003 — 量子化×文脈のinteractionは「タスク依存」

<table class="data-table u-mt-28">
  <thead>
    <tr>
      <th>task family</th>
      <th>Q8 8K → 32K</th>
      <th>Q4 8K → 32K</th>
      <th>interaction</th>
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

<div class="layout--metrics u-mt-34">
  <div class="metric-card">
    <span class="metric-card__value">120/120</span>
    <span class="metric-card__label">completed matched trials</span>
  </div>
  <div class="metric-card">
    <span class="metric-card__value">p50</span>
    <span class="metric-card__label">evidence position only</span>
  </div>
  <div class="metric-card">
    <span class="metric-card__value">64K/128K</span>
    <span class="metric-card__label">not measured</span>
  </div>
</div>
<div class="callout callout--info u-mt-28">「Q4は常に悪い」でも「Q4とQ8は常に同じ」でもない。</div>
<p class="source">
  Calibrated matched pilot · one greedy run per independent task · descriptive labels, not
  significance tests.
</p>

---

## exp_004 — agentの失敗は、履歴長だけでは説明できない

<div class="layout--split">
  <div>
    <div class="stat">
      300/300
      <small>
        <br />
        final task success
        <br />
        trajectory 1 → 32
      </small>
    </div>
    <div class="layout--metrics">
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
    <h3>Protocol changed the observation</h3>
    <ul>
      <li>以前のpilot：64-token limitで30件がinvalid output</li>
      <li>recheck：128-token JSON + 3 action attempts</li>
      <li>最大completionは109 tokens、全件成功</li>
    </ul>
    <div class="callout callout--warning">
      これは因果ablationではない。出力予算とretry policyが同時に変わっている。
    </div>
  </div>
</div>
<div class="quote u-mt-34">固定ポリシー下では、履歴長による劣化は観測されなかった。</div>
<p class="source">
  Q8_0/Q4_K_M · trajectory 1/4/8/16/32 · p50 · 10 tasks × 3 deterministic repeats
</p>

---

## 5つの強い主張を、証拠の強さに合わせて言い換える

<div class="claim-grid">
  <div class="claim-card">
    <strong class="claim-card__title">
      <span class="badge badge--pilot">境界を観測</span>
      262Kまで有効に使える？
    </strong>
    <span class="claim-card__detail">
      このQ8環境では64Kは通過、128K/262Kは900秒timeout。一般的なeffective contextではない。
    </span>
  </div>
  <div class="claim-card">
    <strong class="claim-card__title">
      <span class="badge badge--pilot">指標限定</span>
      Q4とQ8は同等？
    </strong>
    <span class="claim-card__detail">
      answer-bearingは−1.7pp（95% CI −5.0〜0.0pp）で±10pp内。exact/formatは同等性未確定。
    </span>
  </div>
  <div class="claim-card">
    <strong class="claim-card__title">
      <span class="badge badge--pending">未確認</span>
      position bias？
    </strong>
    <span class="claim-card__detail">p50のみ測定。全position sweepが必要。</span>
  </div>
  <div class="claim-card">
    <strong class="claim-card__title">
      <span class="badge badge--measured">支持せず</span>
      履歴が長いほど悪化？
    </strong>
    <span class="claim-card__detail">固定ポリシーのrecheckでは300/300。普遍則ではない。</span>
  </div>
  <div class="claim-card">
    <strong class="claim-card__title">
      <span class="badge badge--pending">未測定</span>
      repositoryでも再現？
    </strong>
    <span class="claim-card__detail">exp_005のcurated vs broad pilotが次の検証。</span>
  </div>
</div>

<p class="source">
  Claim boundary follows the experiment manifests, scorer version, task catalog, and raw-result
  provenance.
</p>

---

<!-- _class: slide--closing -->
<div class="slide-closing">
  <div class="eyebrow">TAKEAWAY</div>
  <h1>Fits ≠ Useful</h1>
  <div class="rule"></div>
  <p class="lead">
    モデルを評価する前に、
    <br />
    評価器と出力プロトコルを校正する。
  </p>
  <div class="layout--next">
    <div class="card">
      <h3>1 · Scorer</h3>
      <p>exact / answer-bearing / formatを分離</p>
    </div>
    <div class="card">
      <h3>2 · Position</h3>
      <p>matched evidence sweep</p>
    </div>
    <div class="card">
      <h3>3 · Transfer</h3>
      <p>repository task validation</p>
    </div>
  </div>
  <p class="source">
    Current status: exp_001 baseline + bounded feasibility probe; exp_002–exp_004 measured or pilot;
    exp_005 remains unmeasured.
  </p>
</div>
