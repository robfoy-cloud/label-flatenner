// node web/test/engine.test.js [path to label_75.f32 rendered by Python make_label(75,110,8)]
const fs = require("fs");
const LF = require("../engine.js");
let fails = 0;
const check = (name, ok, info) => { console.log(`${ok ? "ok  " : "FAIL"} ${name}  ${info ?? ""}`); if (!ok) fails++; };

const views = [{}, { elevation: 0 }, { elevation: -25, rotation: -10 },
  { elevation: 50, lean: -20, pan: -8, labelW: 70 }, { distance: 160, labelW: 60 }];
for (const v of views) {
  const sc = LF.Scene.fromView(v);
  check(`visible ${JSON.stringify(v)}`, sc.labelVisible());
  const m = new LF.Model(sc.exactFeatures(), sc.f, [sc.W / 2, sc.H / 2]).fitHeights();
  const dw = Math.abs(LF.deg(m.width) - LF.deg(sc.th1 - sc.th0));
  const da = Math.abs(m.width / m.height / (sc.lw / sc.lh) - 1);
  check(`exact geometry ${JSON.stringify(v)}`, dw < 1e-6 && da < 1e-6, `span err ${dw.toExponential(1)}°`);
  const img = new Uint8ClampedArray(sc.W * sc.H * 3);
  const r = LF.flatten(img, sc.W, sc.H, sc.f, { feats: sc.exactFeatures() });
  const s = LF.score(sc, r.cyl.map, r.outW, r.outH, 2);
  check(`exact flattening ${JSON.stringify(v)}`, s.rms < 0.05 && s.max < 0.35, `rms ${s.rms.toFixed(3)} max ${s.max.toFixed(3)} mm`);
}

const path = process.argv[2];
if (path) {
  const buf = fs.readFileSync(path);
  const data = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const label = { w: 75 * 8, h: 110 * 8, data };
  for (const v of [{ labelW: 75 }, { elevation: 40, rotation: -8, lean: 10, labelW: 70, W: 960, H: 720 }]) {
    const sc = LF.Scene.fromView(v);
    let t = Date.now();
    const img = sc.render(label, 2, 1.5);
    const tr = Date.now() - t; t = Date.now();
    const r = LF.flatten(img, sc.W, sc.H, sc.f);
    const tf = Date.now() - t;
    const s = LF.score(sc, r.cyl.map, r.outW, r.outH), n = LF.score(sc, r.naive.map, r.outW, r.outH);
    check(`rendered ${JSON.stringify(v)}`, s.rms < 1.0 && n.rms > 3.0,
      `cyl rms ${s.rms.toFixed(2)} naive ${n.rms.toFixed(2)} mm | render ${tr} ms flatten ${tf} ms`);
    if (!v.W) fs.writeFileSync("/tmp/claude-0/js_render.rgb", Buffer.from(img.buffer));
  }
}
process.exit(fails ? 1 : 0);
