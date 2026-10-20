const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/yifan/.claude/skills/synced/4b7f1779-d009-4eb3-9635-27273f50c522_e4be8fd0-4976-41ca-84ca-94c8f154e3cb/pptx/scripts/apply_theme.js");
const path = require("path");
const fs = require("fs");
const IMG = (f) => path.join(__dirname, "img", f);
const LINKS = JSON.parse(fs.readFileSync(path.join(__dirname, "links.json"), "utf8"));

const THEME = {
  name: "Lacrosse Scoop",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2226", lt1: "FFFFFF", dk2: "3A4149", lt2: "F1F3F4",
    accent1: "E6B800", accent2: "1B9E77", accent3: "D64545", accent4: "2F6FD6", accent5: "E8743B", accent6: "8A939C",
    hlink: "2F6FD6", folHlink: "6B4FA0",
  },
};
const HEX = THEME.colors;

const deck = new pptxgen();
deck.layout = "LAYOUT_WIDE";
deck.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
deck.title = "RoboMaster Lacrosse Scoop - Progress Report";
deck.author = "Yifan";
const C = deck.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

deck.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: 7.4, h: 1.8, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.2, w: 7.0, h: 1.4, fontSize: 18, color: C.accent1, valign: "top", margin: 0 }, text: "" } },
  ],
});
deck.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.8, fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});
deck.defineSlideMaster({
  title: "DARK_CONTENT",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.8, fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(deck.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.08 });
}
function numDot(slide, x, y, n, fill = C.accent1, txt = C.text1) {
  slide.addShape(deck.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: fill }, line: { type: "none" } });
  slide.addText(String(n), { x, y, w: 0.46, h: 0.46, align: "center", valign: "middle", fontSize: 16, bold: true, color: txt, margin: 0, isTextBox: true });
}
function caption(slide, text, x, y, w, color = C.accent6) {
  slide.addText(text, { x, y, w, h: 0.35, fontSize: 11, italic: true, color, margin: 0, isTextBox: true });
}
// Video tile: thumbnail (hyperlinked to the Drive file when a link is set), play badge, label and link line.
function video(slide, key, x, y, w, label, dur, dark = false) {
  const h = (w * 9) / 16;
  const url = LINKS[key];
  const img = { path: IMG("vid/" + key + ".jpg"), x, y, w, h, objectName: "video-" + key };
  if (url) img.hyperlink = { url, tooltip: "Open video on Google Drive" };
  slide.addImage(img);
  const r = Math.min(0.7, w * 0.16);
  const play = { x: x + w / 2 - r / 2, y: y + h / 2 - r / 2, w: r, h: r, fill: { color: C.text1, transparency: 25 }, line: { color: C.background1, width: 1.5 } };
  if (url) play.hyperlink = { url };
  slide.addShape(deck.shapes.OVAL, play);
  slide.addShape(deck.shapes.ISOSCELES_TRIANGLE, { x: x + w / 2 - r * 0.16, y: y + h / 2 - r * 0.2, w: r * 0.4, h: r * 0.4, rotate: 90, fill: { color: C.background1 }, line: { type: "none" } });
  slide.addText([
    { text: label, options: { bold: true, color: dark ? C.background1 : C.text1, breakLine: true } },
    url
      ? { text: "▶ Watch on Google Drive (" + dur + ")", options: { color: C.accent4, hyperlink: { url } } }
      : { text: "Google Drive link: to be added (" + dur + ")", options: { color: C.accent6, italic: true } },
  ], { x, y: y + h + 0.08, w, h: 0.6, fontSize: 12, margin: 0, valign: "top", isTextBox: true });
}
const chartBase = (title) => ({
  showTitle: true, title, titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk2, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent6, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFormatCode: "0\"%\"",
  valGridLine: { color: "E3E6E8", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: false, barGapWidthPct: 60,
});
const HDR = (t, align) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: align || "left" } });

// ================= 1. Title =================
deck.addSection({ title: "Intro" });
let s = deck.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Intro" });
s.addText("Scooping a Lacrosse Ball with a RoboMaster", { placeholder: "title" });
s.addText([
  { text: "Open-loop scoop trajectory, step-by-step randomization, and the plan for torque sensing", options: { breakLine: true } },
  { text: "Isaac Lab task Isaac-RoboMaster-Lacrosse-Direct-v0  ·  October 2026", options: { fontSize: 14, color: C.accent6 } },
], { placeholder: "body" });
video(s, "wheels_slow", 8.3, 1.7, 4.4, "Current best scoop, 4× slow motion", "12 s", true);
s.addNotes("Goal: a RoboMaster base with a 2-DOF arm holding a lacrosse stick drives to a ball and scoops it into the pocket. All results are from Isaac Lab simulation, with the robot model built from measured real-robot numbers. Video thumbnails in this deck link to the clips on Google Drive.");

// ================= 2. Summary =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Intro" });
s.addText("A hand-designed open-loop scoop is near-perfect in simulation", { placeholder: "title" });
const stats = [
  ["98.3%", "393 / 400 robots hold the ball", "1200 Hz physics, ball placed ±3 cm / ±1 cm at random", C.accent2],
  ["300/300", "with realistic uncertainty", "wheel-driven 4.14 kg base, slip, latency, friction, gravity-comp error", C.accent2],
  ["±2 cm", "alignment tolerance", "lateral ±2 cm and heading ±1° still 95–98%; ±4 cm or ±2° fails", C.accent3],
];
stats.forEach(([big, label, sub, col], i) => {
  const x = M + i * 4.1, y = 1.6, w = 3.8;
  card(s, x, y, w, 3.1);
  s.addText(big, { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: 1.1, fontSize: 48, bold: true, color: col, margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.3, y: y + 1.45, w: w - 0.6, h: 0.6, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText(sub, { x: x + 0.3, y: y + 2.1, w: w - 0.6, h: 0.9, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
});
s.addText([
  { text: "Bottom line: ", options: { bold: true } },
  { text: "the scoop motion itself works in sim under every realistic randomization we could add. What is left is (1) the robot's position error relative to the ball and (2) transfer to hardware. Next week: measure real motor torque at impact, as a first step toward torque-triggered or learned control." },
], { x: M, y: 5.05, w: W - 2 * M, h: 1.2, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Three headline numbers. 98.3% is the latest 400-robot run at 1200 Hz with random ball placement. 300/300 is with every realistic-uncertainty knob on. The red number is the open problem: the robot must be lined up within about 2 cm and 1 degree.");

// ================= 3. Setup =================
deck.addSection({ title: "Setup" });
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("The simulated robot is built from measured real-robot numbers", { placeholder: "title" });
s.addImage({ path: IMG("frame_contact_side.png"), x: M, y: 1.5, w: 6.0, h: 6.0 * 410 / 730, objectName: "side-view" });
caption(s, "Side view at contact (t = 0.9 s): the lip presses on the floor and the ball enters the pocket", M, 1.5 + 6.0 * 410 / 730 + 0.1, 6.0);
[
  ["Base", "Wheel-driven RoboMaster, 4.14 kg total; 1.5 m/s² accel, ≤2 m/s; ±10% speed error"],
  ["Arm", "Shoulder motor on chassis, elbow motor at the elbow; link 1: 0.30 m, 0.68 kg incl. motor"],
  ["Stick", "1.146 m, 0.457 kg, CoM 0.142 m ahead of the elbow"],
  ["Motors", "MIT control: Kp 10 / 25, Kd 3, 4 N·m and 5 rad/s limits, gravity-comp feed-forward"],
  ["Task", "Ball (≈140 g, r 32 mm) at rest ~1 m ahead; success = ball held in the net, off the ground"],
].forEach(([k, v], i) => {
  const y = 1.55 + i * 1.0;
  s.addText(k, { x: 7.1, y, w: 1.3, h: 0.8, fontSize: 16, bold: true, color: C.accent3, margin: 0, valign: "top", isTextBox: true });
  s.addText(v, { x: 8.4, y, w: 4.3, h: 0.85, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Masses, link lengths and motor parameters come from the real robot. With Kd fixed at 3, Kp 10 / 25 gives a damping ratio of about 1.2. Gravity compensation is fed forward.");

// ================= 4. Video: best run =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("Watch: the current best scoop on the wheel-driven base", { placeholder: "title" });
video(s, "wheels_real", M, 1.55, 3.85, "Real time", "3 s");
video(s, "wheels_slow", M + 4.1, 1.55, 3.85, "4× slow motion", "12 s");
video(s, "wheels_anim", M + 8.2, 1.55, 3.85, "Side-view animation with joint plots", "6 s");
card(s, M, 4.55, W - 2 * M, 1.85);
s.addText([
  { text: "What to look for", options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } },
  { text: "Lip pressed on the floor while driving at 0.9 m/s (elbow feed-forward −1.0 N·m)", options: { bullet: true, breakLine: true } },
  { text: "Ball rolls onto the net at ~0.85 s; lift starts at 0.9 s at 1 rad/s; driving stops 1.0 s later", options: { bullet: true, breakLine: true } },
  { text: "Ball ends ~0.5 m above the floor, held in the pocket", options: { bullet: true } },
], { x: M + 0.3, y: 4.7, w: W - 2 * M - 0.6, h: 1.6, fontSize: 14, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true });
s.addNotes("Click any thumbnail to open the clip on Google Drive.");

// ================= 5. RL first attempt =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
s.addText("Pure RL kept falling into local optima, so we switched approach", { placeholder: "title" });
[
  ["Fix the sim first", "Contact stack overflow at 8192 envs dropped contacts; the first policy exploited it. Spawn height and ball placement were also fixed."],
  ["Shape the reward", "Policies folded the arm into a corner covering the ball. Fixed with a lip-to-ball reach term, arm range limits and a 2-stage curriculum."],
  ["Stage 1 worked, partly", "Arm locked, forward-only: ball popped onto the net 99.4%, but the robot hit at 1.23 m/s and 32% of balls were carried out of the 3 m arena."],
  ["Stage 2 failed", "Unlocking the arm: untrained arm outputs raised the stick before impact. We stopped and switched to an open-loop trajectory (residual RL kept for later)."],
].forEach(([h, b], i) => {
  const x = M + i * 3.07, y = 1.6, w = 2.8;
  card(s, x, y, w, 4.6);
  numDot(s, x + 0.25, y + 0.3, i + 1, i === 3 ? C.accent3 : C.accent1, i === 3 ? C.background1 : C.text1);
  s.addText(h, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.7, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText(b, { x: x + 0.25, y: y + 1.7, w: w - 0.5, h: 2.8, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("RSL-RL PPO with up to 8192 parallel environments. Both the reward and the simulator had exploitable holes. The pivot: hand-design the motion, search its few parameters, keep RL for residual corrections.");

// ================= 6. Trajectory design =================
deck.addSection({ title: "Method" });
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("The scoop is a 5-parameter open-loop trajectory found by grid search", { placeholder: "title" });
[
  ["Drive", "ramp to v = 0.9 m/s"],
  ["Press", "elbow FF −1.0 N·m holds the lip on the floor"],
  ["Lift", "at t = 0.9 s, 1 rad/s ramp: shoulder +0.3, elbow +0.7 rad"],
  ["Brake", "stop driving 1.0 s after lift"],
].forEach(([h, b], i) => {
  const x = M + i * 3.07, y = 1.55, w = 2.75;
  s.addShape(deck.shapes.CHEVRON, { x, y, w, h: 0.6, fill: { color: i === 2 ? C.accent2 : C.text2 }, line: { type: "none" } });
  s.addText(h, { x: x + 0.3, y, w: w - 0.6, h: 0.6, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(b, { x, y: y + 0.7, w, h: 0.6, fontSize: 14, color: C.text1, margin: 0, align: "center", valign: "top", isTextBox: true });
});
s.addImage({ path: IMG("trajectory_timeseries.png"), x: M, y: 3.05, w: 4.4 * 1350 / 1275 * 0.95, h: 4.4 * 0.95, objectName: "timeseries" });
const tx = M + 4.6;
s.addText([
  { text: "How the search works", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
  { text: "Grid over speed, lift time, lift rate, joint targets, brake delay and press torque; each combo run in many parallel envs", options: { bullet: true, breakLine: true } },
  { text: "Top combos re-validated with 150–500 randomized runs", options: { bullet: true, breakLine: true } },
  { text: "Final grid run with every uncertainty knob on, so the winner is robust by construction", options: { bullet: true, breakLine: true } },
  { text: "Key finding: the lip must be on the floor at impact. Lifting it 5 mm drops success to ~0%", options: { bullet: true, bold: true } },
], { x: tx + 0.3, y: 3.1, w: W - M - tx - 0.3, h: 3.8, fontSize: 14, color: C.text2, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Executed open loop: velocity command for the base; position + velocity targets and torque feed-forward for the arm motors. Scripts: traj_grid*.py for search, traj_validate*.py and traj_robust.py for validation.");

// ================= 7. Randomization ladder =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("We started deterministic and added randomization step by step", { placeholder: "title" });
const G = (t) => ({ text: t, options: { align: "center", bold: true, color: HEX.accent2 } });
const N = (t) => ({ text: t, options: { align: "center" } });
const ladder = [
  [HDR("#", "center"), HDR("Added"), HDR("Range / model"), HDR("Success", "center")],
  [N("0"), "Deterministic baseline (soft arm PD)", "no slip, fixed 0.1 s lag, ball exactly 1.0 m dead ahead", N("96.0%")],
  [N("1"), "Real actuators and masses (fidelity)", "MIT Kp 10/25, Kd 3, 4 N·m, gravity FF, measured link masses", N("96.0%")],
  [N("2"), "Arm joint friction", "explicit Coulomb + viscous torque", N("94.8%")],
  [N("3"), "Gravity-compensation error", "feed-forward scaled ±20%", N("96.8%")],
  [N("4"), "Action latency", "0–33 ms delay", N("95.4%")],
  [N("5"), "Ball and robot contact", "ball friction / restitution; robot friction scale", N("94.8 / 95.0%")],
  [N("6"), "Measured base slip + dynamics, then re-search", "speed ×0.9–1.1, 0.02 m/s noise, lag 0.05–0.2 s, accel 1.5 m/s²; all knobs on", G("97.6%")],
  [N("7"), "Wheel-driven base, real 4.14 kg", "wheels + exact cylinder colliders; all knobs on", G("300 / 300")],
  [N("8"), "Ball placement, finer physics", "±3 cm along / ±1 cm across, 1200 Hz", G("98.3%")],
  [N("9"), "Floor tilt", "random roll/pitch ±0.2° / ±0.3°", N("99.3 / 94.0%")],
  [N("10"), "Ball variation", "137–145 g, CoM offset ≤2 mm, radius 31–33 mm", G("100%")],
];
s.addTable(ladder, { x: M, y: 1.45, w: W - 2 * M, colW: [0.5, 4.0, 5.9, 1.733], fontSize: 13, color: HEX.dk1, fontFace: "Calibri", border: { type: "solid", pt: 0.75, color: "D5D9DC" }, rowH: 0.42, valign: "middle", objectName: "ladder-table" });
caption(s, "Rows 2–5: one knob at a time on the MIT trajectory, 500 runs each. Rows 6–10 keep all realistic knobs on; 300–500 runs (400 for row 8).", M, 6.75, W - 2 * M);
s.addNotes("The trajectory was first found with no randomness at all. Then each source of uncertainty was added one at a time to see what it costs. The only one that broke it was base slip, so we asked for measured base numbers, re-ran the grid search with everything on, and got the robust trajectory. That trajectory then held on the wheel-driven base, with random ball placement, small floor tilts and ball variation.");

// ================= 8. What could not be added =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Some randomizations could not be added, or are not modeled yet", { placeholder: "title" });
const notAdded = [
  ["Breaks the open-loop scoop", C.accent3, [
    ["Lateral / heading misalignment", "beyond ±2 cm / ±1° success collapses: an open-loop motion cannot absorb it"],
    ["Floor tilt ±1°", "61%: the ball rolls ~5 cm before contact (indoor floors are ~0.1–0.3°)"],
    ["Old base-slip model", "speed ×0.7–1.2 + drift gave 43.8%; replaced by the measured ±10%"],
  ]],
  ["Simulator cannot represent it reliably", C.accent1, [
    ["Mesh / ellipsoid ball", "convex-mesh balls sink 2–5 mm into the floor; even a round one drops to 85%"],
    ["Ball resting with off-center CoM", "random orientation makes it roll up to 10 cm to settle; placed CoM-down instead"],
    ["Turning, lateral drive", "skid-steer cylinder wheels cannot turn in place and vy is ignored (real base: mecanum)"],
  ]],
  ["Not modeled yet", C.accent6, [
    ["Torque sensing", "motor torque/current noise, bandwidth and delay: needed before torque-based control"],
    ["Net and pocket compliance", "stick and net are rigid bodies in sim, so impact is likely stiffer than real"],
    ["Yaw acceleration limit", "4 rad/s² is assumed, not measured"],
  ]],
];
notAdded.forEach(([h, col, items], i) => {
  const x = M + i * 4.1, y = 1.5, w = 3.85;
  card(s, x, y, w, 5.1);
  s.addShape(deck.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" } });
  s.addText(h, { x: x + 0.7, y: y + 0.2, w: w - 0.9, h: 0.55, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  const runs = [];
  items.forEach(([a, b], j) => {
    runs.push({ text: a, options: { bold: true, color: C.text1, breakLine: true } });
    runs.push({ text: b, options: { color: C.text2, breakLine: j < items.length - 1, paraSpaceAfter: 10 } });
  });
  s.addText(runs, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 4.0, fontSize: 13, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Three kinds of gaps. Red: real effects that the open-loop scoop cannot handle; these motivate closed-loop control. Yellow: effects PhysX does not model well, so results would be artifacts. Grey: not modeled yet; torque sensing is the one we need next.");

// ================= 9. Video: model iterations =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Watch: the scoop at each step of model fidelity", { placeholder: "title" });
[
  ["soft_slow", "1. Soft arm PD (deterministic)"],
  ["mit_slow", "2. MIT motors + real masses"],
  ["robust_slow", "3. Robust re-search, all knobs on"],
  ["wheels_slow", "4. Wheel-driven 4.14 kg base"],
].forEach(([k, l], i) => {
  const x = M + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 2.85;
  video(s, k, x + 1.0, y, 3.9, l, "12 s, 4× slow");
});
s.addNotes("Same camera for all four clips, 4× slow motion.");

// ================= 10. Robustness chart =================
deck.addSection({ title: "Results" });
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The scoop tolerates actuator and contact uncertainty", { placeholder: "title" });
s.addChart(deck.charts.BAR, [{ name: "Success", labels: ["Nominal", "Joint friction", "Grav-comp ±20%", "Latency 0–33 ms", "Ball contact", "Robot friction", "Old slip model"], values: [96.0, 94.8, 96.8, 95.4, 94.8, 95.0, 43.8] }], {
  x: M, y: 1.45, w: 8.0, h: 5.2, barDir: "col", ...chartBase("Success rate, MIT trajectory, one knob at a time (500 runs each)"),
  chartColors: [HEX.accent2, HEX.accent2, HEX.accent2, HEX.accent2, HEX.accent2, HEX.accent2, HEX.accent3],
  dataLabelFormatCode: "0.0\"%\"", objectName: "robustness-chart",
});
card(s, 9.0, 1.6, 3.7, 4.8);
s.addText([
  { text: "Reading the chart", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
  { text: "Each knob alone costs ≤1.2 points", options: { bullet: true, breakLine: true } },
  { text: "Only the old base-slip model (×0.7–1.2 speed, drift) broke it, and that model was unrealistic", options: { bullet: true, breakLine: true } },
  { text: "With the measured slip (±10%) and the robust re-search: 97.6%, then 300/300 on the wheel-driven base", options: { bullet: true, bold: true } },
], { x: 9.25, y: 1.85, w: 3.2, h: 4.4, fontSize: 14, color: C.text2, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });

// ================= 11. Alignment tolerance =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Alignment with the ball is now the main limit", { placeholder: "title" });
[
  ["Lateral offset", ["±2 cm", "±3 cm", "±4 cm"], [96.5, 45, 0]],
  ["Heading error", ["±1°", "±1.5°", "±2°"], [96.5, 82, 0]],
  ["Ball distance", ["15 cm closer", "5 cm farther", "10 cm farther"], [95, 92.5, 35]],
].forEach(([t, labels, vals], i) => {
  s.addChart(deck.charts.BAR, [{ name: t, labels, values: vals }], {
    x: M + i * 4.1, y: 1.45, w: 3.9, h: 3.9, barDir: "col", ...chartBase(t),
    chartColors: vals.map((v) => (v >= 90 ? HEX.accent2 : v >= 60 ? HEX.accent1 : HEX.accent3)),
    dataLabelFormatCode: "0\"%\"", objectName: "align-" + i,
  });
});
s.addText([
  { text: "Open loop needs the robot lined up within ~2 cm and ~1°, and the ball not much farther than planned. ", options: { bold: true, color: C.text1 } },
  { text: "A farther ball means late contact while the lift has already started; that is exactly what a contact (torque) trigger would fix." },
], { x: M, y: 5.6, w: W - 2 * M, h: 0.8, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
caption(s, "Realistic uncertainty on. Where a range was measured (e.g. 95–98%), the bar shows its midpoint.", M, 6.5, W - 2 * M);

// ================= 12. Environment variation =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Floor tilt matters, ball variation does not", { placeholder: "title" });
s.addChart(deck.charts.BAR, [{ name: "Tilt", labels: ["±0.2°", "±0.3°", "±1°"], values: [99.3, 94.0, 61.3] }], {
  x: M, y: 1.45, w: 5.6, h: 4.6, barDir: "col", ...chartBase("Random floor roll/pitch (300 runs each)"),
  chartColors: [HEX.accent2, HEX.accent2, HEX.accent3], dataLabelFormatCode: "0.0\"%\"", objectName: "tilt-chart",
});
[
  ["Tilt", "A sphere rolls ~5 cm in 0.9 s on a 1° slope. Indoor floors are typically 0.1–0.3°, inside the safe range.", C.accent1],
  ["Ball mass, CoM, size", "137–145 g, CoM offset ≤2 mm, radius 31–33 mm: 100% in every case.", C.accent2],
  ["Mesh-ball caveat", "Convex-mesh balls sink 2–5 mm into the floor; even a round mesh ball drops to 85%. A PhysX artifact, so ellipsoid results (64–78%) are not trusted.", C.accent6],
].forEach(([h, b, col], i) => {
  const y = 1.55 + i * 1.6;
  card(s, 6.8, y, 5.9, 1.4);
  s.addShape(deck.shapes.OVAL, { x: 7.05, y: y + 0.25, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" } });
  s.addText(h, { x: 7.5, y: y + 0.18, w: 5.0, h: 0.4, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText(b, { x: 7.5, y: y + 0.58, w: 5.0, h: 0.75, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
});

// ================= 13. Video: array =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Watch: 400 robots at 1200 Hz physics, 393 hold the ball", { placeholder: "title" });
video(s, "array", M, 1.45, 7.4, "10 × 10 slice of the 400-robot run, 5× slow motion", "18 s");
card(s, 8.4, 1.45, 4.3, 4.8);
s.addText([
  { text: "Setup", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
  { text: "Robust trajectory, all realistic knobs", options: { bullet: true, breakLine: true } },
  { text: "Physics 1200 Hz, control 60 Hz", options: { bullet: true, breakLine: true } },
  { text: "Ball offset up to ±3 cm along, ±1 cm across", options: { bullet: true, breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Result", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
  { text: "393 / 400 (98.3%) hold the ball", options: { bullet: true, bold: true, color: C.accent2, breakLine: true } },
  { text: "No tipping, no physics blow-ups", options: { bullet: true, breakLine: true } },
  { text: "All 7 failures: ball ends > 3 m away", options: { bullet: true } },
], { x: 8.7, y: 1.7, w: 3.8, h: 4.4, fontSize: 14, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true });

// ================= 14. Video: failures =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Watch: the 7 failures fall into two modes", { placeholder: "title" });
s.addText("A · Late contact: ball ~2 cm farther, slides off the lip", { x: M, y: 1.35, w: 6.0, h: 0.4, fontSize: 15, bold: true, color: C.accent3, margin: 0, isTextBox: true });
s.addText("B · Lifted to ~43 cm, thrown out of the pocket side", { x: M + 6.15, y: 1.35, w: 6.0, h: 0.4, fontSize: 15, bold: true, color: C.accent5, margin: 0, isTextBox: true });
// failure thumbnails use the cropped frames (caption band removed)
["f145", "f203", "f172", "f307"].forEach((k) => { LINKS[k + "x"] = LINKS[k]; });
[["f145", "Robot #145: ball 2.6 cm farther"], ["f203", "Robot #203: ball 2.2 cm farther, lifted 21 cm then lost"]].forEach(([k, l], i) => {
  fs.copyFileSync(IMG("vid/" + k + "_c.jpg"), IMG("vid/" + k + "x.jpg"));
  video(s, k + "x", M + i * 3.05, 1.9, 2.9, l, "8 s");
});
[["f172", "Robot #172"], ["f307", "Robot #307"]].forEach(([k, l], i) => {
  fs.copyFileSync(IMG("vid/" + k + "_c.jpg"), IMG("vid/" + k + "x.jpg"));
  video(s, k + "x", M + 6.15 + i * 3.05, 1.9, 2.9, l, "8 s");
});
card(s, M, 4.6, 5.95, 1.8);
s.addText("Fix idea: trigger the lift on contact (torque spike) instead of at a fixed time, so a farther ball is not met by a stick that is already rising.", { x: M + 0.25, y: 4.75, w: 5.45, h: 1.5, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
card(s, M + 6.15, 4.6, 5.95, 1.8);
s.addText("Fix idea: slow down the end of the lift, or add a final tilt that cradles the ball; detect a lost ball from the drop in static arm torque and retry.", { x: M + 6.4, y: 4.75, w: 5.45, h: 1.5, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Close-ups at 1200 Hz physics, 5× slow motion. Mode A is a timing problem caused by the ball being a little farther than planned; mode B is a containment problem after a successful pick-up.");

// ================= 15. Torque at impact (sim) =================
deck.addSection({ title: "Torque" });
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Torque" });
s.addText("In simulation, ball impact shows up clearly in the arm torque", { placeholder: "title" });
s.addImage({ path: IMG("torque_impact.png"), x: M, y: 1.4, w: 7.6, h: 7.6 * 630 / 1200, objectName: "torque-plot" });
card(s, 8.5, 1.45, 4.2, 4.9);
s.addText([
  { text: "What the sim says", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
  { text: "Impact step: shoulder −0.64 N·m, elbow +0.75 N·m, about 15–20× the approach noise", options: { bullet: true, breakLine: true } },
  { text: "It lasts only 1–2 samples at 60 Hz (≈17–33 ms), so the real robot needs ≥ 500 Hz logging", options: { bullet: true, breakLine: true } },
  { text: "Contact comes 50 ms before the scheduled lift; the lift itself causes a much larger 3.5–3.9 N·m spike, which would hide a late contact", options: { bullet: true, breakLine: true } },
  { text: "Caveat: rigid net and stick in sim, so the real peak is likely smaller and smoother", options: { bullet: true, italic: true } },
], { x: 8.75, y: 1.7, w: 3.7, h: 4.5, fontSize: 13, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
caption(s, "Applied joint torque = PD + gravity/press feed-forward, from the best wheel-driven run (one run).", M, 1.4 + 7.6 * 630 / 1200 + 0.1, 7.6);
s.addNotes("From best_run_wheels/trajectory_record.npz. The torque includes the PD response to the stick being pushed by the ball. This is one run, recorded at the control rate, so the true peak may be higher and shorter.");

// ================= 16. Is torque-based learning a good idea? =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Torque" });
s.addText("Torque as an observation: helps with timing, not with lateral aim", { placeholder: "title" });
const cols = [
  ["Where torque helps", C.accent2, [
    ["Contact timing", "Lift when the ball actually hits instead of at t = 0.9 s. Covers distance and speed errors (failure mode A; 10 cm farther is only 35% today)."],
    ["Success check", "The ball adds a static load of ~0.65 N·m at the shoulder: detect a lost ball (mode B) and retry."],
    ["Lip contact", "Press-down torque vs. position tells whether the lip is really on the floor before impact."],
  ]],
  ["Where it does not", C.accent3, [
    ["Lateral / heading error", "Both joints pitch in the same plane, so a ±3 cm side offset barely changes the torque, and by contact it is too late to correct."],
    ["Base speed error", "±10% is already absorbed by the open-loop trajectory (300/300)."],
    ["Needs instead", "Vision (camera or tag) before contact, plus lateral correction (the real base is mecanum)."],
  ]],
];
cols.forEach(([h, col, items], i) => {
  const x = M + i * 4.1, y = 1.45, w = 3.85;
  card(s, x, y, w, 5.0);
  s.addShape(deck.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" } });
  s.addText(h, { x: x + 0.7, y: y + 0.2, w: w - 0.9, h: 0.5, fontSize: 17, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  const runs = [];
  items.forEach(([a, b], j) => {
    runs.push({ text: a, options: { bold: true, color: C.text1, breakLine: true } });
    runs.push({ text: b, options: { color: C.text2, breakLine: j < items.length - 1, paraSpaceAfter: 10 } });
  });
  s.addText(runs, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: 4.0, fontSize: 13, margin: 0, valign: "top", isTextBox: true });
});
s.addShape(deck.shapes.ROUNDED_RECTANGLE, { x: 8.8, y: 1.45, w: 3.93, h: 5.0, fill: { color: C.text1 }, line: { type: "none" }, rectRadius: 0.08 });
s.addText([
  { text: "Recommendation", options: { bold: true, fontSize: 18, color: C.accent1, breakLine: true } },
  { text: "Yes, worth doing, in this order:", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "Measure the real impact torque (next week)", options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Rule-based contact trigger in sim: torque residual > threshold → start lift", options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Only if that is not enough: residual RL with a short torque history, with sensor noise and delay randomized to the measured values", options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Treat lateral aim as a separate, vision-based problem", options: { bullet: { type: "number" } } },
], { x: 9.05, y: 1.7, w: 3.45, h: 4.6, fontSize: 13, color: C.background1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
s.addNotes("Torque feedback is a good idea for one specific kind of base error: where along the path the ball is met, which covers ball distance and speed errors. It does not tell the robot about sideways or heading error, which is the biggest current limit; that needs vision before contact. A simple threshold trigger should be tried before learning, because it is easy to test on hardware and it gives a baseline for any learned policy.");

// ================= 17. Next week: measurement plan =================
s = deck.addSlide({ masterName: "CONTENT", sectionTitle: "Torque" });
s.addText("Next week: can we see the impact peak on the real motors?", { placeholder: "title" });
[
  ["Log at high rate", "Record MIT feedback (position, velocity, torque / current) at ≥ 500 Hz, with the same Kp / Kd and gravity feed-forward as in sim."],
  ["Baselines, no ball", "Static hold, then driving at 0.9 m/s with the lip pressed down: floor scraping and chassis vibration set the noise floor."],
  ["With the ball", "≈10 runs each at 0.5 and 0.9 m/s; film at high fps and sync the video with the log to timestamp the hit."],
  ["Calibrate", "Hang a known weight (the 141 g ball) in the pocket at rest to check torque-constant scale and the static ball load."],
].forEach(([h, b], i) => {
  const y = 1.45 + i * 1.22;
  numDot(s, M, y + 0.05, i + 1);
  s.addText(h, { x: M + 0.7, y, w: 6.6, h: 0.4, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  s.addText(b, { x: M + 0.7, y: y + 0.42, w: 6.6, h: 0.75, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
});
card(s, 8.2, 1.45, 4.5, 4.9);
s.addText([
  { text: "What we will compare", options: { bold: true, fontSize: 17, color: C.text1, breakLine: true } },
  { text: "Peak size vs. sim (0.6–0.75 N·m)", options: { bullet: true, breakLine: true } },
  { text: "Peak / baseline noise; suggested target ≥ 5", options: { bullet: true, breakLine: true } },
  { text: "Detection delay; must be well under the ~50 ms before the lift", options: { bullet: true, breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "What it feeds back into sim", options: { bold: true, fontSize: 17, color: C.text1, breakLine: true } },
  { text: "Torque-sensor noise, bandwidth and delay model", options: { bullet: true, breakLine: true } },
  { text: "Trigger threshold for the contact-triggered lift", options: { bullet: true, breakLine: true } },
  { text: "Whether net compliance must be modeled", options: { bullet: true } },
], { x: 8.45, y: 1.7, w: 4.0, h: 4.5, fontSize: 13, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 5, isTextBox: true });
s.addNotes("If the peak is clearly visible, the next sim step is the contact-triggered lift with a torque-noise model taken from these logs. If it is buried in noise, we look at filtering, higher Kp during the approach, or a different contact signal (e.g. velocity error).");

// ================= 18. Next directions =================
deck.addSection({ title: "Next" });
s = deck.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Next" });
s.addText("Possible next directions", { placeholder: "title" });
[
  ["Torque-triggered scoop", "Measure real impact torque, then lift on contact instead of at a fixed time. Targets failure mode A and ball-distance error."],
  ["Closed-loop alignment", "Ball perception + approach controller (mecanum lateral correction) to get within ±2 cm, ±1° before the scoop."],
  ["Residual RL with torque", "PPO corrections on top of the open-loop trajectory, with torque history as observation and measured sensor noise."],
  ["Hardware transfer", "System ID of the MIT motors and wheels, run the trajectory on the robot, and compare with sim; contain mode B failures."],
].forEach(([h, b], i) => {
  const x = M + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.55, w = 5.9;
  card(s, x, y, w, 2.3, C.text2);
  numDot(s, x + 0.3, y + 0.3, i + 1);
  s.addText(h, { x: x + 0.95, y: y + 0.3, w: w - 1.25, h: 0.46, fontSize: 18, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true });
  s.addText(b, { x: x + 0.95, y: y + 0.9, w: w - 1.25, h: 1.3, fontSize: 14, color: "DDE1E4", margin: 0, valign: "top", isTextBox: true });
});
const footer = [{ text: "Code and results: github.com/emmmum/IsaacLab, branch yifan/robomaster-lacrosse-scoop", options: {} }];
if (LINKS.folder) footer.push({ text: "   ·   All videos on Google Drive", options: { hyperlink: { url: LINKS.folder }, color: C.accent1 } });
s.addText(footer, { x: M, y: 6.75, w: W - 2 * M - 0.6, h: 0.35, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
s.addNotes("Item 1 starts next week with the torque measurement. Items 1 and 2 address the two failure types we see in sim (timing and aim). Residual RL is only needed if simple rules are not enough.");

(async () => {
  const out = path.join(__dirname, "robomaster_lacrosse_scoop_progress_v2.pptx");
  await deck.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
