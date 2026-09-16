/**
 * Pure-JS forward pass for the lacrosse_scoop PPO policy (dense_v16 / best_agent.pt).
 * No dependencies. Load lacrosse_scoop_policy.json and call runPolicy(obs, model).
 *
 * Network: Linear(11,256) -> ELU -> Linear(256,128) -> ELU -> Linear(128,3)  [policy head]
 *                                                            -> Linear(128,1) [value head]
 * Weight matrices are stored PyTorch-style: shape (out_features, in_features),
 * so y = W · x + b (row i of W dotted with x gives output i) — NOT x · W.
 */

function elu(x) {
  return x > 0 ? x : Math.exp(x) - 1;
}

/** y = W . x + b, where W is [out][in], x is [in], b is [out]. Returns [out]. */
function linear(x, W, b) {
  const out = new Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let sum = b[i];
    for (let j = 0; j < row.length; j++) sum += row[j] * x[j];
    out[i] = sum;
  }
  return out;
}

/**
 * Normalize a raw 11-dim observation the same way skrl's RunningStandardScaler
 * does at inference time: clip((x - mean) / (sqrt(var) + eps), -clip, clip).
 */
function normalizeObs(obs, model) {
  const { mean, var: variance } = model.obs_normalization;
  const eps = model.obs_normalization.eps;
  const clip = model.obs_normalization.clip;
  return obs.map((v, i) => {
    const n = (v - mean[i]) / (Math.sqrt(variance[i]) + eps);
    return Math.max(-clip, Math.min(clip, n));
  });
}

/**
 * Full forward pass. `obs` is a raw (un-normalized) 11-dim array in the order
 * given by model.obs_labels. Returns { action: [3], value: number, hidden: {h1, h2} }.
 * `action` is the deterministic (mean) action, clipped to [-1, 1] — this is what
 * the trained policy actually outputs at evaluation time (no sampling).
 */
function runPolicy(obs, model) {
  const xn = normalizeObs(obs, model);

  const h1 = linear(xn, model.layers.fc1.weight, model.layers.fc1.bias).map(elu);
  const h2 = linear(h1, model.layers.fc2.weight, model.layers.fc2.bias).map(elu);

  const rawAction = linear(h2, model.layers.policy_head.weight, model.layers.policy_head.bias);
  const action = rawAction.map((a) => Math.max(-1, Math.min(1, a)));

  const value = linear(h2, model.layers.value_head.weight, model.layers.value_head.bias)[0];

  return { action, value, hidden: { h1, h2 } };
}

// Example (browser):
//   const model = await fetch('lacrosse_scoop_policy.json').then(r => r.json());
//   const obs = [0, 0, 0, 0, 0, 0, -0.9, 0.6, 0, 0, 0]; // ball 0.9m ahead, 0.6m above pocket
//   const { action, value } = runPolicy(obs, model);
//   console.log(model.action_labels.map((name, i) => `${name}: ${action[i].toFixed(3)}`));

if (typeof module !== 'undefined') {
  module.exports = { runPolicy, normalizeObs, linear, elu };
}
