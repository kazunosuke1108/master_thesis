"use strict";
// Local prototype only: no requests, uploads, or automatic answer storage.
const config = JSON.parse(document.getElementById("questionnaire-config").textContent);
let startedAt = null, startedTick = null, completed = null;
const byId = id => document.getElementById(id);
const elements = selector => Array.from(document.querySelectorAll(selector));

function ratio(text) {
  text = text.trim();
  if (!/^(?:\d+(?:\.\d+)?|\.\d+)(?:\s*\/\s*(?:\d+(?:\.\d+)?|\.\d+))?$/.test(text)) throw Error("一対比較は分数または小数で入力してください。");
  const parts = text.split("/").map(Number);
  const value = parts.length === 2 ? parts[0] / parts[1] : parts[0];
  if (!Number.isFinite(value) || value < 1/9 || value > 9) throw Error("一対比較は1/9〜9の範囲です。");
  return value;
}

function start() {
  if (startedAt !== null) return;
  startedAt = new Date().toISOString();
  startedTick = performance.now();
  byId("answers").disabled = false;
  byId("start").disabled = true;
  byId("timer").textContent = "回答中（説明を読む時間・休憩を含む）";
}

function toggleUnable(box) {
  const section = box.closest(".case");
  section.querySelectorAll(".ranking").forEach(input => {
    input.disabled = box.checked;
    input.required = !box.checked;
  });
  section.querySelector(".reason").required = box.checked;
}

function collect() {
  if (startedAt === null) throw Error("始めるボタンを押してください。");
  const respondent = byId("respondent").value.trim();
  const kind = byId("kind").value;
  if (!/^[A-Za-z0-9_-]+$/.test(respondent)) throw Error("匿名回答者IDを英数字・ハイフン・アンダースコアで記入してください。");
  if (!["synthetic_ai", "human_trial"].includes(kind)) throw Error("回答の種類を選択してください。");
  const matrices = {};
  for (const [group, labels] of Object.entries(config.groups)) {
    const n = labels.length;
    const matrix = Array.from({length:n}, (_, i) => Array.from({length:n}, (_, j) => i === j ? 1 : null));
    elements(`.pairwise[data-group="${group}"]`).forEach(input => {
      const i = Number(input.dataset.i), j = Number(input.dataset.j), value = ratio(input.value);
      matrix[i][j] = value; matrix[j][i] = 1/value;
    });
    if (matrix.some(row => row.some(value => value === null))) throw Error("一対比較に未回答があります。");
    matrices[group] = matrix;
  }
  const fuzzy = {};
  for (const group of ["internal", "external", "total"]) {
    fuzzy[group] = elements(`.fuzzy[data-group="${group}"]`).map(input => Number(input.value));
    if (fuzzy[group].length !== 4 || fuzzy[group].some(value => !Number.isInteger(value) || value < 1 || value > 5)) throw Error("組合せ判断12問をすべて回答してください。");
  }
  const responses = [];
  for (const item of config.cases) {
    const box = elements(".unable").find(input => input.dataset.case === item.case_id);
    const section = box.closest(".case");
    const reason = section.querySelector(".reason").value.trim();
    const inputs = Array.from(section.querySelectorAll(".ranking"));
    if (box.checked && !reason) throw Error(`${item.case_id}: 判断不能の理由を記入してください。`);
    const ranks = inputs.map(input => Number(input.value));
    if (!box.checked && ranks.some(rank => !Number.isInteger(rank) || rank < 1)) throw Error(`${item.case_id}: 全患者の順位を記入してください。`);
    if (!box.checked && ranks.some(rank => rank !== 1 + ranks.filter(other => other < rank).length)) throw Error(`${item.case_id}: 同順位は1,1,3の形式です。`);
    inputs.forEach((input, i) => responses.push({case_id:item.case_id, respondent_id:respondent, patient_id:input.dataset.patient, rank:box.checked ? "" : ranks[i], status:box.checked ? "unable" : "ranked", reason}));
  }
  return {schema_version:"REQ003-full-session-trial-v1", prototype:true, data_kind:kind,
    presentation_sha256:config.presentation_sha256,
    respondent_id:respondent, synthetic:kind === "synthetic_ai", action_pairwise:matrices.action,
    object_pairwise:matrices.object, knowledge_item_order:config.groups,
    fuzzy_order:["HH","HL","LH","LL"], fuzzy_answers:fuzzy, responses};
}

function finish() {
  if (completed) return completed;
  try {
    const data = collect();
    const elapsed = performance.now() - startedTick;
    completed = {...data, timing:{started_at:startedAt, completed_at:new Date().toISOString(), elapsed_ms:elapsed, includes_reading_and_breaks:true, target_ms:300000, target_achieved:elapsed <= 300000}};
    byId("answers").disabled = true;
    byId("downloads").hidden = false;
    byId("error").textContent = "";
    byId("timer").textContent = `完了：${(elapsed/1000).toFixed(1)}秒（実測・この試行のみ）`;
    return completed;
  } catch (error) {
    byId("error").textContent = error.message;
    return null;
  }
}

function csv(rows, fields) {
  const quote = value => '"' + String(value ?? "").replaceAll('"', '""') + '"';
  return fields.join(",") + "\r\n" + rows.map(row => fields.map(field => quote(row[field])).join(",")).join("\r\n") + "\r\n";
}

function knowledgeRows(data) {
  const rows = [];
  for (const [group, matrix] of [["action",data.action_pairwise],["object",data.object_pairwise]]) {
    for (let i=0; i<matrix.length; i++) for (let j=i+1; j<matrix.length; j++) rows.push({group,item:`${i},${j}`,answer:matrix[i][j]});
  }
  for (const group of ["internal","external","total"]) data.fuzzy_answers[group].forEach((answer,i) => rows.push({group,item:data.fuzzy_order[i],answer}));
  return rows.map(row => ({respondent_id:data.respondent_id,data_kind:data.data_kind,...row,started_at:data.timing.started_at,completed_at:data.timing.completed_at,elapsed_ms:data.timing.elapsed_ms}));
}

function download(kind) {
  if (!completed) return;
  let text, suffix, mime;
  if (kind === "json") {text = JSON.stringify(completed,null,2); suffix="session.json"; mime="application/json";}
  else if (kind === "ranks") {text=csv(completed.responses,["case_id","respondent_id","patient_id","rank","status","reason"]);suffix="responses.csv";mime="text/csv";}
  else {text=csv(knowledgeRows(completed),["respondent_id","data_kind","group","item","answer","started_at","completed_at","elapsed_ms"]);suffix="knowledge.csv";mime="text/csv";}
  const url = URL.createObjectURL(new Blob([text],{type:mime+";charset=utf-8"}));
  const anchor = document.createElement("a"); anchor.href=url; anchor.download=completed.respondent_id+"-"+suffix; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

byId("questionnaire").addEventListener("submit",event => event.preventDefault());
byId("start").addEventListener("click",start);
byId("finish").addEventListener("click",finish);
elements(".unable").forEach(input => input.addEventListener("change",() => toggleUnable(input)));
for (const kind of ["json","ranks","knowledge"]) byId(kind).addEventListener("click",() => download(kind));
// Public helpers also support deterministic browser verification of exports.
window.questionnaireTrial = {collect,finish,ratio,csv,knowledgeRows,getResult:() => completed};
