/* Small helpers shared by both pages. No dependencies. */
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, k.startsWith("aria-") ? "true" : "");
    else if (v === false && k.startsWith("aria-")) n.setAttribute(k, "false");
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) n.append(k);
  return n;
}

const api = (path, opts) => fetch(path, opts).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
const LANG = { en: "EN", fr: "FR", ar: "AR", es: "ES" };
const fmtSize = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
const today = () => new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });

/* Tabs: click, or arrow keys while a tab has focus. onChange(view) runs after a switch. */
function initTabs(onChange) {
  const tabs = [...document.querySelectorAll(".tabs button")];
  tabs.forEach((b) => {
    b.tabIndex = b.getAttribute("aria-selected") === "true" ? 0 : -1;
    b.addEventListener("click", () => {
      tabs.forEach((x) => { x.setAttribute("aria-selected", x === b); x.tabIndex = x === b ? 0 : -1; });
      document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== "view-" + b.dataset.view));
      onChange && onChange(b.dataset.view);
    });
    b.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      const next = tabs[(tabs.indexOf(b) + step + tabs.length) % tabs.length];
      next.focus(); next.click();
    });
  });
}

/* Online / offline indicator for the status area. */
function initNet(dotEl, textEl) {
  const set = () => { const on = navigator.onLine; dotEl.className = "dot" + (on ? "" : " off"); textEl.textContent = on ? "Online" : "Offline"; };
  addEventListener("online", set); addEventListener("offline", set); set();
}

const CRISIS = { "cholera": "Cholera", "ebola": "Ebola", "measles": "Measles", "malaria": "Malaria", "tuberculosis": "Tuberculosis", "hiv": "HIV/AIDS",
  "malnutrition": "Malnutrition", "covid-19": "COVID-19", "other-epidemic": "Other epidemic", "conflict": "Armed conflict", "displacement": "Displacement",
  "natural-disaster": "Natural disaster", "water-sanitation": "Water and sanitation", "sexual-violence": "Sexual violence", "maternal-health": "Maternal health",
  "mental-health": "Mental health", "access-to-medicines": "Access to medicines" };
// PDF text keeps the printed line breaks: join single breaks, keep paragraph breaks, drop "|" paragraph marks.
const reflow = (t) => t.replace(/^\s*\|\s?/gm, "").replace(/[ \t]*\n(?![ \t]*\n)[ \t]*/g, " ").replace(/\n{3,}/g, "\n\n").trim();
const crisisLabel = (t) => CRISIS[t] || t;

/* Dropdown (replaces <select>) and combobox (replaces <input list>). Both keyboard-operable, with ARIA. */
let ddSeq = 0;
function closeAll(except) { document.querySelectorAll(".dd.open").forEach((d) => { if (d !== except) d._close(); }); }
document.addEventListener("click", (e) => { if (!e.target.closest(".dd")) closeAll(); });

function menu(root, trigger, getItems, pick) {
  const ul = el("ul", { class: "dd-menu", role: "listbox", id: "dd" + ++ddSeq, hidden: true });
  root.append(ul); trigger.setAttribute("aria-controls", ul.id);
  let active = -1, items = [];
  const paint = () => [...ul.children].forEach((li, i) => li.dataset.active = i === active);
  const render = (selected) => {
    items = getItems();
    ul.replaceChildren(...(items.length ? items.map((it, i) => el("li", { role: "option", "aria-selected": it.value === selected, onmousedown: (e) => e.preventDefault(), onclick: () => pick(it), onmousemove: () => { active = i; paint(); } }, it.text))
      : [el("li", { class: "none", text: "No match" })]));
  };
  const api = {
    ul, render,
    open(selected) { closeAll(root); render(selected); ul.hidden = false; root.classList.add("open"); trigger.setAttribute("aria-expanded", "true");
      active = Math.max(0, items.findIndex((i) => i.value === selected)); paint(); ul.children[active]?.scrollIntoView({ block: "nearest" }); },
    close() { ul.hidden = true; root.classList.remove("open"); trigger.setAttribute("aria-expanded", "false"); active = -1; },
    move(d) { if (!items.length) return; active = (active + d + items.length) % items.length; paint(); ul.children[active].scrollIntoView({ block: "nearest" }); },
    choose() { if (active >= 0 && items[active]) { pick(items[active]); return true; } return false; },
    get isOpen() { return !ul.hidden; },
  };
  root._close = api.close;
  return api;
}

function dropdown({ label, options = [], value = "", placeholder = "Select", onChange }) {
  const root = el("div", { class: "dd" });
  let opts = options, val = value;
  const btn = el("button", { class: "dd-btn", type: "button", "aria-haspopup": "listbox", "aria-expanded": "false", "aria-label": label });
  root.append(btn);
  const show = () => { const o = opts.find((x) => x.value === val); btn.replaceChildren(o ? o.text : el("span", { class: "ph", text: placeholder })); };
  const m = menu(root, btn, () => opts, (it) => { val = it.value; show(); m.close(); btn.focus(); onChange && onChange(val); });
  btn.addEventListener("click", () => (m.isOpen ? m.close() : m.open(val)));
  btn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); m.isOpen ? m.move(e.key === "ArrowDown" ? 1 : -1) : m.open(val); }
    else if ((e.key === "Enter" || e.key === " ") && m.isOpen) { e.preventDefault(); m.choose(); }
    else if (e.key === "Escape" && m.isOpen) { e.preventDefault(); m.close(); }
    else if (e.key === "Tab") m.close();
  });
  show();
  return { el: root, get value() { return val; }, set(o) { opts = o; if (!opts.some((x) => x.value === val)) val = opts[0] ? opts[0].value : ""; show(); } };
}

function combobox({ label, options = [], placeholder = "", onSelect, onClear }) {
  const root = el("div", { class: "dd combo" });
  let opts = options;
  const input = el("input", { type: "text", role: "combobox", autocomplete: "off", spellcheck: "false", placeholder, "aria-label": label, "aria-expanded": "false", "aria-autocomplete": "list" });
  root.append(input);
  const match = () => { const q = input.value.trim().toLowerCase(); if (!q) return opts;
    return [...opts.filter((o) => o.text.toLowerCase().startsWith(q)), ...opts.filter((o) => !o.text.toLowerCase().startsWith(q) && o.text.toLowerCase().includes(q))]; };
  const m = menu(root, input, match, (it) => { input.value = it.text; m.close(); onSelect && onSelect(it.value); });
  input.addEventListener("focus", () => m.open());
  input.addEventListener("input", () => { m.open(); if (!input.value.trim() && onClear) onClear(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); m.isOpen ? m.move(e.key === "ArrowDown" ? 1 : -1) : m.open(); }
    else if (e.key === "Enter") { e.preventDefault(); if (!m.isOpen) m.open(); else if (!m.choose()) { const f = match()[0]; if (f) { input.value = f.text; m.close(); onSelect && onSelect(f.value); } } }
    else if (e.key === "Escape") m.close();
    else if (e.key === "Tab") m.close();
  });
  return { el: root, input, set(o) { opts = o; }, setText(t) { input.value = t; } };
}
