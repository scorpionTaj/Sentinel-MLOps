import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class FakeElement {
  constructor() {
    this.className = "";
    this.disabled = false;
    this.innerHTML = "";
    this.style = {};
    this.textContent = "";
  }

  addEventListener() {}
}

const elements = new Map();
const element = (selector) => {
  if (!elements.has(selector)) elements.set(selector, new FakeElement());
  return elements.get(selector);
};

let fetchCalls = 0;
const context = {
  clearTimeout() {},
  document: { querySelector: element },
  fetch: async () => {
    fetchCalls += 1;
    throw new Error("file mode must not fetch");
  },
  location: { protocol: "file:" },
  setTimeout() { return 1; },
};

const source = fs.readFileSync(new URL("../src/sentinel/web/app.js", import.meta.url), "utf8");
vm.runInNewContext(source, context, { filename: "app.js" });
await new Promise((resolve) => setImmediate(resolve));

assert.equal(fetchCalls, 0, "file mode attempted a network request");
assert.match(
  element("#connection").textContent,
  /http:\/\/localhost:8000/,
  "file mode did not explain how to open the served dashboard",
);

console.log("file-mode guard: pass");
