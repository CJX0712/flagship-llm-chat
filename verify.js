/*
 * verify.js · 一键聚合自检（可复现闭环）
 * 跑：apex-engine 9/9 + apex-chat 3/3 共 12 项不变量，全部通过则退出码 0
 */
const E = require('./apex-engine.js');
const CHAT = require('./apex-chat.js');

const eng = E.runSelfTest();
const chat = CHAT.runSelfTest();
const all = eng.concat(chat);

let pass = 0;
console.log('\n=== APEX-CHAT 聚合自检 ===');
all.forEach((x) => {
  console.log(`${x.pass ? 'PASS' : 'FAIL'} ${x.name.padEnd(22)} | ${x.detail}`);
  if (x.pass) pass++;
});
console.log('------------------------------------------');
console.log(`${pass}/${all.length} 全绿`);
process.exit(pass === all.length ? 0 : 1);
