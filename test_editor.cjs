const assert = require('node:assert/strict');
const fs = require('node:fs');
const E = require('./workbench_ui/editor.js');
const original = JSON.parse(fs.readFileSync('outputs/automatic/onset_refined/annotations/episode_000001.json','utf8'));
function valid(d) {let cursor=0;for(const s of d.subtasks){assert.equal(s.start_frame,cursor);assert.ok(s.end_frame>=cursor);cursor=s.end_frame+1;}assert.equal(cursor,394);}
for(let cut=13;cut<=original.subtasks[2].end_frame;cut++){
  const d=E.boundary(original,2,'start',cut);valid(d);assert.equal(d.subtasks[1].end_frame,cut-1);
}
for(let i=0;i<original.subtasks.length;i++){
  const s=original.subtasks[i];
  for(let f=s.start_frame+1;f<=s.end_frame;f++){
    const d=E.split(original,i,f);valid(d);assert.equal(d.subtasks.length,8);
    const merged=E.merge(d,i);valid(merged);assert.deepEqual(E.normalize(merged,15),original);
  }
}
assert.throws(()=>E.boundary(original,0,'start',1));
assert.throws(()=>E.boundary(original,2,'start',12));
assert.throws(()=>E.split(original,0,0));
assert.throws(()=>E.merge(original,6));
console.log('Editor: exhaustive split/merge roundtrips and boundary coverage passed');
