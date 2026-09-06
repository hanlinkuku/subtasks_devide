/* Pure editing operations: every successful change preserves inclusive coverage. */
(function (root) {
  const clone = x => JSON.parse(JSON.stringify(x));
  function normalize(doc, fps) {
    doc.subtasks.forEach((s, i) => {s.id = i + 1; s.start_time = +(s.start_frame / fps).toFixed(6); s.end_time = +(s.end_frame / fps).toFixed(6);});
    return doc;
  }
  function boundary(doc, index, edge, value) {
    const d = clone(doc), s = d.subtasks[index];
    if (!Number.isInteger(value)) throw Error('请输入整数帧号');
    if (edge === 'start') {
      if (index === 0) {if (value !== d.trajectory_start) throw Error('第一段的开始帧固定'); return d;}
      const previous = d.subtasks[index - 1];
      if (value <= previous.start_frame || value > s.end_frame) throw Error('边界调整不能使相邻子任务为空；可先合并');
      s.start_frame = value; previous.end_frame = value - 1;
    } else {
      if (index === d.subtasks.length - 1) {if (value !== d.trajectory_end) throw Error('最后一段的结束帧固定'); return d;}
      const next = d.subtasks[index + 1];
      if (value < s.start_frame || value >= next.end_frame) throw Error('边界调整不能使相邻子任务为空；可先合并');
      s.end_frame = value; next.start_frame = value + 1;
    }
    return d;
  }
  function split(doc, index, frame) {
    const d = clone(doc), s = d.subtasks[index];
    if (!Number.isInteger(frame) || frame <= s.start_frame || frame > s.end_frame) throw Error('当前帧须位于选中段内部，且不能是开始帧');
    const next = clone(s); next.start_frame = frame; s.end_frame = frame - 1;
    d.subtasks.splice(index + 1, 0, next); return d;
  }
  function merge(doc, index) {
    const d = clone(doc), s = d.subtasks[index], next = d.subtasks[index + 1];
    if (!next) throw Error('最后一段没有可合并的下一段');
    s.end_frame = next.end_frame;
    if (next.notes && next.notes !== s.notes) s.notes += '\n合并段备注：' + next.notes;
    d.subtasks.splice(index + 1, 1); return d;
  }
  root.Editor = {clone, normalize, boundary, split, merge};
  if (typeof module !== 'undefined') module.exports = root.Editor;
})(globalThis);
