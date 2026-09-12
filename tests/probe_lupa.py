"""Probe lupa/LuaJIT interop semantics that the Aegisub API shim depends on."""

from __future__ import annotations

import json

import lupa.luajit21 as lupa

r = lupa.LuaRuntime(unpack_returned_tuples=True)
g = r.globals()


class Line:
    """Stand-in for the real line proxy (attribute access both ways)."""

    def __init__(self, text="", start=0):
        self._text = text
        self._start = start

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def start_time(self):
        return self._start

    @start_time.setter
    def start_time(self, value):
        self._start = value

    def __str__(self):
        return f"<Line {self._text!r}>"

    __repr__ = __str__


class Handle:
    """Document handle: the Python side of the ``subs`` table."""

    def __init__(self, n=3):
        self.lines = [Line(f"l{i}", i * 100) for i in range(1, n + 1)]
        self.ops: list[tuple] = []

    def count(self):
        return len(self.lines)

    def get(self, i):
        return self.lines[int(i) - 1]

    def set(self, i, table):
        self.ops.append(("set", int(i), to_pydict(table)))
        self.lines[int(i) - 1] = Line(to_pydict(table).get("text", ""))

    def append(self, table):
        self.ops.append(("append", to_pydict(table)))
        self.lines.append(Line(to_pydict(table).get("text", "")))

    def insert(self, i, table):
        self.ops.append(("insert", int(i), to_pydict(table)))
        self.lines.insert(int(i) - 1, Line(to_pydict(table).get("text", "")))

    def delete(self, i, count=1):
        self.ops.append(("delete", int(i), int(count)))
        del self.lines[int(i) - 1:int(i) - 1 + int(count)]


def to_pydict(value):
    """Convert a Lua table (or Python object) coming from Lua into a dict."""
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(k): to_pyvalue(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): to_pyvalue(v) for k, v in value.items()}
    out = {}
    try:
        for k in value.keys():
            out[str(k)] = to_pyvalue(value[k])
    except Exception:
        pass
    return out


def to_pyvalue(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "items"):
        return {str(k): to_pyvalue(v) for k, v in value.items()}
    return value


SUBS_BUILDER = """
function __build_subs(handle)
  local t = {}
  local n = handle:count()
  for i = 1, n do t[i] = handle:get(i) end
  local mt
  mt = {
    __len = function() return handle:count() end,
    __newindex = function(tbl, k, v)
      if type(k) == "number" then
        handle:set(k, v)
        rawset(tbl, k, handle:get(k))
      else
        rawset(tbl, k, v)
      end
    end,
    __index = function(tbl, k)
      if type(k) == "number" then return handle:get(k) end
      return nil
    end,
  }
  t.append = function(line) handle:append(line); rawset(t, handle:count(), handle:get(handle:count())) end
  t.insert = function(idx, line) handle:insert(idx, line); table.insert(t, idx, handle:get(idx)) end
  t.delete = function(idx, count)
    count = count or 1
    handle:delete(idx, count)
    for _ = 1, count do table.remove(t, idx) end
  end
  return setmetatable(t, mt)
end
"""

r.execute(SUBS_BUILDER)

handle = Handle(3)
subs = g.__build_subs(handle)
g.subs = subs

checks = []
checks.append(("len(#subs)", r.eval("#subs")))
checks.append(("subs[1].text", r.eval("subs[1].text")))
checks.append(("subs[1].start_time", r.eval("subs[1].start_time")))
checks.append(("tostring(subs[2])", r.eval("tostring(subs[2])")))
r.execute("subs[1].text = 'changed'")
checks.append(("attr assign -> py", handle.lines[0].text))
checks.append(("attr assign -> lua", r.eval("subs[1].text")))
checks.append(("ipairs", r.eval(
    "(function() local t={} for i,v in ipairs(subs) do t[#t+1]=i..v.text end return table.concat(t,',') end)()")))
checks.append(("pairs-ish loop 1..#subs", r.eval(
    "(function() local t={} for i=1,#subs do t[#t+1]=subs[i].text end return table.concat(t,'|') end)()")))
r.execute("subs.append({text='appended', start_time=10})")
checks.append(("after append: #subs", r.eval("#subs")))
checks.append(("after append: last text", r.eval("subs[#subs].text")))
checks.append(("after append: py ops", handle.ops[-1]))
r.execute("subs.insert(2, {text='inserted'})")
checks.append(("after insert: #subs", r.eval("#subs")))
checks.append(("after insert: subs[2].text", r.eval("subs[2].text")))
checks.append(("after insert: subs[3].text", r.eval("subs[3].text")))
r.execute("subs.delete(1, 2)")
checks.append(("after delete: #subs", r.eval("#subs")))
checks.append(("after delete: subs[1].text", r.eval("subs[1].text")))
r.execute("subs[1] = {text='replaced'}")
checks.append(("after replace: subs[1].text", r.eval("subs[1].text")))
checks.append(("after replace: py", [ln.text for ln in handle.lines]))

# Lua table -> Python callable
def take_table(tbl):
    return json.dumps(to_pydict(tbl), sort_keys=True)


def take_line(line):
    return json.dumps({"text": getattr(line, "text", None), "start": getattr(line, "start_time", None)})


g.take_table = take_table
g.take_line = take_line
checks.append(("callback gets lua table", r.eval("take_table({text='x', start_time=5, extra={a=1}})")))
checks.append(("callback gets line proxy", r.eval("take_line(subs[1])")))

# errors: Python -> Lua -> Python
def boom(msg):
    raise ValueError(f"boom: {msg}")


g.boom = boom
checks.append(("lua pcall of py error", r.eval(
    "(function() local ok, err = pcall(boom, 'hi') return tostring(ok)..'|'..tostring(err) end)()")[:100]))
r.execute("function luafunc() error('lua-side failure') end")
try:
    g.luafunc()
except Exception as exc:  # noqa: BLE001
    checks.append(("lua error -> py", f"{type(exc).__name__}: {str(exc)[:50]}"))

# macro function pattern: Lua function returned to Python, called with args
r.execute("function macrofn(subs, sel) subs[1].text = 'from macro' return 'ok', #sel end")
fn = g.macrofn
try:
    checks.append(("macro return tuple", fn(subs, r.table_from([1, 2]))))
except Exception as exc:  # noqa: BLE001
    checks.append(("macro call failed", f"{type(exc).__name__}: {str(exc)[:80]}"))
try:
    checks.append(("macro return w/ py list sel", fn(subs, [1, 2])))
except Exception as exc:  # noqa: BLE001
    checks.append(("py list sel rejected", str(exc)[:60]))
checks.append(("macro mutated py", handle.lines[0].text))

# configuration table round-trip
r.execute("config = {foo='bar', n=1}")
cfg = g.config
cfg["foo"] = "baz"
checks.append(("py -> lua table", r.eval("config.foo")))

# os/io/math/string availability + Aegisub-ish modules
for name in ("os", "io", "math", "string", "table", "bit", "jit", "utf8", "debug", "package"):
    checks.append((f"lib {name}", r.eval(f"type({name})")))
checks.append(("jit.version", r.eval("jit and jit.version or 'none'")))
checks.append(("os.time", r.eval("type(os.time)")))
checks.append(("io.open", r.eval("type(io.open)")))
checks.append(("string.format ok", r.eval("string.format('%.2f', 1.5)")))
try:
    checks.append(("lua string.sub utf8 naive", r.eval("string.sub('กาน', 1, 1)")))
except Exception as exc:  # noqa: BLE001
    checks.append(("lua string.sub utf8 naive", f"{type(exc).__name__}: {str(exc)[:40]}"))
checks.append(("# operator on string utf8", r.eval("#'กาน'")))

# --- byte-transparent (latin-1) mode: the design we will actually use -------
r2 = lupa.LuaRuntime(encoding="latin-1", unpack_returned_tuples=True)


def py_to_lua_bytes(text):
    return text.encode("utf-8").decode("latin-1")


def lua_to_py_bytes(text):
    if isinstance(text, str):
        return text.encode("latin-1").decode("utf-8", "surrogateescape")
    return text


thai = "กาน"
r2.globals().t = py_to_lua_bytes(thai)
checks.append(("latin1: round-trip", lua_to_py_bytes(r2.eval("t")) == thai))
checks.append(("latin1: byte len", r2.eval("#t")))
checks.append(("latin1: naive sub round-trip",
               lua_to_py_bytes(r2.eval("string.sub(t, 1, 3)")) == "ก"))
checks.append(("latin1: split mid-char is recoverable",
               lua_to_py_bytes(r2.eval("string.sub(t, 1, 2)")).encode("utf-8", "surrogateescape")[:2].hex()))
r2.globals().back = py_to_lua_bytes("สวัสดี &#1; ok")
checks.append(("latin1: full round-trip mixed", lua_to_py_bytes(r2.eval("back"))))
checks.append(("latin1: string.format", lua_to_py_bytes(r2.eval("string.format('%s!', back)"))))
# methods on a Python object returning byte-transparent strings
class Echo:
    def get(self):
        return py_to_lua_bytes("ก")


r2.globals().echo = Echo()
checks.append(("latin1: obj return", lua_to_py_bytes(r2.eval("echo:get()"))))

for name, value in checks:
    print(f"{name:34} = {value!r}")
