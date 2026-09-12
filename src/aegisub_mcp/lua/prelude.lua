--[[
  aegisub-mcp Lua prelude — runs inside a LuaJIT 2.1 state (the same VM family
  Aegisub's Automation 4 uses), hosted by lupa in "byte transparent" mode:
  every Python string handed to Lua is UTF-8 bytes reinterpreted as latin-1 and
  every string handed back is re-decoded, so Lua's byte string semantics (and
  `string.sub`) behave exactly like they do inside Aegisub.

  Everything pure-Lua lives here:

    * aegisub.util     copy / clamp / colour helpers / interpolation
    * aegisub.re       regex, backed by Python's `re` through __host_re_*
    * unicode          Automation 4 unicode module (__host_unicode_*)
    * aegisub.progress title/task/set/is_cancelled (+ printf formatting)
    * aegisub.debug.out / aegisub.log (level + printf formatting)
    * include / require (automation include directories)
    * __build_subs     the `subtitles` object over the host document

  Host callables injected by the Python engine are prefixed `__host_`.
  Anything an Aegisub script calls that aegisub-mcp does not implement raises a
  descriptive Lua error instead of a confusing "attempt to call a nil value".
]]

local host = __host
if type(host) ~= "table" then
  error("aegisub-mcp prelude: __host table missing (engine did not install it)")
end

aegisub = aegisub or {}

----------------------------------------------------------------- misc helpers

local function printf(...)
  local n = select("#", ...)
  if n == 0 then return "" end
  if n == 1 then return tostring((...)) end
  local ok, out = pcall(string.format, ...)
  if ok then return out end
  -- A broken format string must not kill the macro: join the arguments.
  local parts = {}
  for i = 1, n do parts[i] = tostring((select(i, ...))) end
  return table.concat(parts, " ")
end

------------------------------------------------------------------ progress

local progress = {}
aegisub.progress = progress

function progress.title(...)
  host.__host_progress("title", printf(...))
end

function progress.task(...)
  host.__host_progress("task", printf(...))
end

function progress.set(percent)
  host.__host_progress("set", tonumber(percent) or 0)
end

function progress.is_cancelled()
  return host.__host_is_cancelled()
end

function aegisub.cancel()
  host.__host_cancel()
end

--------------------------------------------------------------------- logging

local debug_t = {}
aegisub.debug = debug_t

function debug_t.out(...)
  local n = select("#", ...)
  if n == 0 then return end
  local first = (select(1, ...))
  local level, rest
  if type(first) == "number" then
    level = first
    rest = { select(2, ...) }
  else
    level = 0
    rest = { ... }
  end
  host.__host_log(level, printf(unpack(rest)))
end

aegisub.log = debug_t.out

function aegisub.set_undo_point(description)
  host.__host_undo_point(tostring(description or ""))
end

----------------------------------------------------------- module loading ---

local util = {}
aegisub.util = util

local function clamp01(x)
  if x < 0 then return 0 elseif x > 1 then return 1 else return x end
end

function util.copy(t)
  local out = {}
  for k, v in pairs(t) do out[k] = v end
  return out
end

function util.deep_copy(t)
  if type(t) ~= "table" then return t end
  local out = {}
  for k, v in pairs(t) do
    if type(v) == "table" then out[k] = util.deep_copy(v) else out[k] = v end
  end
  return out
end

function util.clamp(value, min, max)
  if value < min then return min end
  if value > max then return max end
  return value
end

function util.interpolate(p, delta, t)
  return p * (1 - t) + delta * t
end

-- ASS colours are &HAABBGGRR with an inverted alpha (00 = opaque).
function util.extract_color(str)
  local rest = tostring(str)
  local amp = rest:find("&")
  if amp then rest = rest:sub(amp + 1) end
  local amp2 = rest:find("&")
  if amp2 then rest = rest:sub(1, amp2 - 1) end
  if rest:sub(1, 1):lower() == "h" then rest = rest:sub(2) end
  local value = tonumber(rest, 16) or 0
  return {
    r = value % 256,
    g = math.floor(value / 256) % 256,
    b = math.floor(value / 65536) % 256,
    a = math.floor(value / 16777216) % 256,
  }
end

function util.ass_color(r, g, b)
  return string.format("&H%02X%02X%02X&", b, g, r)
end

function util.ass_style_color(r, g, b, a)
  a = a or 0
  return string.format("&H%02X%02X%02X%02X&", a, b, g, r)
end

function util.ass_alpha(a)
  return string.format("&H%02X&", a)
end

function util.alpha_from_style(str)
  return util.extract_color(str).a
end

function util.color_from_style(str)
  local c = util.extract_color(str)
  return c.r, c.g, c.b
end

function util.interpolate_color(t, color1, color2, t2)
  if t2 == nil then t2 = t end
  local c1 = util.extract_color(color1)
  local c2 = util.extract_color(color2)
  local r = util.interpolate(c1.r, c2.r, t)
  local g = util.interpolate(c1.g, c2.g, t)
  local b = util.interpolate(c1.b, c2.b, t)
  local a = util.interpolate(c1.a, c2.a, t2)
  return util.ass_style_color(math.floor(r + 0.5), math.floor(g + 0.5),
                              math.floor(b + 0.5), math.floor(a + 0.5))
end

function util.interpolate_alpha(t, color1, color2, t2)
  if t2 == nil then t2 = t end
  local a1 = util.extract_color(color1).a
  local a2 = util.extract_color(color2).a
  return util.ass_alpha(math.floor(util.interpolate(a1, a2, t2) + 0.5))
end

function util.HSV_to_RGB(h, s, v)
  return host.__host_hsv_to_rgb(h, s, v)
end

function util.HSL_to_RGB(h, s, l)
  return host.__host_hsl_to_rgb(h, s, l)
end

function util.headtail(str, n)
  if str == nil then return "", "" end
  n = n or 1
  if #str <= 2 * n then return str, "" end
  return str:sub(1, n), str:sub(-n)
end

function util.trim(str)
  if str == nil then return "" end
  return (tostring(str):gsub("^%s+", ""):gsub("%s+$", ""))
end

function util.words(str)
  local out = {}
  for w in tostring(str):gmatch("%S+") do out[#out + 1] = w end
  return out
end

--------------------------------------------------------------------- regex

local re = {}
aegisub.re = re

function re.find(str, pattern)
  return host.__host_re_find(str, pattern)
end

function re.match(str, pattern)
  return host.__host_re_match(str, pattern)
end

function re.gsub(str, pattern, replacement)
  return host.__host_re_gsub(str, pattern, replacement)
end

function re.split(str, pattern)
  return host.__host_re_split(str, pattern)
end

--------------------------------------------------------------- unicode module

unicode = {
  charwidth = function(text, index) return host.__host_unicode_charwidth(text, index) end,
  len = function(str) return host.__host_unicode_len(str) end,
  sub = function(str, i, j) return host.__host_unicode_sub(str, i, j) end,
  char = function(...) return host.__host_unicode_char(...) end,
  codepoint = function(str, index) return host.__host_unicode_codepoint(str, index) end,
  upper = function(str) return host.__host_unicode_upper(str) end,
  lower = function(str) return host.__host_unicode_lower(str) end,
  reverse = function(str) return host.__host_unicode_reverse(str) end,
}

----------------------------------------------------------- include / require

local loaded = {}

function include(path)
  return host.__host_include(path)
end

function require(name)
  if loaded[name] ~= nil then return loaded[name] end
  if name == "aegisub.util" then
    loaded[name] = util
    return util
  end
  if name == "aegisub.re" then
    loaded[name] = re
    return re
  end
  if name == "aegisub.unicode" then
    loaded[name] = unicode
    return unicode
  end
  local value = host.__host_require(name)
  loaded[name] = value
  return value
end

-- `table.copy` exists in Aegisub (installed by utils.lua); provide a fallback
-- so scripts that forget to include utils.lua still work.
if table.copy == nil then
  table.copy = util.copy
end

--------------------------------------------------------------- subs object --

-- Upstream Aegisub builds the `subtitles`/`subs` object as a *userdata* whose
-- metatable provides `__index` / `__newindex` / `__len` / `__ipairs`
-- (src/auto4_lua_assfile.cpp, LuaAssFile::ObjectIndexRead/ObjectIndexWrite/
-- ObjectGetLen/ObjectIPairs).  That is what gives Automation 4 scripts three
-- properties they all rely on:
--
--   * `subs[i]` hands back a *fresh* table every time, so a macro can mutate
--     the copy freely and only `subs[i] = line` writes back to the file;
--   * `#subs` and `subs.n` are live views of the file, out-of-range reads are
--     nil rather than an error;
--   * `subs[0] = line` appends, `subs[-i] = line` inserts before i,
--     `subs[i] = nil` deletes, plus delete/deleterange/insert/append/
--     script_resolution.
--
-- A plain table cannot express that in this VM: LuaJIT is built *without*
-- LUA52COMPAT, so for a key that already exists in a table's raw part Lua
-- never consults `__index`/`__newindex`.  A table cache would therefore hand
-- the same line table out twice (breaking copy semantics) and every
-- `subs[i] = line` would be swallowed by the cache instead of reaching the
-- file.  A userdata has no raw part, so every access goes through the
-- metatable - exactly like Aegisub.
local subs_metatables = setmetatable({}, { __mode = "k" })

local native_ipairs, native_pairs = ipairs, pairs

local function subs_metatable_of(value)
  if type(value) ~= "userdata" then return nil end
  local mt = getmetatable(value)
  if mt == nil or subs_metatables[mt] == nil then return nil end
  return mt
end

function __is_subs(value)
  return subs_metatable_of(value) ~= nil
end

-- Aegisub's LuaJIT is built with LUA52COMPAT, so `__ipairs`/`__pairs` (and
-- iterating the subtitles userdata with `ipairs(subtitles)`, which every
-- karaskel script does) work there.  This VM's builtin `ipairs`/`pairs` reject
-- non-tables outright: "bad argument #1 to 'ipairs' (table expected, got
-- userdata)".  Restore Lua 5.2 semantics: honour the metamethod when the value
-- has one, otherwise defer to the builtin unchanged.
function ipairs(value)
  local mt = getmetatable(value)
  if mt ~= nil and mt.__ipairs ~= nil then return mt.__ipairs(value) end
  return native_ipairs(value)
end

function pairs(value)
  local mt = getmetatable(value)
  if mt ~= nil and mt.__pairs ~= nil then return mt.__pairs(value) end
  return native_pairs(value)
end

local SUBS_BUILDER = [[
-- Runs as its own chunk; prelude locals arrive as varargs.
local subs_metatables, native_ipairs, native_pairs = ...

function __build_subs(handle)
  local methods = {}
  local fields = {} -- non-numeric assignments, e.g. subs.foo = 1

  local function count() return handle:count() end

  methods.refresh = function() return count() end

  methods.append = function(...)
    local n = select("#", ...)
    for i = 1, n do handle:append((select(i, ...))) end
    return count()
  end

  methods.insert = function(index, ...)
    local at = math.floor(tonumber(index) or 1)
    local n = select("#", ...)
    for i = 1, n do handle:insert(at, (select(i, ...))) end
    return count()
  end

  methods.delete = function(...)
    local n = select("#", ...)
    local indices = {}
    if n == 1 and type((...)) == "table" then
      for _, value in ipairs((...)) do indices[#indices + 1] = tonumber(value) end
    else
      for i = 1, n do indices[#indices + 1] = tonumber((select(i, ...))) end
    end
    -- Aegisub resolves every index against the file as it was before any
    -- deletion, so remove from the back to keep the lower indices valid.
    table.sort(indices, function(a, b) return a > b end)
    for _, index in ipairs(indices) do
      if index ~= nil then handle:delete(index, 1) end
    end
    return count()
  end

  methods.deleterange = function(first, last)
    handle:delete_range(tonumber(first) or 1, tonumber(last) or 0)
    return count()
  end

  -- Undocumented upstream method (auto4_lua_assfile.cpp:
  -- LuaAssFile::LuaGetScriptResolution) that karaskel.collect_head() calls.
  -- Returns the script resolution, falling back to libass' 384x288.
  methods.script_resolution = function()
    local x, y
    for i = 1, count() do
      local line = handle:get(i)
      if line ~= nil and line.class == "info" then
        local key = tostring(line.key or ""):lower()
        if key == "playresx" then x = tonumber(line.value) end
        if key == "playresy" then y = tonumber(line.value) end
      end
    end
    if x == nil and y == nil then
      local video_x, video_y = aegisub.video_size()
      if video_x then x, y = video_x, video_y end
    end
    return x or 384, y or 288
  end

  local subs = newproxy(true)
  local mt = getmetatable(subs)
  subs_metatables[mt] = true

  -- reads: subs[i], subs.n, subs.delete, ...  (Aegisub: ObjectIndexRead)
  mt.__index = function(_, key)
    local kind = type(key)
    if kind == "number" then
      if key >= 1 and key == math.floor(key) then
        return handle:get(key) -- a fresh table on every read
      end
      return nil
    end
    if kind == "string" then
      if key == "n" then return count() end
      local method = methods[key]
      if method ~= nil then return method end
      return fields[key]
    end
    return nil
  end

  -- writes: subs[0] = line, subs[-i] = line, subs[i] = line, subs[i] = nil
  -- (Aegisub: ObjectIndexWrite)
  mt.__newindex = function(_, key, value)
    if type(key) ~= "number" then fields[tostring(key)] = value return end
    local index = math.floor(key)
    if index ~= key then
      error("Subtitle file object index must be an integer, got " .. tostring(key), 2)
    end
    if index == 0 then
      handle:append(value)
    elseif index < 0 then
      handle:insert(-index, value)
    elseif value == nil then
      handle:delete(index, 1)
    else
      handle:set(index, value)
    end
  end

  mt.__len = function() return count() end

  mt.__ipairs = function()
    local i = 0
    return function()
      i = i + 1
      local line = handle:get(i)
      if line == nil then return nil end
      return i, line
    end
  end

  -- The file object has no enumerable fields; `pairs(subs)` yields nothing,
  -- matching the userdata Aegisub hands macros.
  mt.__pairs = function() return native_pairs({}) end

  return subs
end
]]

-- The builder is a separate chunk, so the prelude's `subs_metatables` /
-- `native_*` locals have to be handed to it explicitly as chunk varargs.
local load_subs_builder = assert(loadstring or load)
load_subs_builder(SUBS_BUILDER, "aegisub-mcp/subs-builder")(subs_metatables, native_ipairs, native_pairs)

function __new_subs(handle)
  return __build_subs(handle)
end

------------------------------------------------------------- protected calls

-- Runs a macro / filter and returns ok, err where err carries a Lua traceback.
function __protected_call(fn, ...)
  local args = { ... }
  local n = table.maxn(args)
  local values = {}
  local ok, err = xpcall(function()
    values = { fn(unpack(args, 1, n)) }
    return true
  end, function(e)
    local tb = debug and debug.traceback and debug.traceback("", 2) or ""
    return tostring(e) .. "\n" .. tb
  end)
  if not ok then return false, err end
  -- Also hand back what the macro returned: Aegisub reads up to two values back
  -- from a macro (a new active line index and a new selection table -- see
  -- src/auto4_lua.cpp, LuaCommand::operator()), so dropping them here would
  -- make a headless run quietly disagree with the GUI.
  return true, nil, unpack(values, 1, table.maxn(values))
end

------------------------------------------------------------ missing API trap

-- Aegisub exposes more surface than any single reimplementation; anything we
-- do not provide should fail loudly and clearly instead of returning nil.
setmetatable(aegisub, {
  __index = function(_, name)
    if name == "lua_automation_version" then return 4 end
    rawset(aegisub, name, nil)
    host.__host_missing_api(tostring(name))
  end,
})

function __aegisub_prelude_version()
  return "aegisub-mcp prelude 2"
end

-- Explicit contract with the Python loader: hand the subs builder back as the
-- chunk's return value. `lupa`'s LuaRuntime.execute() returns the *chunk's*
-- return values, so a prelude that returns nothing makes the engine see `None`
-- even though `__build_subs` is a perfectly good global.
return __build_subs
