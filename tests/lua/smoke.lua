-- aegisub-mcp Lua engine smoke test: exercises the Automation 4 surface Aegisub
-- scripts actually use. Run: python -m aegisub_mcp.lua.engine this.lua --ass FILE
script_name = "smoke"
script_description = "engine smoke test"
script_version = "1.0"
script_author = "aegisub-mcp"

include("karaskel.lua")

local function macro_info(subs)
  local n, classes = 0, {}
  for i, l in ipairs(subs) do
    n = n + 1
    classes[l.class] = (classes[l.class] or 0) + 1
  end
  aegisub.debug.out(3, "lines=%d info=%d style=%d dialogue=%d unknown=%d\n",
    n, classes.info or 0, classes.style or 0, classes.dialogue or 0, classes.unknown or 0)
  aegisub.debug.out("n field=%s #subs=%d\n", tostring(subs.n), #subs)
  -- copies are copies: editing one must NOT change the file
  local probe = subs[1]
  probe.value = "SHOULD_NOT_LAND"
  assert(subs[1].value ~= "SHOULD_NOT_LAND", "subs[i] must return a copy")
  -- first dialogue line
  for i = 1, #subs do
    local l = subs[i]
    if l.class == "dialogue" then
      aegisub.debug.out("line %d: start=%d end=%d style=%s actor=%r text=%r\n",
        i, l.start_time, l.end_time, tostring(l.style), tostring(l.actor), l.text)
      aegisub.set_undo_point("smoke: edit line")
      l.text = l.text .. "{\\b1}SMOKE{\\b0}"
      subs[i] = l
      break
    end
  end
  -- mutate an existing line: tag edit on a copy then write back
  for i = 1, #subs do
    local l = subs[i]
    if l.class == "dialogue" then
      local edited = subs[i]
      edited.effect = "smoke"
      subs[i] = edited
      break
    end
  end
  -- structural edits
  local meta = karaskel.collect_head(subs, false)
  aegisub.debug.out("styles collected: %d (playres %dx%d)\n", meta.styles.n, meta.playresx or -1, meta.playresy or -1)
  local newline = {class = "dialogue", layer = 9, start_time = 0, end_time = 500,
                   style = "Default", actor = "smoke", margin_l = 0, margin_r = 0,
                   margin_v = 0, effect = "", text = "appended by smoke"}
  subs.append(newline)
  subs.insert(1, {class = "info", key = "SmokeFlag", value = "1"})
  local after_append = #subs
  -- classify the appended line: must come back as a dialogue line, not unknown
  local last = subs[after_append]
  aegisub.debug.out("appended class=%s text=%r\n", tostring(last.class), tostring(last.text))
  subs.delete(1)  -- drop the info line we inserted
  aegisub.progress.title("smoke: half way")
  aegisub.progress.task("counting %d lines", #subs)
  aegisub.progress.set(0.5)
  return {format = "smoke ok", lines = #subs}
end

local function macro_api_probe(subs)
  local out = {}
  out[#out + 1] = ("text_extents=%s"):format(tostring(select(1, aegisub.text_extents({fontname = "Arial", fontsize = 48}, "Hello"))))
  out[#out + 1] = ("frame_from_ms=%s"):format(tostring(aegisub.frame_from_ms(1000)))
  out[#out + 1] = ("ms_from_frame=%s"):format(tostring(aegisub.ms_from_frame(1)))
  out[#out + 1] = ("decode_path=%s"):format(tostring(aegisub.decode_path("?script")))
  out[#out + 1] = ("charwidth=%d len=%d sub=%r"):format(unicode.charwidth("กnan", 1), unicode.len("กnan"), unicode.sub("กnan", 1, 2))
  out[#out + 1] = ("re_find=%s"):format(tostring(#aegisub.re.find("a1 b2 c3", "%d")))
  out[#out + 1] = ("re_match=%s"):format(tostring(select(1, aegisub.re.match("123", "(%d+)"))))
  out[#out + 1] = ("gsub=%s"):format(tostring(aegisub.re.gsub("abc", "b", "X")))
  out[#out + 1] = ("util.interpolate=%s"):format(aegisub.util.interpolate(0, 10, 0.25))
  out[#out + 1] = ("util.ass_color=%s"):format(aegisub.util.ass_color(255, 0, 0))
  out[#out + 1] = ("hsv=%s %s %s"):format(aegisub.util.hsv_to_rgb(0, 1, 1))
  out[#out + 1] = ("util.copy=%s"):format(aegisub.util.copy({1, 2})[2])
  local props = aegisub.project_properties()
  out[#out + 1] = ("project.video=%s fps=%s"):format(tostring(props.video_file), tostring(props.fps))
  for i = 1, #out do aegisub.debug.out(3, "%s\n", out[i]) end
  return out[1]
end

local function macro_filter(subtitles, selected, active)
  local count = 0
  for _, i in ipairs(selected) do
    local l = subtitles[i]
    if l.class == "dialogue" then
      l.text = string.gsub(l.text, "SMOKE", "FILTERED")
      subtitles[i] = l
      count = count + 1
    end
  end
  aegisub.debug.out("filter saw active=%s count=%d\n", tostring(active), count)
  return true
end

aegisub.register_macro("Smoke Info", "classify + edit lines", macro_info)
aegisub.register_macro("Smoke API", "probe the API surface", macro_api_probe)
aegisub.register_filter("Smoke Filter", "rewrite one word", 50, macro_filter)
