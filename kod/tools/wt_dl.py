import sys, re, json, requests
url, out = sys.argv[1], sys.argv[2]
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"
url = s.head(url, allow_redirects=True).url
p = [x for x in url.split("?")[0].split("/") if x][3:]
tid, sh = p[0], p[-1]
home = s.get("https://wetransfer.com/").text
m = re.search(r'name="csrf-token" content="([^"]+)"', home)
h = {"x-requested-with": "XMLHttpRequest", "content-type": "application/json"}
if m: h["x-csrf-token"] = m.group(1)
j = {"intent": "entire_transfer", "security_hash": sh}
if len(p) == 3: j["recipient_id"] = p[1]
r = s.post(f"https://wetransfer.com/api/v4/transfers/{tid}/download", json=j, headers=h)
link = r.json().get("direct_link"); print(r.status_code, bool(link))
with s.get(link, stream=True) as g, open(out, "wb") as f:
    for c in g.iter_content(1 << 20): f.write(c)
print("ok", out)
