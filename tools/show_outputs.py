"""Print the text outputs of an executed notebook (for review in a terminal)."""
import json
import sys

nb = json.load(open(sys.argv[1]))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    n += 1
    print(f"--- cell {n}")
    for o in c.get("outputs", []):
        if o["output_type"] == "stream":
            print(o["text"][:limit])
        elif o["output_type"] in ("execute_result", "display_data"):
            d = o["data"]
            if "image/png" in d:
                print("[figure]")
            else:
                t = d.get("text/plain", "")
                print((t if isinstance(t, str) else "".join(t))[:limit])
        elif o["output_type"] == "error":
            print("ERROR", o["ename"], o["evalue"])
