path = r"D:\agent\course-agent\docs\figures\src\stitch_gate_three_arms_v5.html"
with open(path, "rb") as f:
    data = f.read()
idx = data.find(b'green')
print(data[idx:idx+80])
