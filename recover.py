from pathlib import Path
j = 0
for i in Path(".git/lost-found/other/").glob("*"):
    try:
        src = open(i, "r")
        content = src.read()
        for keywords in [""]:
            if keywords in content:
                dst = open(f"{j}.py", "w")
                dst.write(content)
                j += 1
                break

    except:
        continue