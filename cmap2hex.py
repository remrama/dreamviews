import colorcet as cc

cmap = cc.cm.bwy  # CET_CBD1

for val, name in [(0., "low"), (.5, "mid"), (1., "high")]:
    rgb_decimals = cmap(val) # must be float!!
    rgb_ints = [ round(v*255) for v in rgb_decimals ]
    hex_code = "#{0:02x}{1:02x}{2:02x}".format(*rgb_ints)
    print(f"{name} hex code: {hex_code}")
