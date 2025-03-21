# we just read in the tikz picture and compute the pairwise distances by using the point coordinates
# from the text file
import re
import math
import os

# read the file into memory
file_path = r"C:\Users\lucas\Documents\Veröffentlichungen\2025_HILDA\relationship-scattering.tex"
with open(file_path, 'r') as f:
    # read in the file content
    content = f.read()

# find the point coordinates
pattern = r"\\coordinate\s*\((\w+)\)\s*at\s*\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\);"
point_coords = [(ele[0], float(ele[1]), float(ele[2])) for ele in re.findall(pattern, content)]

# get the lines
content = content.split("\n")

# compute the distances and place them in the scrip
for idx, (point1, x1, y1) in enumerate(point_coords):
    for point2, x2, y2 in point_coords[idx+1:]:

        # compute the distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # go through the lines and replace them
        for ldx, line in enumerate(content):
            if f"{point1} - {point2}" in line:
                content[ldx] = f"{line.split(':')[0]} : {distance:.1f}}};"
            elif f"({point1}) -- ({point2})" in line:
                content[ldx] = f"{line.split('{')[0]} {{{distance:.1f}}};"

# save the new file in the current directory
with open(os.path.split(file_path)[-1], 'w') as f:
    f.write("\n".join(content))
