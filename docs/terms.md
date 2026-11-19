# Words on the panels

The MCP App's panels use plain words. Each is explained once here, beside the word the tools and
olmoearth-inferencex use for it. The panels themselves: [mcp-app.md](mcp-app.md).

| On the panels | In the tools | What it is |
|---|---|---|
| square | window | A small block of map pixels, for example 11 x 11 pixels, about 110 m a side. The map is cut into squares, and you answer one square at a time. The Check card says how big a square is. |
| point | point | One pixel of a Studio map, read at the centre of each cell of an even grid laid over the map. The panels say point where the app reads a Studio map this way: a map that holds only classes, a map of numbers that is not a score from 0 to 1, two Studio maps compared, or a list of where to look first on a Studio map. Points have no pictures. |
| part | UTM zone | Studio saves a map as one file for each strip of the world it covers; a strip that crosses the equator is two parts, north and south. Each part is checked on its own. |
| range | interval | Two numbers that the map's error rate lies between, in at least 95 of every 100 checks done this way. More squares give a narrower range. It assumes your answers are right. |
| rule | labelling rule | One sentence on how you decide what a square is. It is fixed when the squares are picked, and shown on the sheet and with the result. |
| Cannot tell | `?`, not judged | The answer for a square you cannot judge, for example under cloud on every date. It widens the range; a guess can make the result wrong. |
| labelling blind | blind | You answer without seeing what the map says at the square; the app works out right or wrong. A sheet that asks only "right or wrong" must show the map's class, so it is not blind. |
| the pick | sample | The squares to check, picked at random when you press Pick the squares. They are fixed: pressing again opens the same squares. |
| one result per part | one estimate per zone | Each part gets its own error rate and range. Report them side by side; never average them. |
| error rate | error rate | How often the map is wrong: the share of a part's squares where the map's class is not what you saw, estimated from the squares you checked. |
