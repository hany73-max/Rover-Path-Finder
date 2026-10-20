# Getting a small piece of Mars to plan a route on

## Where the data comes from
NASA's HiRISE camera makes **terrain models (DTMs)**: small patches of Mars (a few km wide) where every pixel is an elevation in metres, at 1 or 2 m per pixel. 709 exist. `small_mars_dtms.csv` lists the 60 smallest with direct download links (the full index is the GitHub repo roncapat/NASA-Hirise-DTMs-DEMs-index).

## Good picks (links from that index; click or `curl -O <url>`)
| Size | Pixel | What it is | Link |
|---|---|---|---|
| 9 MB | 1 m | Possible Opportunity rover landing site (flat, safe terrain) | https://www.uahirise.org/PDS/DTM/PSP/ORB_001400_001499/PSP_001414_1780_PSP_001612_1780/DTEEC_001414_1780_001612_1780_U01.IMG |
| 81 MB | 2 m | Constriction in Kasei Valles (rugged channel, best for demos) | https://www.uahirise.org/PDS/DTM/ESP/ORB_032100_032199/ESP_032193_1990_ESP_031837_1990/DTEED_032193_1990_031837_1990_A01.IMG |
| 128 MB | 2 m | Peace Vallis in Gale Crater (Curiosity's area) | https://www.uahirise.org/PDS/DTM/ESP/ORB_062900_062999/ESP_062963_1760_ESP_062172_1760/DTEED_062963_1760_062172_1760_A01.IMG |
| 160 MB | 1 m | East Endeavour Crater dune field (Opportunity's area) | https://www.uahirise.org/PDS/DTM/ESP/ORB_042600_042699/ESP_042674_1775_ESP_042740_1775/DTEEC_042674_1775_042740_1775_A01.IMG |

## Steps
1. Download one `.IMG` file.
2. Look at the map and read pixel coordinates (the pixel size is read from the file):
   `python plan_on_heightmap.py FILE.IMG --ticks --out look`
   then open `look/map_contours.png`.
3. Plan with your chosen points (x y in original pixels):
   `python plan_on_heightmap.py FILE.IMG --start 300 2500 --goal 2800 400 --max-slope 20 --rover-radius 20 --out result`
4. Or click the points: add `--pick` (needs a desktop window).

## Tips
- The default 100 m safety margin is too large for tiny patches: use `--rover-radius 10` to `30`.
- If it says every cell is a hazard, the slope limit is too strict for that terrain; raise `--max-slope`.
- A flat landing-site patch gives an almost straight route. Rugged terrain (Kasei Valles) shows the method off.
- Big file or want detail in one area? Add `--crop X0 Y0 X1 Y1`.
- If a file won't open: `gdal_translate FILE.IMG FILE.tif`, then use the .tif.
