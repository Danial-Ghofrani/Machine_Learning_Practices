from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

fonts = ["fonts/BKoodkBd.TTF", "fonts/BShiraz.TTF"]

for font_file in fonts:
    ttf_font = TTFont(font_file)

    for char in  ttf_font.getGlyphNames():
        print(char)
