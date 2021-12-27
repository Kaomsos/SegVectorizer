# %%
import argparse
from segvec.utils import palette
from segvec.vetorizer import PaletteConfiguration
from segvec.func import convert_an_image

###########################################
# create a PaletteConfiguration
p_config = PaletteConfiguration(palette)
p_config.add_open("door&window")
for item in ['bathroom/washroom',
             'livingroom/kitchen/dining add_room',
             'bedroom',
             'hall',
             'balcony',
             'closet']:
    p_config.add_room(item)
p_config.add_boundary("wall")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", "-i", type=str, help="input of the script")
    param = parser.parse_known_args()[0]
    path = param.img
    convert_an_image(path, palette_config=p_config)
