# %%
from SegVec.utils import plot_wcl_against_target, plot_position_of_rects, plot_wcl_o_against_target
from SegVec.vetorizer import Vectorizer, PaletteConfiguration
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from SegVec.utils import palette


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


##################################################
# defining top-level conversion functions
def convert_a_image(path, palette_config=p_config):
    vectorizer = Vectorizer(palette_config=p_config)
    wcl_o = vectorizer(path=path)
    plot_wcl_o_against_target(wcl_o, vectorizer.boundary)


def convert_a_segmentation(segmentation, palette_config=p_config):
    vectorizer = Vectorizer(palette_config=p_config)
    wcl_o = vectorizer(segmentation=segmentation)
    plot_wcl_o_against_target(wcl_o, vectorizer.boundary)


# %%
if __name__ == "__main__":
    path = 'data/Figure_47541863.png'
    convert_a_image(path)




