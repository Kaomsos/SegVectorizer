# %%
from __future__ import annotations
from segvec.vetorizer import Vectorizer, PaletteConfiguration
from segvec.utils import plot_wcl_o_against_target


##################################################
# defining top-level conversion functions
def convert_an_image(path, palette_config: PaletteConfiguration):
    vectorizer = Vectorizer(palette_config=palette_config)
    wcl_o = vectorizer(path=path)
    plot_wcl_o_against_target(wcl_o, vectorizer.boundary)


def convert_a_segmentation(segmentation, palette_config: PaletteConfiguration):
    vectorizer = Vectorizer(palette_config=palette_config)
    wcl_o = vectorizer(segmentation=segmentation)
    plot_wcl_o_against_target(wcl_o, vectorizer.boundary)





