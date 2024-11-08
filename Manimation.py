from manim import *
import numpy as np

class ModelFlow(Scene):
    def construct(self):

        border = Rectangle(height=7, width=2.5, grid_xstep=0.25, grid_ystep=0.25)
        border.grid_lines.set_stroke(width=2, color=DARK_GREY)

        np_tki = (np.load('tki.npy') * 255).astype(np.uint8)
        tki = ImageMobject(np_tki).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        tki.fade(1.0).match_height(border)
        self.add(tki)

        np_conv1 = (np.load('conv1.npy') * 255).astype(np.uint8)
        conv1 = ImageMobject(np_conv1).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        conv1.fade(1.0).match_height(border)
        self.add(conv1)

        np_conv2 = (np.load('conv2.npy') * 255).astype(np.uint8)
        conv2 = ImageMobject(np_conv2).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        conv2.fade(1.0).match_height(border)
        self.add(conv2)

        np_conv3 = (np.load('conv3.npy') * 255).astype(np.uint8)
        conv3 = ImageMobject(np_conv3).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        conv3.fade(1.0).match_height(border)
        self.add(conv3)
        
        self.wait()

        self.play(Write(border))
        
        self.play(tki.animate.fade(0.0))
        self.wait()
        self.play(conv1.animate.fade(0.0))
        self.wait()
        self.play(conv2.animate.fade(0.0))
        self.wait()
        self.play(conv3.animate.fade(0.0), Transform(border, Rectangle(height=7, width=2.5, grid_xstep=0.5, grid_ystep=0.5)))
        self.wait()