from manim import *
import numpy as np

class ModelFlow(Scene):
    def construct(self):
        I_PIECE = ManimColor.from_rgb((0.0, 1.0, 1.0, 0.5))
        S_PIECE = ManimColor.from_rgb((0.0, 1.0, 0.0, 0.5))
        T_PIECE = ManimColor.from_rgb((1.0, 0.0, 1.0, 0.5))
        
        border = Rectangle(height=7, width=2.5, grid_xstep=0.25, grid_ystep=0.25)
        border.grid_lines.set_stroke(width=2, color=DARK_GREY)
        border.z_index = 2

        conv_kernels = VGroup(*[Square(stroke_color=I_PIECE) for _ in range(280)]).arrange_in_grid(rows=28, buff=0).match_height(border)
        [kernel.save_state().scale(3) for kernel in conv_kernels]

        last_conv_kernels = VGroup(*[Square(stroke_color=T_PIECE) for _ in range(70)]).arrange_in_grid(rows=14, buff=0).match_height(border)
        last_conv_kernels.z_index = 1.5
        [kernel.save_state().scale(1.5) for kernel in last_conv_kernels]
        last_conv_kernels.align_to(border, RIGHT).align_to(border, DOWN)
        
        np_tki = (np.load('tki.npy') * 255).astype(np.uint8)
        tki = ImageMobject(np_tki).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).match_height(border)

        np_conv1 = (np.load('conv1.npy') * 255).astype(np.uint8)
        conv1 = ImageMobject(np_conv1).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).match_height(border)

        np_conv2 = (np.load('conv2.npy') * 255).astype(np.uint8)
        conv2 = ImageMobject(np_conv2).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).match_height(border)

        np_conv3 = (np.load('conv3.npy') * 255).astype(np.uint8)
        conv3 = ImageMobject(np_conv3).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).match_height(border)
        
        self.wait()

        self.play(Write(border))
        
        self.play(FadeIn(tki))
        self.wait()
        
        # Show first kernel
        self.play(Write(conv_kernels[0]))
        self.wait()
        self.play(conv_kernels[0].animate.restore())
        self.wait()

        # Show the rest of the kernels and resulting convolution
        self.play(Write(conv_kernels[1:]))
        self.play(AnimationGroup([kernel.animate.restore() for kernel in conv_kernels[1:]], lag_ratio=0.01))
        self.wait()
        self.play(FadeOut(conv_kernels), FadeIn(conv1))
        self.wait()

        # Repeat for next convolution layer
        [kernel.set_stroke_color(S_PIECE).save_state().scale(2.9) for kernel in conv_kernels]
        self.play(Write(conv_kernels))
        self.play(AnimationGroup([kernel.animate.restore() for kernel in conv_kernels], lag_ratio=0.01))
        self.play(FadeOut(conv_kernels), FadeIn(conv2))
        self.wait()
        self.remove(conv_kernels)
        
        # Repeat for last convolution layer
        self.play(Write(last_conv_kernels))
        self.play(AnimationGroup([kernel.animate.restore() for kernel in last_conv_kernels], lag_ratio=0.01))
        self.play(FadeOut(border), last_conv_kernels.animate.set_stroke_color(DARK_GREY), FadeIn(conv3))
        self.wait()
        self.remove(last_conv_kernels)
        