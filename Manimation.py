from manim import *

class ModelFlow(Scene):
    def construct(self):
        # make board
        cells = VGroup(*[Square(side_length=0.25, stroke_color=DARK_GREY) for _ in range(280)])
        cells.arrange_in_grid(rows=28, buff=0.0)
        self.add(cells)
        
        # setup tki
        tki = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
               1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
               1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        tki_cells = VGroup(*[cell for cell, val in zip(cells[::-1], tki) if val])
        
        self.play(tki_cells.animate(lag_ratio=0.2).set_fill(WHITE, opacity=1.0))
        
        self.wait()