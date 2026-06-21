"""Anchored Elo for the 1v1 opponent-pool AZ trainer.

Only the learner plays, every rating derives from learner games.
Learner is non-stationary, frozen pool snapshots inherit learner's rating,
gen_0 is a pinned anchor.
"""

import json


class EloBook:
    """Player ratings updated from learner-only batches."""

    def __init__(
        self,
        init=1500.0,
        k_learner=2.0,
        k_opp=0.5,
        draw_weight=0.5,
        anchor="gen_0",
        learner="learner",
    ):
        self.init = init
        self.k_learner = k_learner
        self.k_opp = k_opp
        self.draw_weight = draw_weight
        self.anchor = anchor
        self.learner = learner
        self.ratings = {learner: init, anchor: init}
        self.games = {learner: 0, anchor: 0}

    def expected(self, ra, rb):
        """Logistic expected score of `ra` vs `rb`."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def seed(self, player_id):
        """Register a player at init rating if unseen."""
        self.ratings.setdefault(player_id, self.init)
        self.games.setdefault(player_id, 0)

    def update(self, opp_id, wins, losses, draws=0):
        """Fold one generation's learner-vs-`opp_id` batch into the ratings.

        s = wins + draw_weight*draws over n games; learner steps k_learner*(s - n*E),
        opponent -k_opp*(s - n*E); the anchor never moves."""
        n = wins + losses + draws
        if n == 0:
            return
        self.seed(opp_id)
        s = wins + self.draw_weight * draws
        r_l = self.ratings[self.learner]
        r_o = self.ratings[opp_id]
        delta = s - n * self.expected(r_l, r_o)
        self.ratings[self.learner] = r_l + self.k_learner * delta
        if opp_id != self.anchor:
            self.ratings[opp_id] = r_o - self.k_opp * delta
        self.games[self.learner] += n
        self.games[opp_id] += n

    def on_snapshot(self, new_id):
        """New pool snapshot inherits the learner's current rating."""
        self.ratings[new_id] = self.ratings[self.learner]
        self.games.setdefault(new_id, 0)

    def present_summary(self, present_ids):
        """Derived log metrics for the present pool members."""
        r_l = self.ratings[self.learner]
        pool = [
            self.ratings[g]
            for g in present_ids
            if g in self.ratings and g not in (self.learner, self.anchor)
        ]
        best_pool = max(pool) if pool else r_l
        return {
            "best_pool": best_pool,
            "learner_minus_ref": r_l - self.ratings[self.anchor],
            "gap_to_pool": r_l - best_pool,
        }

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "init": self.init,
                    "k_learner": self.k_learner,
                    "k_opp": self.k_opp,
                    "draw_weight": self.draw_weight,
                    "anchor": self.anchor,
                    "learner": self.learner,
                    "ratings": self.ratings,
                    "games": self.games,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        book = cls(
            init=d["init"],
            k_learner=d["k_learner"],
            k_opp=d["k_opp"],
            draw_weight=d["draw_weight"],
            anchor=d["anchor"],
            learner=d["learner"],
        )
        book.ratings = d["ratings"]
        book.games = d["games"]
        return book
