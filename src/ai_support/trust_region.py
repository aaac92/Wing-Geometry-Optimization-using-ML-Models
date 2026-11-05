"""
Trust-region search over airfoil shapes: the optimisation method of the iterative inference (inference_and_testing.py).

Every distance and radius here is measured in dataset grid steps, per CST coefficient, with the Chebyshev
(largest single coefficient) distance: 1 step is the spacing of the dataset grid in that coefficient, so LE,
intermediate and TE coefficients are comparable.

  Dataset_Peaks   local maxima of E on the dataset grid: the starting points of the search (one track each)
  Trust_Region    one track: a box of half-width `radius` around its best simulated shape. The surrogate models
                  pick candidates inside the box; after they are simulated, update() moves and resizes the box
                  with the classic trust-region ratio rho = actual gain / predicted gain:
                      improvement, rho >= 0.75 (or no prediction): centre moves, radius doubles (up to r_max)
                      improvement, rho < 0.75:                      centre moves, radius kept
                      no improvement:                               radius halves
                  A track whose radius falls below r_min has converged. prune() drops converged and trailing tracks
                  and merges tracks that met.
  Lane_Monitor    stop rule of one search lane: the best E gained less than min_gain (relative) for `patience`
                  iterations in a row.
  diverse_top     best-first selection that keeps picks apart and skips shapes rejected by a check.

Self-test: python src/ai_support/trust_region.py (an analytic 6-D function whose optimum lies outside the
"dataset" window; the search must find it from the dataset peaks).
"""
import numpy as np
from scipy.spatial import cKDTree


class Dataset_Peaks:
    """
    Peak shapes of a gridded dataset. A row is a peak when no grid neighbour of the same configuration (every shape
    within 1 grid step in each coefficient) has a higher E and its E is at least `share` x the best E of that
    configuration. Peaks are counted per shape: a shape that peaks in several configurations is one peak, with
    the E of its best configuration. At most `max_peaks` peaks (highest first) are returned.
    """

    def __init__(self, cst, config_keys, E, steps, share: float = 0.90, max_peaks: int = 6):
        self.cst = np.asarray(cst, dtype=np.float64)
        self.config_keys = np.asarray(config_keys)
        self.E = np.asarray(E, dtype=np.float64)
        self.steps = np.asarray(steps, dtype=np.float64)
        self.share = share
        self.max_peaks = max_peaks
        self.found = 0      # peaks before the max_peaks cap
        self.per_config = {}

    def find(self) -> list:
        """[{"cst", "E", "config", "E_by_config", "peak_in"}], best first."""
        grid = np.round(self.cst / self.steps)
        shapes = {}
        for key in np.unique(self.config_keys):
            rows = np.where(self.config_keys == key)[0]
            e = self.E[rows]
            limit = self.share * np.nanmax(e)
            candidates = np.where(e >= limit)[0]
            tree = cKDTree(grid[rows])
            neighbours = tree.query_ball_point(grid[rows][candidates], r=1.01, p=np.inf)
            peaks = [rows[c] for c, nb in zip(candidates, neighbours) if e[c] >= np.max(e[nb])]
            self.per_config[str(key)] = len(peaks)
            for r in peaks:
                shape = tuple(grid[r].astype(int))
                entry = shapes.setdefault(shape, {"cst": self.cst[r].copy(), "E": -np.inf, "config": None,
                                                  "E_by_config": {}, "peak_in": []})
                entry["peak_in"].append(str(key))
                entry["E_by_config"][str(key)] = float(self.E[r])
                if self.E[r] > entry["E"]:
                    entry["E"], entry["config"] = float(self.E[r]), str(key)
        ordered = sorted(shapes.values(), key=lambda p: -p["E"])
        self.found = len(ordered)
        return ordered[:self.max_peaks]

    def summary_lines(self, peaks: list, cst_names=None) -> list:
        """Readable description of the peaks (console and reports)."""
        capped = f", following the best {len(peaks)}" if self.found > len(peaks) else ""
        lines = [f"Dataset peaks: {self.found} local maxima within {1 - self.share:.0%} of the best E of their configuration "
                 f"({', '.join(f'{k} {n}' for k, n in self.per_config.items())}){capped} -> {len(peaks)} track(s) per lane"]
        for i, p in enumerate(peaks, start=1):
            values = ", ".join(f"{v:.4f}" for v in p["cst"]) if cst_names is None else \
                ", ".join(f"{n}={v:.4f}" for n, v in zip(cst_names, p["cst"]))
            lines.append(f"  peak {i}: E {p['E']:.3f} ({p['config']}), peak in {'+'.join(p['peak_in'])} | {values}")
        return lines


class Trust_Region:
    """One search track: a box of half-width `radius` grid steps around the best simulated shape found so far."""
    R0 = 1.0          # starting half-width (grid steps)
    R_MIN = 0.1       # below this the track has converged (the robustness perturbation size)
    R_MAX = 3.0
    EXPAND_RHO = 0.75

    def __init__(self, track_id: str, centre, centre_E: float, centre_config: str, steps, radius: float = None,
                 r_min: float = None, r_max: float = None, origin: str = "dataset peak"):
        self.track_id = track_id
        self.centre = np.asarray(centre, dtype=np.float64)
        self.centre_E = float(centre_E)
        self.centre_config = centre_config
        self.steps = np.asarray(steps, dtype=np.float64)
        self.radius = float(self.R0 if radius is None else radius)
        self.r_min = float(self.R_MIN if r_min is None else r_min)
        self.r_max = float(self.R_MAX if r_max is None else r_max)
        self.origin = origin
        self.history = []

    # --- geometry --- #
    def distance(self, cst) -> np.ndarray:
        """Chebyshev distance (grid steps) of each shape to the centre."""
        return np.abs((np.atleast_2d(cst) - self.centre) / self.steps).max(axis=1)

    def inside(self, cst) -> np.ndarray:
        return self.distance(cst) <= self.radius + 1e-9

    def sample_uniform(self, n: int, rng) -> np.ndarray:
        """n shapes uniformly distributed in the box (not clipped to any dataset window)."""
        offsets = rng.uniform(-self.radius, self.radius, size=(n, len(self.centre)))
        return (self.centre + offsets * self.steps).astype(np.float32)

    def sample_around(self, seeds, n: int, rng, spread: float = 0.25) -> np.ndarray:
        """n shapes around randomly chosen seed shapes (Gaussian, std spread x radius steps), kept inside the box."""
        seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
        chosen = seeds[rng.integers(0, len(seeds), size=n)]
        noise = rng.normal(0.0, spread * self.radius, size=chosen.shape) * self.steps
        low, high = self.centre - self.radius * self.steps, self.centre + self.radius * self.steps
        return np.clip(chosen + noise, low, high).astype(np.float32)

    # --- update --- #
    @property
    def converged(self) -> bool:
        return self.radius < self.r_min

    def update(self, iteration: int, best_E, best_cst=None, best_config=None, predicted_E=None) -> dict:
        """
        Moves and resizes the box after the track's picks were simulated. best_E / best_cst / best_config: the best
        simulated shape among the picks (best_E None or NaN: nothing valid came back). predicted_E: the highest
        prediction among the picks, for rho. Returns the step record (also kept in history).
        """
        old_E, old_r = self.centre_E, self.radius
        valid = best_E is not None and np.isfinite(best_E)
        gain = (best_E - old_E) if valid else None
        rho = None
        if valid and predicted_E is not None and np.isfinite(predicted_E) and predicted_E > old_E:
            rho = float(gain / (predicted_E - old_E))
        if valid and gain > 0:
            self.centre = np.asarray(best_cst, dtype=np.float64)
            self.centre_E = float(best_E)
            self.centre_config = best_config
            if rho is None or rho >= self.EXPAND_RHO:
                self.radius, action = min(2 * old_r, self.r_max), "moved, radius doubled"
            else:
                action = "moved"
        else:
            self.radius, action = old_r / 2, "no gain, radius halved"
        step = {"iteration": int(iteration), "E_before": old_E, "E_after": self.centre_E,
                "gain": None if gain is None else float(gain), "rho": rho, "radius_before": old_r,
                "radius_after": self.radius, "action": action + (" -> converged" if self.converged else "")}
        self.history.append(step)
        return step

    @staticmethod
    def prune(tracks: list, best_E: float, iteration: int, drop_share: float = 0.95, merge_steps: float = 1.0,
              drop_from_iteration: int = 2):
        """
        (kept, removed) where removed is [(track, reason)]:
          converged  radius below r_min
          merged     centre within merge_steps of a better track's centre (the better one stays)
          trailing   from iteration drop_from_iteration on, centre E below drop_share x the lane's best E
        """
        kept, removed = [], []
        for track in sorted(tracks, key=lambda t: -t.centre_E):
            if track.converged:
                removed.append((track, f"converged (radius {track.radius:.3f} < {track.r_min:g} steps)"))
                continue
            near = next((k for k in kept if k.distance(track.centre)[0] <= merge_steps), None)
            if near is not None:
                removed.append((track, f"merged into track {near.track_id}"))
                continue
            if iteration >= drop_from_iteration and track.centre_E < drop_share * best_E:
                removed.append((track, f"trailing (E {track.centre_E:.3f} < {drop_share:.0%} of the lane best {best_E:.3f})"))
                continue
            kept.append(track)
        return kept, removed

    # --- recovery --- #
    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "centre": self.centre.tolist(), "centre_E": self.centre_E,
                "centre_config": self.centre_config, "steps": self.steps.tolist(), "radius": self.radius,
                "r_min": self.r_min, "r_max": self.r_max, "origin": self.origin, "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> "Trust_Region":
        track = cls(d["track_id"], d["centre"], d["centre_E"], d["centre_config"], d["steps"], radius=d["radius"],
                    r_min=d["r_min"], r_max=d["r_max"], origin=d.get("origin", "dataset peak"))
        track.history = list(d.get("history", []))
        return track


class Lane_Monitor:
    """Stop rule of a lane: done once the best E gains less than min_gain (relative) `patience` iterations in a row."""

    def __init__(self, best_E: float, min_gain: float = 0.005, patience: int = 2):
        self.best_E = float(best_E)
        self.min_gain = min_gain
        self.patience = patience
        self.stall = 0
        self.gains = []

    def update(self, best_E: float) -> float:
        """Records this iteration's best E; returns the relative gain."""
        gain = (best_E - self.best_E) / abs(self.best_E) if self.best_E else 0.0
        if best_E > self.best_E:
            self.best_E = float(best_E)
        self.stall = self.stall + 1 if gain < self.min_gain else 0
        self.gains.append(float(gain))
        return float(gain)

    @property
    def done(self) -> bool:
        return self.stall >= self.patience

    def to_dict(self) -> dict:
        return {"best_E": self.best_E, "min_gain": self.min_gain, "patience": self.patience, "stall": self.stall,
                "gains": self.gains}

    @classmethod
    def from_dict(cls, d: dict) -> "Lane_Monitor":
        monitor = cls(d["best_E"], d["min_gain"], d["patience"])
        monitor.stall, monitor.gains = int(d["stall"]), list(d["gains"])
        return monitor


def diverse_top(score, cst_scaled, k: int, mask=None, min_steps: float = 0.5, accept=None, taken=None) -> list:
    """
    Walks the rows from the highest score down and keeps a row only if its shape (cst_scaled: CST / grid steps)
    differs from every kept row, and from every shape in `taken` (scaled), by at least min_steps grid steps in some
    coefficient, and accept(i) is True (e.g. a geometry check; only called for rows that would be kept). Stops at
    k rows. Returns indices in rank order.
    """
    s = np.where(np.isfinite(score), score, -np.inf)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    kept = []
    others = np.zeros((0, cst_scaled.shape[1])) if taken is None or len(taken) == 0 else np.atleast_2d(taken)
    for i in np.argsort(-s):
        if len(kept) >= k or not np.isfinite(s[i]):
            break
        if kept and np.abs(cst_scaled[kept] - cst_scaled[i]).max(axis=1).min() < min_steps:
            continue
        if len(others) and np.abs(others - cst_scaled[i]).max(axis=1).min() < min_steps:
            continue
        if accept is not None and not accept(int(i)):
            continue
        kept.append(int(i))
    return kept


# =============================================================================================== #
# === Self-test === #
# =============================================================================================== #

if __name__ == "__main__":
    import itertools
    import sys

    rng = np.random.default_rng(0)
    d, levels = 6, 4
    steps = np.full(d, 0.04)
    optimum = np.array([3.6, 1.0, 2.0, 3.8, 0.5, 2.0]) * steps      # outside the grid window [0, 3] in two coefficients
    second = np.array([0.0, 3.0, 0.0, 0.0, 3.0, 0.0]) * steps

    def true_E(cst):
        x = np.atleast_2d(cst) / steps
        main = 8 * np.exp(-np.sum((x - optimum / steps) ** 2, axis=1) / (2 * 1.2 ** 2))
        side = 5 * np.exp(-np.sum((x - second / steps) ** 2, axis=1) / (2 * 0.8 ** 2))
        return 18 + main + side

    grid = np.array(list(itertools.product(range(levels), repeat=d)), dtype=np.float64) * steps
    E = true_E(grid)
    finder = Dataset_Peaks(grid, np.full(len(grid), "rec"), E, steps, share=0.80)
    peaks = finder.find()
    print("\n".join(finder.summary_lines(peaks)))
    checks = [("two peaks found", len(peaks) == 2)]

    def surrogate(cst):  # an imperfect model: right trend, biased and noisy
        return true_E(cst) * 0.97 + rng.normal(0, 0.15, size=len(np.atleast_2d(cst)))

    tracks = [Trust_Region(f"T{i + 1}", p["cst"], p["E"], p["config"], steps) for i, p in enumerate(peaks)]
    monitor = Lane_Monitor(max(p["E"] for p in peaks))
    simulations = 0
    for iteration in range(1, 40):
        for track in tracks:
            cand = track.sample_uniform(3000, rng)
            pred = surrogate(cand)
            top = cand[np.argsort(-pred)[:50]]
            cand = np.vstack([cand, track.sample_around(top, 1000, rng)])
            pred = surrogate(cand)
            picks = diverse_top(pred, cand / steps, 6, min_steps=0.25 * track.radius)
            sims = true_E(cand[picks])
            simulations += len(picks)
            j = int(np.argmax(sims))
            track.update(iteration, sims[j], cand[picks][j], "rec", predicted_E=float(np.max(pred[picks])))
        best = max(t.centre_E for t in tracks)
        gain = monitor.update(best)
        tracks, removed = Trust_Region.prune(tracks, monitor.best_E, iteration)
        for t, reason in removed:
            print(f"  iteration {iteration}: track {t.track_id} removed: {reason}")
        print(f"iteration {iteration:2d}: best E {best:.4f} (gain {gain:+.2%}), tracks {len(tracks)}, stall {monitor.stall}")
        if monitor.done or not tracks:
            break
    best_track_E = monitor.best_E
    true_max = float(true_E(optimum[None])[0])
    print(f"Best E found {best_track_E:.4f} of the true optimum {true_max:.4f} after {simulations} simulations "
          f"(dataset best {E.max():.4f}).")
    checks.append(("beats the dataset best", best_track_E > E.max()))
    checks.append(("within 1% of the true optimum", best_track_E >= 0.99 * true_max))
    restored = Trust_Region.from_dict(Trust_Region(f"T9", optimum, 1.0, "rec", steps).to_dict())
    checks.append(("to_dict/from_dict round trip", np.allclose(restored.centre, optimum) and restored.radius == Trust_Region.R0))
    m2 = Lane_Monitor.from_dict(monitor.to_dict())
    checks.append(("Lane_Monitor round trip", m2.stall == monitor.stall and m2.best_E == monitor.best_E))
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    sys.exit(0 if all(ok for _, ok in checks) else 1)
