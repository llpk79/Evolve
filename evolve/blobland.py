import pandas as pd
import plotly.express as px
from settings import *

from blob import Blob
from collections import defaultdict
from datetime import datetime
from random import randint, sample


class Blobland:
    """A land for Blobs.

    A square grid of size world_size where Blobs may find each other, mate, and perhaps survive to the next epoch.

    """

    def __init__(self):
        self.population = 0
        self.blobs = {}
        self.data = defaultdict(list)
        self.peak = 0
        self.epoch_peak = 0

    def add_blob(self, blob: Blob) -> None:
        """Inhabitants are tracked by position in a dictionary.
        Increment population
        """
        self.blobs[blob.pos] = blob
        self.population += 1
        self.peak = max(self.peak, self.population)
        self.epoch_peak = max(self.epoch_peak, self.population)

    def cull(self, epoch, step) -> None:
        """Remove Blobs not in the safe zone determined by SCENARIO in settings.py.
        Reset population.
        Save data post culling.
        """
        keepers = False
        if SCENARIO == 'interior':
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[0] in range(SAFE_ZONE_SIZE, WORLD_SIZE - SAFE_ZONE_SIZE + 1)
                   and blob.pos[1] in range(SAFE_ZONE_SIZE, WORLD_SIZE - SAFE_ZONE_SIZE + 1)
            }
        if SCENARIO == 'bottom':
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[0] in range(SAFE_ZONE_SIZE + 1)
            }
        if SCENARIO == 'top':
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[0] in range(WORLD_SIZE - SAFE_ZONE_SIZE, WORLD_SIZE + 1)
            }
        if SCENARIO == "left_side":
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[1] in range(SAFE_ZONE_SIZE + 1)
            }
        if SCENARIO == "corner":
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[0] in range(SAFE_ZONE_SIZE + 1)
                   and blob.pos[1] in range(SAFE_ZONE_SIZE + 1)
            }
        if SCENARIO == "sides":
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[1] in range(SAFE_ZONE_SIZE + 1)
                   or blob.pos[1] in range(WORLD_SIZE - SAFE_ZONE_SIZE, WORLD_SIZE + 1)
            }
        if SCENARIO == "right_side":
            keepers = {
                blob.pos: blob
                for blob in self.blobs.values()
                if blob.pos[1] in range(WORLD_SIZE - SAFE_ZONE_SIZE, WORLD_SIZE + 1)
            }
        if not keepers:
            raise ValueError(f"Check SCENARIO in settings.py. You entered {SCENARIO}.")
        self.population = len(keepers)
        self.blobs.clear()
        self.blobs = keepers
        for blob in self.blobs.values():
            self.save_step_data(epoch, step + 1, blob)

    def update(self, epoch: int = 0, start_time: datetime = None) -> None:
        """Reset population.
        If epoch end:
            - increment survival count
            - allow each Blob to mate
            - spawn each Blob
            - print epoch stats to stdout.
        """
        end_time = datetime.now()
        self.population = len(self.blobs)
        repeat_survivors, ultimate_survivor = 0, 0
        ultimate_fucker, mated = 0, 0
        gene_pool = set()
        blobs = list(self.blobs.values())
        for blob in blobs:
            blob.survived += 1
            blob.find_mate()
            blob.spawn()
            mated += blob.mated > 0
            repeat_survivors += blob.survived > 1
            ultimate_survivor = max(ultimate_survivor, blob.survived)
            ultimate_fucker = max(ultimate_fucker, blob.mated)
            gene_pool.add(blob.genome)
        print(
            f"EPOCH: {epoch}\tepoch time {(end_time - start_time).seconds} seconds\nsurviving population {self.population}\tpeak population {self.peak}\tepoch peak {self.epoch_peak}\tsurviving proportion {self.population / self.epoch_peak:.2f}"
        )
        print(
            f"repeats {repeat_survivors}\ttotal mated {mated}\tsurviving gene pool {len(gene_pool)}\noldest survivor {ultimate_survivor}\tUltimate fucker {ultimate_fucker}\n"
        )

    def manage_population(self):
        """Repopulate up to initial population setting.
        If population has more than doubled initial setting, cull to doubled.
        Option to slice population keeping the oldest members, or keep a random selection.
        """
        for _ in range(INITIAL_POPULATION - self.population):
            self.add_blob(Blob(self, randint(0, 255)))

        if self.population >= 2 * INITIAL_POPULATION:
            # self.blobs = dict(
            #     sample(list(self.blobs.items()), INITIAL_POPULATION * 2)  # Equal opportunity
            # )
            self.blobs = dict(
                list(self.blobs.items())[: INITIAL_POPULATION * 2]  # Genetic Domination
            )
        self.population = len(self.blobs)

    def cleanup_epoch(self, epoch: int, step: int, start_time: datetime) -> None:
        """Call end of epoch functions."""
        self.cull(epoch, step)
        self.update(epoch, start_time=start_time)

    def save_step_data(self, epoch: int, step: int, blob: Blob) -> None:
        """Records data from step in data dictionary."""
        if step % PLOT_DATA_SAVE_MOD == 0:
            self.data["epoch"].append(epoch)
            self.data["step"].append(step)
            self.data["x"].append(blob.pos[1])
            self.data["y"].append(blob.pos[0])
            self.data["genome"].append(int(blob.genome, base=2))
            self.data['string_genome'].append(str(int(blob.genome, base=2)))

    def animate_epoch(self, epoch: int) -> None:
        """Create animated scatter plot of epoch steps and histogram of genomes."""
        df = pd.DataFrame.from_dict(self.data)
        df = df[df["epoch"] == epoch]
        scatter = px.scatter(
            data_frame=df,
            x="x",
            y="y",
            animation_frame="step",
            color='string_genome',
            range_x=[-1, WORLD_SIZE + 1],
            range_y=[-1, WORLD_SIZE + 1],
            height=800,
            width=800,
            title=f"Epoch {epoch}",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        if SCENARIO == 'interior':
            scatter.add_shape(
                type='line',
                x0=SAFE_ZONE_SIZE,
                y0=SAFE_ZONE_SIZE,
                x1=SAFE_ZONE_SIZE,
                y1=WORLD_SIZE - SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
            scatter.add_shape(
                type='line',
                x0=SAFE_ZONE_SIZE,
                y0=WORLD_SIZE - SAFE_ZONE_SIZE,
                x1=WORLD_SIZE - SAFE_ZONE_SIZE,
                y1=WORLD_SIZE - SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
            scatter.add_shape(
                type='line',
                x0=WORLD_SIZE - SAFE_ZONE_SIZE,
                y0=WORLD_SIZE - SAFE_ZONE_SIZE,
                x1=WORLD_SIZE - SAFE_ZONE_SIZE,
                y1=SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
            scatter.add_shape(
                type='line',
                x0=SAFE_ZONE_SIZE,
                y0=SAFE_ZONE_SIZE,
                x1=WORLD_SIZE - SAFE_ZONE_SIZE,
                y1=SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
        if SCENARIO == 'bottom':
            scatter.add_shape(
                type='line',
                x0=0,
                y0=SAFE_ZONE_SIZE,
                x1=WORLD_SIZE,
                y1=SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
        if SCENARIO == 'top':
            scatter.add_shape(
                type='line',
                x0=0,
                y0=WORLD_SIZE - SAFE_ZONE_SIZE,
                x1=WORLD_SIZE,
                y1=WORLD_SIZE - SAFE_ZONE_SIZE,
                line=dict(width=1, dash='solid'),
            )
        if SCENARIO == "sides":
            scatter.add_shape(
                type="line",
                x0=SAFE_ZONE_SIZE,
                y0=0,
                x1=SAFE_ZONE_SIZE,
                y1=WORLD_SIZE,
                line=dict(width=1, dash="solid"),
            )
            scatter.add_shape(
                type="line",
                x0=WORLD_SIZE - SAFE_ZONE_SIZE,
                y0=0,
                x1=WORLD_SIZE - SAFE_ZONE_SIZE,
                y1=WORLD_SIZE,
                line=dict(width=1, dash="solid"),
            )
        if SCENARIO == "corner":
            scatter.add_shape(
                type="line",
                x0=SAFE_ZONE_SIZE,
                y0=0,
                x1=SAFE_ZONE_SIZE,
                y1=SAFE_ZONE_SIZE,
                line=dict(width=1, dash="solid"),
            )
            scatter.add_shape(
                type="line",
                x0=0,
                y0=SAFE_ZONE_SIZE,
                x1=SAFE_ZONE_SIZE,
                y1=SAFE_ZONE_SIZE,
                line=dict(width=1, dash="solid"),
            )
        if SCENARIO == "left_side":
            scatter.add_shape(
                type="line",
                x0=SAFE_ZONE_SIZE,
                y0=0,
                x1=SAFE_ZONE_SIZE,
                y1=WORLD_SIZE,
                line=dict(width=1, dash="solid"),
            )
        if SCENARIO == "right_side":
            scatter.add_shape(
                type="line",
                x0=WORLD_SIZE - SAFE_ZONE_SIZE,
                y0=0,
                x1=WORLD_SIZE - SAFE_ZONE_SIZE,
                y1=WORLD_SIZE,
                line=dict(width=1, dash="solid"),
            )
        scatter.update_xaxes(showgrid=False, showticklabels=False, visible=False)
        scatter.update_yaxes(showgrid=False, showticklabels=False, visible=False)
        scatter.show()
        hist = px.histogram(
            data_frame=df[(df["epoch"] == epoch) & (df['step'] == df['step'].max())],
            x="string_genome",
            y="string_genome",
            color="string_genome",
            nbins=df['string_genome'].nunique(),
            histfunc="count",
            height=400,
            width=800,
            title=f"Epoch {epoch} survivors",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        hist.show()
