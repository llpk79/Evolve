from random import randint, random
from .settings import *
from .brain import Brain


class Blob:
    """An inhabitant of Blobland.

    Just a simple Blob who may, or may not, be able to look up, down, left or right in search of mates.
    Who similarly may, or may not, be able to move up, down, left, or right in pursuit of mates.
    Who possesses a multi-layer perceptron brain that may, or may not, effectively translate sense to motion.
    Sense, movement and brain configuration are determined by an 8-bit integer 'genome'.

    """

    def __init__(self, blobland, genome: int, mutant: bool = False):
        self.genome = f"{genome:08b}"[:8]
        self.blobland = blobland
        self.action_map = {
            0: self.move_up,
            1: self.move_down,
            2: self.move_left,
            3: self.move_right,
        }
        self.sense_map = {
            0: self.look_up,
            1: self.look_down,
            2: self.look_left,
            3: self.look_right,
        }
        self.senses = None
        # self.sense_distance = int(self.genome[3:7])
        self.actions = None
        self.brain = self.create_brain()
        self.pos = None
        self.survived = 0
        self.mated = 0
        self.training_inputs = None
        self.training_labels = []
        self.mutant = mutant
        self.spawn()

    def create_training_data(self) -> None:
        """Make training data for Blobs.
        
        Labels input signals > TRAINING_INPUT_THRESHOLD as 1 else 0
        """
        self.training_inputs = [
            [random() for _ in range(self.brain.n_input)] for _ in range(5)
        ]
        for training_input in self.training_inputs:
            self.training_labels.append(
                [
                    input_ > TRAINING_INPUT_THRESHOLD
                    for input_ in training_input[: self.brain.n_out]
                ]
                + [randint(0, 1) for _ in range(self.brain.n_out - self.brain.n_input)]
            )

    def create_brain(self) -> Brain:
        """Parses genome to create multi-layer perceptron Brain.

        Takes first 4 bits to encode senses according to sense_map.
        Takes last 4 bits to encode movement according to action_map.
        Takes middle 4 bits to encode number of hidden layers in Brain.
        """
        self.senses = [
            self.sense_map[i] for i, x in enumerate(self.genome[:4]) if x == "1"
        ]
        num_inputs = len(self.senses)
        hidden = int(self.genome[2:6], base=2)
        self.actions = [
            self.action_map[i] for i, x in enumerate(self.genome[4:]) if x == "1"
        ]
        num_outputs = len(self.actions)
        return Brain(
            n_input=num_inputs if num_inputs else 1,
            n_hidden=hidden if hidden else 1,
            n_out=num_outputs,
        )

    def find_mate(self) -> None:
        """Look one square up, down, left, and right. Attempt to mate with all."""
        for up_down, left_right in zip([1, -1, 0, 0], [0, 0, 1, -1]):
            adjacent_pos = (self.pos[0] + up_down, self.pos[1] + left_right)
            if adjacent_pos in self.blobland.blobs:
                self.mate(self.blobland.blobs[adjacent_pos])

    def mate(self, other) -> None:
        """Create new genome taking half from self and half from other.

        Mate and mutate according to settings.
        Mutate by flipping one random bit in offspring's genome.
        """
        chance = random()
        if self.survived and chance < MATING_CHANCE:
            mutant = False
            egg = self.genome[:4]
            sperm = other.genome[4:]
            fertilized = int(egg + sperm, base=2)
            if chance < MUTATION_CHANCE:
                # print('Mutation!!')
                mutate = 1 << randint(0, 8)
                fertilized ^= mutate
                mutant = True
            self.mated += 1
            self.blobland.add_blob(
                Blob(self.blobland, genome=fertilized, mutant=mutant)
            )

    def spawn(self) -> None:
        """ Place this Blob in a random free space."""
        while True:
            pos = (randint(0, WORLD_SIZE - 1), randint(0, WORLD_SIZE - 1))
            if self.check_move(pos):
                self.update_position(pos)
                break

    def update_position(self, pos: tuple) -> None:
        """Update position in blobland.blobs.

        Check if current position is in blobland.blobs, if so delete.
        Change self.pos to updated pos.
        Add self to blobland.blobs with updated self.pos.
        """
        if self.pos in self.blobland.blobs:
            del self.blobland.blobs[self.pos]
        self.pos = pos
        self.blobland.blobs[self.pos] = self

    def move_up(self, signal: float) -> None:
        if signal >= WORLD_INPUT_THRESHOLD:
            temp = list(self.pos)
            temp[0] += 1
            temp = tuple(temp)
            if self.check_move(temp):
                self.update_position(temp)

    def move_down(self, signal: float) -> None:
        if signal >= WORLD_INPUT_THRESHOLD:
            temp = list(self.pos)
            temp[0] -= 1
            temp = tuple(temp)
            if self.check_move(temp):
                self.update_position(temp)

    def move_left(self, signal: float) -> None:
        if signal >= WORLD_INPUT_THRESHOLD:
            temp = list(self.pos)
            temp[1] -= 1
            temp = tuple(temp)
            if self.check_move(temp):
                self.update_position(temp)

    def move_right(self, signal: float) -> None:
        if signal >= WORLD_INPUT_THRESHOLD:
            temp = list(self.pos)
            temp[1] += 1
            temp = tuple(temp)
            if self.check_move(temp):
                self.update_position(temp)

    def check_move(self, position) -> bool:
        """Ensure position is free and within Blobland."""
        row, col = position
        return (
            not (row, col) in self.blobland.blobs
            and 0 <= row < WORLD_SIZE
            and 0 <= col < WORLD_SIZE
        )

    def look(self, row_range: range, column_range: range) -> float:
        """Look a direction determined by row_range and column_range.

        Return population of blobs found relative to total population.
        """
        output = 0
        for row in row_range:
            if not 0 <= self.pos[0] + row < WORLD_SIZE:
                continue
            for column in column_range:
                if not 0 <= self.pos[1] + column < WORLD_SIZE:
                    continue
                if (self.pos[0] + row, self.pos[1] + column) in self.blobland.blobs:
                    output += 1
        return output / self.blobland.population

    def look_up(self) -> float:
        row_range = range(SENSE_DISTANCE)
        column_range = range(-SENSE_DISTANCE, SENSE_DISTANCE)
        return self.look(row_range, column_range)

    def look_down(self) -> float:
        row_range = range(-SENSE_DISTANCE, 0)
        column_range = range(-SENSE_DISTANCE, SENSE_DISTANCE)
        return self.look(row_range, column_range)

    def look_left(self) -> float:
        row_range = range(-SENSE_DISTANCE, SENSE_DISTANCE)
        column_range = range(-SENSE_DISTANCE, 0)
        return self.look(row_range, column_range)

    def look_right(self) -> float:
        row_range = range(-SENSE_DISTANCE, SENSE_DISTANCE)
        column_range = range(SENSE_DISTANCE)
        return self.look(row_range, column_range)

    def update(self) -> None:
        """Map sense input signals to actions."""
        input_signals = [sense() for sense in self.senses]
        if input_signals:
            output_signals = self.brain.predict(input_signals)
            for output, output_signal in zip(self.actions, output_signals):
                output(output_signal)

    def train(self) -> None:
        """Train all surviving Blobs with any senses plus any actions."""
        if self.survived and self.senses and self.brain.n_out:
            self.training_inputs, self.training_labels = [], []
            self.create_training_data()
            self.brain.fit(self.training_inputs, self.training_labels)
