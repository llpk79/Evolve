# Mess around with these and see what happens.
GENERATIONS = 100
STEPS_PER_WORLD_GENERATION = 150
SCENARIO = 'right_side' # Options: 'corner', 'sides', 'left_side', 'right_side', 'top', 'bottom', 'interior'
SAFE_ZONE_SIZE = 5
INITIAL_POPULATION = 100
OVER_POPULATION_STRATEGY = "random_sample"  # Options: 'random_sample', 'keep_oldest'

# You can mess with these, too. Just be careful.
WORLD_SIZE = 100
STEPS_BETWEEN_TRAINING = 10
LEARNING_EPOCHS = 50
LEARNING_RATE = 1
PEAK_POPULATION = 1.5  # Multiple of INITIAL_POPULATION
MATING_CHANCE = 0.15
MUTATION_CHANCE = 0.001
WORLD_INPUT_THRESHOLD = 0.5
TRAINING_INPUT_THRESHOLD = 0.6
SENSE_DISTANCE = 20
PLOT_DATA_SAVE_MOD = STEPS_PER_WORLD_GENERATION // 50
