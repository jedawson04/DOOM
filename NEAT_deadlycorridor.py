import os, math, random
import neat, visualize
import vizdoom as vzd

from time import sleep, time
from random import sample
from tqdm import trange
from statistics import fmean

import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU

global game
num_generations = 10

# Other parameters
frames_per_action = 12
resolution = (30, 45)
episodes_to_watch = 20

save_model = True
load = False
skip_learning = False
watch = True

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
model_savefolder = "./model"

def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    # print(img.flatten(), len(img.flatten()))
    # img = np.expand_dims(img, axis=-1)
    # print(img, img.shape)
    return img.flatten()

def eval_genomes(genomes, config):
    global actions
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        testfits = []
        for i in range(3):
            game.new_episode()
            testfit = 0
            while not game.is_episode_finished():
                buf = game.get_state().screen_buffer
                # print(len(buf))
                screen_buf = preprocess(buf)
                # print(screen_buf)
                action = net.activate(screen_buf)
                # print(action)

                reward = game.make_action(actions[action.index(max(action))])
                testfit += reward
            testfits.append(testfit)
        genome.fitness += fmean(testfits)
        print(genome.fitness)

def initialize_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    assert game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.clear_available_buttons()
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.init()
    print("Doom initialized.")

    return game

def run(config_file):
    # time_start = time()
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    
    winner = p.run(eval_genomes, num_generations)
    stats.save()
    
    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    test(game, net)


def test(game, agent):
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_actions = agent.activate(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_actions.index(max(best_actions))])
            for _ in range(frames_per_action):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)


def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-deadlycorridor')
    # define and initialize global vars
    global game
    game = initialize_game()
    global actions
    # optimal action space
    actions = [[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1]]
    
    run(config_path)


if __name__ == '__main__':
    main()
