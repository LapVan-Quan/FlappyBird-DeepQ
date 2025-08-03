import pygame


class HumanAgent:
    def __init__(self, show_screen=False):
        self.show_screen = show_screen

    def choose_action(self, state, action_table):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return action_table['quit_game']
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return action_table['jump']

        # no keystroke, so do nothing
        return action_table['do_nothing']

    def receive_after_action_observation(self, state, action_table):
        pass
