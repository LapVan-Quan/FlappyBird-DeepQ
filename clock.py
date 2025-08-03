import pygame

class ClockWrapper:
    def __init__(self, show_screen=False, frame_rate=30):
        self.show_screen = show_screen
        self.frame_rate = frame_rate

        if self.show_screen:
            self.clock = pygame.time.Clock()
        else:
            self.clock_counter = 0

    def current_time(self):
        if self.show_screen:
            return pygame.time.get_ticks()
        else:
            return self.clock_counter

    def tick(self):
        if self.show_screen:
            self.clock.tick(self.frame_rate)  # frame rate
        else:
            self.clock_counter += 1000 / self.frame_rate