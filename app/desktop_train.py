import pygame
import numpy as np
import matplotlib.pyplot as plt

pygame.init()

DRAW_WIDTH, DRAW_HEIGHT = 560, 560
WIN_WIDTH, WIN_HEIGHT = DRAW_WIDTH, 760
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

pygame.display.set_caption("Drawing App")
BRUSH_COLOR,BRUSH_SIZE  = (0, 0, 0), 20
GRID_COLOR = (211, 211, 211)

conf = ""
drawing_array = np.zeros((28, 28))
current_num = 0
# running first time or resteting use 21 and 22 line, otherwise 19 and 20
count = np.load("count.npy")
test_data = np.load("test_data.npy")
# count = np.array([99])
# test_data = np.zeros((100, 784))
print(test_data)
test_numbers = []
for i in range(10):
    for j in range(10):
        test_numbers.append(i)

current_num = test_numbers[count[0]]
previous_drawings = []
drawing = False
font = pygame.font.SysFont("Roboto", 25)

def draw(number, conf):
    WIN.fill((255, 255, 255))
    for i in range(28):
        pygame.draw.rect(WIN, GRID_COLOR, (i * 20, 100, 2 , DRAW_HEIGHT))

    for i in range(29):
        pygame.draw.rect(WIN, GRID_COLOR, (0, 100 + i * 20, DRAW_WIDTH, 2))


    for rect in previous_drawings:
        pygame.draw.rect(WIN, BRUSH_COLOR, rect)

    info = font.render("SPACE to reset. ENTER to confirm.", True, (255, 0, 0))
    num_to_draw = font.render(f"Draw number {number}.", True, (0, 255, 0))
    curr_count = font.render(f"Images drown {count}", True, (0, 255, 0))
    confiramtion = font.render(conf, True, (255, 0, 0))
    WIN.blit(info, (20, DRAW_HEIGHT + 120))
    WIN.blit(num_to_draw, (20, 80))
    WIN.blit(curr_count, (350, 80))
    WIN.blit(confiramtion, (20, DRAW_HEIGHT + 140))

def show_digit(digit):
    plt.imshow(digit.reshape((28, 28)), cmap="gray")
    plt.show()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            np.save("test_data.npy", test_data)
            np.save("count.npy", count)
            quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                previous_drawings = []
                drawing_array = np.zeros((28, 28))
                conf = ""
            if event.key == pygame.K_RETURN:
                digit = drawing_array.reshape((1,784))
                test_data[count] = digit
                conf = f"Current image for {current_num} has been saved."
                if not count[0] > 99:
                    if not count[0] >= 99:
                        count[0] += 1
                    current_num = test_numbers[count[0]]
                previous_drawings = []
                drawing_array = np.zeros((28, 28))
            if event.key == pygame.K_z:
                count[0] -= 1
                if not count[0] > 99:
                    current_num = test_numbers[count[0]]
            if event.key == pygame.K_s:
                np.save("test_data.npy", test_data)
                np.save("count.npy", count)
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = pygame.mouse.get_pos()
            if (x < 0 or x >= DRAW_WIDTH) or (y < 100 or y >= DRAW_HEIGHT + 100): continue

            row, col = int(y / 20), int(x / 20)
            drawing_array[int((y - 100) / 20)][col] = 16

            rect = pygame.draw.rect(WIN, BRUSH_COLOR, (col * 20, row * 20, BRUSH_SIZE, BRUSH_SIZE))
            previous_drawings.append(rect)

    draw(current_num,conf)
    pygame.display.update()