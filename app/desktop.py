import pygame
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

pygame.init()

DRAW_WIDTH, DRAW_HEIGHT = 560, 560
WIN_WIDTH, WIN_HEIGHT = DRAW_WIDTH, 660
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

pygame.display.set_caption("Drawing App")
BRUSH_COLOR,BRUSH_SIZE  = (0, 0, 0), 20
GRID_COLOR = (211, 211, 211)

drawing_array = np.zeros((28, 28))
previous_drawings = []
drawing = False
font = pygame.font.SysFont("Roboto", 25)
model = pickle.load(open("../svm.sav", 'rb'))
predicted = "-"

def draw(predicted):
    WIN.fill((255, 255, 255))
    for i in range(28):
        pygame.draw.rect(WIN, GRID_COLOR, (i * 20, 0, 2 , DRAW_HEIGHT))

    for i in range(29):
        pygame.draw.rect(WIN, GRID_COLOR, (0, i * 20, DRAW_WIDTH, 2))


    for rect in previous_drawings:
        pygame.draw.rect(WIN, BRUSH_COLOR, rect)

    info = font.render("SPACE to reset. ENTER to predict.", True, (255, 0, 0))
    prediction = font.render(f"Predicted: {predicted}", True, (255, 0, 0))
    WIN.blit(info, (20, DRAW_HEIGHT + 20))
    WIN.blit(prediction, (20, DRAW_HEIGHT + 40))

def show_digit(digit):
    plt.imshow(digit.reshape((28, 28)), cmap="gray")
    plt.show()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                previous_drawings = []
                predicted = "-"
                drawing_array = np.zeros((28, 28))
            if event.key == pygame.K_RETURN:
                digit = drawing_array.reshape((1,784))
                predicted = model.predict(digit)
                show_digit(digit)
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = pygame.mouse.get_pos()
            if (x < 0 or x >= DRAW_WIDTH) or (y < 0 or y >= DRAW_HEIGHT): continue

            row, col = int(y / 20), int(x / 20)
            drawing_array[row][col] = 16

            rect = pygame.draw.rect(WIN, BRUSH_COLOR, (col * 20, row * 20, BRUSH_SIZE, BRUSH_SIZE))
            previous_drawings.append(rect)

    draw(predicted)
    pygame.display.update()