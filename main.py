import imageio
import numpy as np
from tqdm import tqdm
from camera import Camera
from figures import Cube, Hexahedron

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


# camera intrinsics
F = 1024
X_size = 1024
Y_size = 1024

random_seed = 42


def render(camera, figure):
    figure.project(camera)

    image = Image.new("RGB", (X_size, Y_size), 'white')
    draw = ImageDraw.Draw(image)

    # draw.point([(figure.projected_points[pid][0], figure.projected_points[pid][1]) for pid in range(8)], 'black')

    # for e in figure.edges:
    #     p1 = figure.projected_points[e[0]]
    #     p2 = figure.projected_points[e[1]]
    #     draw.line([p1[0], p1[1], p2[0], p2[1]], 'black', 3)

    for face in figure.sorted_faces:
        draw.polygon([(figure.projected_points[pid][0], figure.projected_points[pid][1]) for pid in face], fill='orange', outline='black')

    return np.array(image)


def run_cube_simulation():
    camera = Camera()
    camera.set_K_elements(X_size/2, Y_size/2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Cube([-1500, 0, 2560], 1024)

    move_direction = [10, 0, 0]
    rendered_frames = []

    for _ in tqdm(range(300)):
        rendered_frames.append(render(camera, figure))
        figure.move(move_direction)

    imageio.mimwrite('video.mp4', np.array(rendered_frames), fps=30)


def run_hex_simulation():
    camera = Camera()
    camera.set_K_elements(X_size / 2, Y_size / 2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Hexahedron([-1500, 0, 2560], 1024, random_seed)
    figure.generate(random_seed)
    figure.generate(random_seed)

    move_direction = [10, 0, 0]
    rendered_frames = []

    for _ in tqdm(range(300)):
        rendered_frames.append(render(camera, figure))
        figure.move(move_direction)

    imageio.mimwrite('video.mp4', np.array(rendered_frames), fps=30)


def run_hex_simulation2():
    camera = Camera()
    camera.set_K_elements(X_size / 2, Y_size / 2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Hexahedron([0, 0, 2560], 1024, random_seed)

    seed = 100
    rendered_frames = []

    for i in tqdm(range(300)):
        rendered_frames.append(render(camera, figure))
        if i % 30 == 0:
            seed += 1
            figure.generate(seed)

    imageio.mimwrite('video.mp4', np.array(rendered_frames), fps=30)


if __name__ == '__main__':
    run_hex_simulation()
