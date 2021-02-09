import imageio
import numpy as np
from tqdm import tqdm
from camera import Camera
from figures import Cube, Hexahedron
from copy import deepcopy
from direction_predictor import predict_direction

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


# camera intrinsics
F = 1024
X_size = 1024
Y_size = 1024

random_seed = 42


def render(camera, figure, save_path=None):
    figure.project(camera)

    image = Image.new("RGB", (X_size, Y_size), 'white')
    draw = ImageDraw.Draw(image)

    # draw.point([(figure.projected_points[pid][0], figure.projected_points[pid][1]) for pid in range(8)], 'black')

    # for e in figure.edges:
    #     p1 = figure.projected_points[e[0]]
    #     p2 = figure.projected_points[e[1]]
    #     draw.line([p1[0], p1[1], p2[0], p2[1]], 'black', 3)

    for face in figure.sorted_faces:
        draw.polygon([(figure.projected_points[pid][0], figure.projected_points[pid][1]) for pid in face], fill='white', outline='black')

    if save_path is not None:
        image.save(save_path)

    return np.array(image)


def run_cube_movement():
    camera = Camera()
    camera.set_K_elements(X_size/2, Y_size/2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Cube([0, 0, 2560], 1024)
    figure.rotate_around_axis([0, 1, 0], np.pi/4)
    figure.rotate_around_axis([1, 0, 0], np.pi/4)

    move_direction = [0, 10, 150]
    rendered_frames = []

    for _ in tqdm(range(300)):
        rendered_frames.append(render(camera, figure))
        figure.move(move_direction)

    imageio.mimwrite('video.mp4', np.array(rendered_frames), fps=30)


def run_hex_movement():
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


def run_hex_generation():
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


def save_hex_creenshot():
    camera = Camera()
    camera.set_K_elements(X_size / 2, Y_size / 2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Hexahedron([0, 0, 2560], 1024, random_seed)
    figure.generate(random_seed)
    figure.generate(random_seed)
    figure.rotate_around_axis([1, 0, 0], np.pi/4)
    figure.rotate_around_axis([0, 1, 0], np.pi/4)

    render(camera, figure, "hex.png")


def save_cube_creenshot():
    camera = Camera()
    camera.set_K_elements(X_size / 2, Y_size / 2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Cube([0, 0, 2560], 1024)
    # figure.rotate_around_axis([0, 1, 0], np.pi/4)
    # figure.rotate_around_axis([1, 0, 0], np.pi/4)

    render(camera, figure, "cube.png")


def predict_cube_direction():
    camera = Camera()
    camera.set_K_elements(X_size / 2, Y_size / 2, F)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))
    figure = Cube([0, 0, 2560], 1024)
    figure.rotate_around_axis([0, 1, 0], np.pi / 4)
    figure.rotate_around_axis([1, 0, 0], np.pi / 4)

    move_dir = [50, 150, 50]

    figure.project(camera)
    coords1 = deepcopy(figure.projected_points)

    figure.move(move_dir)
    figure.project(camera)
    coords2 = deepcopy(figure.projected_points)

    predicted_dir = predict_direction(coords1, coords2, camera.K)
    dir_norm_by_z = [move_dir[0] / move_dir[2], move_dir[1]/ move_dir[2], 1]
    print(f"direction: {dir_norm_by_z}\npredicted: {predicted_dir}")


def predict_cube_direction2():
    camera = Camera()
    camera.set_K_elements(320, 240, 320)
    camera.set_R_euler_angles([0, 0, 0])
    camera.set_t(np.array([[0], [0], [0]]))

    coords1 = np.array([[[151], [153]], [[488], [97]]])
    coords2 = np.array([[[439], [298]], [[523], [295]]])
    predicted_dir = predict_direction(coords1, coords2, camera.K)
    print(f"predicted: {predicted_dir}")




if __name__ == '__main__':
    predict_cube_direction2()
