import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def get_circuit(filename="input.txt"):
    points = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            try:
                coords = [float(i.strip()) for i in line.split(",")]
            except:
                print("File incorrect!")
            points.append(coords)
    return points


def mesh_step_calc(circuit, box, count):
    # TODO for circle
    extremes_raw = []
    for element in circuit:
        extremes_raw.append(element)
    extremes = np.array(extremes_raw)
    boundaries = [[min(extremes[:, i]), max(extremes[:, i])] for i in range(len(extremes[0]))]
    boundaries = [boundaries[i] if boundaries[i][1] - boundaries[i][0] >= box[i] else [(boundaries[i][1] - boundaries[i][0])/2 - box[i]/2, (boundaries[i][1] - boundaries[i][0])/2 + box[i]/2] for i in range(len(boundaries))]
    print(boundaries)
    ranges = [boundaries[i][1] - boundaries[i][0] for i in range(len(boundaries))]
    steps = [i / count for i in ranges]
    if 0 in steps:
        steps.remove(0)
    step = min(steps)
    return step, boundaries


def get_segments(circuit):
    # TODO for circle
    point_pairs = []
    for i in range(len(circuit) - 1):
        pair = [circuit[i], circuit[i + 1]]
        point_pairs.append(pair)
    return np.array(point_pairs)


def generate_mesh(mesh_basis):
    mesh = np.meshgrid(*mesh_basis)
    return mesh


def mesh_solver(segments, mesh, current, z=None):
    x_plane = mesh[0]
    y_plane = mesh[1]
    if z is None:
        z_plane = mesh[2]
    else:
        z_plane = [z]


    solved_vect_mesh = [[[0 for i in range(len(z_plane))] for j in range(len(y_plane))] for k in range(len(x_plane))]
    solved_magnitude_mesh = [[[0 for i in range(len(z_plane))] for j in range(len(y_plane))] for k in range(len(x_plane))]

    for high in range(len(z_plane)):
        for col in range(len(y_plane)):
            for row in range(len(x_plane)):
                solved_segments_in_point = []
                for segment in segments:
                    solved_point = BSL_solver(segment, [x_plane[row], y_plane[col], z_plane[high]], current)
                    solved_segments_in_point.append(solved_point)
                solved_vect_mesh[col][row][high] = np.nansum(solved_segments_in_point, axis=0)
                solved_magnitude_mesh[col][row][high] = np.linalg.norm(solved_vect_mesh[col][row][high])
        print("Solved ", high+1, " slices of ", len(z_plane))
    return np.array(solved_vect_mesh), np.array(solved_magnitude_mesh)


# calculate distance, s-vector
def point2line_in_space(line, point):
    s = np.array([line[1][i] - line[0][i] for i in range(len(point))])
    r = np.linalg.norm(np.cross(line[0] - point, s)) / np.linalg.norm(s)
    return r, s


# Biot–Savart law for infinite line
def BSL_solver(line, point, current):
    mu_0 = 1.25663706212 * 10 ** (-6)

    r, s = point2line_in_space(line, point)

    A = np.array([line[0][i] - line[1][i] for i in range(len(point))])
    B = np.array([line[0][i] - point[i] for i in range(len(point))])
    C = np.array([line[1][i] - line[0][i] for i in range(len(point))])
    D = np.array([line[1][i] - point[i] for i in range(len(point))])

    cos_a1 = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    cos_a2 = -np.dot(C, D) / (np.linalg.norm(C) * np.linalg.norm(D))
    B_scalar = current * mu_0 / (4 * np.pi * r) * np.round((np.nansum(np.array([cos_a1, -cos_a2]))), 10)

    R = np.cross(A, B)
    R_norm = R / np.linalg.norm(R)
    B_vector = R_norm * B_scalar

    return B_vector


def plot_results(circuit, mesh_basis, mesh, value):
    x_circuit = np.array(circuit)[:, 0]
    y_circuit = np.array(circuit)[:, 1]
    z_circuit = np.array(circuit)[:, 2]
    fig, ax = plt.subplots()
    ax.plot(x_circuit, y_circuit, linewidth=2, color='red')

    ax.contour(mesh_basis[0], mesh_basis[1], value, levels=100)

    # ax.scatter(mesh[0], mesh[1],  s=1, color='green')

    # for i, txt in enumerate(np.array(value).ravel()):
    #     ax.annotate('{0:.2e}'.format(txt), (mesh[0].ravel()[i], mesh[1].ravel()[i]))

    fig.set_figwidth(12)
    fig.set_figheight(9)
    plt.show()


def plot_3D(circuit, mesh_basis, mesh, value):
    x_circuit = np.array(circuit)[:, 0]
    y_circuit = np.array(circuit)[:, 1]
    z_circuit = np.array(circuit)[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(20, 210)


    # ax.quiver(mesh[0], mesh[1], mesh[2], u, v, w, length=1)

    # ax.scatter(mesh[0][0].ravel(), mesh[1][0].ravel(), np.array(value).ravel())
    ax.plot(x_circuit, y_circuit, z_circuit, linewidth=2, color='red')
    ax.contour(mesh_basis[0], mesh_basis[1], value, 100, zdir='z', linestyles="solid", offset=0)

    fig.set_figwidth(12)
    fig.set_figheight(9)

    plt.show()


def save_data(data):
    line = ""
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(len(data[0][0])):
                line += str(data[i][j][k]) + ', '
            line += '\n'
        line += '\n'
    with open('data.txt', 'w') as file:
        file.write(line)


def main_func(count, z=None):
    circuit = get_circuit("input.txt")
    # current = float(input("Current through the circuit, in amperes: "))
    # mesh_points_count = float(input("Mesh points count, in points per shortest axis: "))
    solver_box = [1, 1, 1]
    current = 1
    mesh_points_count = count
    mesh_step, mesh_boundaries = mesh_step_calc(circuit, solver_box, mesh_points_count)
    print("Mesh step ", mesh_step)
    mesh_basis = [np.arange(mesh_boundaries[i][0], mesh_boundaries[i][1] + mesh_step, mesh_step) for i in
                  range(len(mesh_boundaries))]
    # mesh_basis = [i if len(i) > 0 else np.append(i, 0) for i in mesh_basis] # for the existence of at least one element
    # print(mesh_basis)
    mesh = generate_mesh(mesh_basis)
    circuit_segments = get_segments(circuit)
    vect_solve, magnitude_solve = mesh_solver(circuit_segments, mesh_basis, current,z)

    return circuit, mesh_basis, mesh, magnitude_solve, vect_solve

if __name__ == "__main__":
    circuit, mesh_basis, mesh, magnitude_solve, vect_solve = main_func(50, 0)

    plot_results(circuit, mesh_basis, mesh, magnitude_solve[0])
    plot_3D(circuit, mesh_basis, mesh, magnitude_solve[0])

    # save_da