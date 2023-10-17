import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from docplex.mp.model import Context, Model


def viz_objects(objects, tables, bounding_rect):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis("equal")
    ax.set_xlim(bounding_rect[0][0] - 0.5, bounding_rect[0][1] + 0.5)
    ax.set_ylim(bounding_rect[1][0] - 0.5, bounding_rect[1][1] + 0.5)

    for cx, cy, dx, dy in tables:
        x, y = cx - dx, cy - dy
        w, h = 2 * dx, 2 * dy
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="g", facecolor="g")
        ax.add_patch(rect)

    colors = ["red", "blue", "yellow", "black"]
    for i, (x, y, r) in enumerate(objects):
        col = colors[i % len(colors)]
        c = plt.Circle((x, y), r, color=col)
        ax.add_artist(c)

    plt.show()
    plt.close("all")


# Detects if lhs <= rhs and saves the result in flag_var
def make_inequality_flag(lhs_var, rhs_var, flag_var, inf_val):
    # if flag_var == 0, we can cheat and satisfy the inequality
    # else we need to satisfy the lhs <= rhs constraint
    return lhs_var <= rhs_var + inf_val * (1 - flag_var)


def repair_objects(
    objects: np.ndarray,
    regions: np.ndarray,
    object_to_region: np.ndarray,
    bounding_rect: np.ndarray,
    inf_val=1000,
):
    """MIP repair to keep the objects within give regions and avoid collisions by moving
    them as little as possible in terms of Euclidean distance.

    Args:
        objects: Array with (x, y, radius) of each object. The collision box is considered
            to be the square around the object.
        regions: Array with (x, y, dx, dy) of each region. Valid area for objects is
            :math:`(x \pm dx, y \pm dy)`.
        object_to_region: Array with allowed regions for each object.
        bounding_rect: [[x_min, x_max], [y_min, y_max]] covering all valid areas.
        inf_val: Value to satisfy certain MIP constraints automatically in some settings.
            Most likely wouldn't need to be changed from the default value. (default
            1000).

    Returns:
        Array with repaired (x, y, radius) of each object.
    """
    context = Context.make_default_context()
    context.cplex_parameters.threads = 1
    with Model(context=context) as mdl:

        mdl.parameters.optimalitytarget.set(3)
        # print(mdl.parameters.optimalitytarget.get())

        pts = []
        for i, (x, y, r) in enumerate(objects):
            pt = [
                mdl.continuous_var(
                    name="pos_{}_x".format(i),
                    lb=bounding_rect[0][0] + r,
                    ub=bounding_rect[0][1] - r,
                ),
                mdl.continuous_var(
                    name="pos_{}_y".format(i),
                    lb=bounding_rect[1][0] + r,
                    ub=bounding_rect[1][1] - r,
                ),
            ]
            pts.append(pt)

        all_flags = []
        object_in_region_vars_list = []
        for i in range(len(pts)):
            x_var = pts[i][0]
            y_var = pts[i][1]
            r = objects[i][2]

            object_in_region_vars = []
            for j, (x, y, dx, dy) in enumerate(regions):

                if object_to_region[i][j]:
                    min_x = x - dx
                    max_x = x + dx
                    min_y = y - dy
                    max_y = y + dy

                    # Create a variable that detects if object i is in region j
                    variable = mdl.integer_var(
                        name="obj_{}_in_reg_{}".format(i, j), lb=0, ub=1
                    )

                    # Create some variable that detect exclusion for object i in region j.
                    is_left = mdl.integer_var(
                        name="obj_{}_is_left_reg_{}".format(i, j), lb=0, ub=1
                    )
                    is_right = mdl.integer_var(
                        name="obj_{}_is_right_reg_{}".format(i, j), lb=0, ub=1
                    )
                    is_above = mdl.integer_var(
                        name="obj_{}_is_above_reg_{}".format(i, j), lb=0, ub=1
                    )
                    is_below = mdl.integer_var(
                        name="obj_{}_is_below_reg_{}".format(i, j), lb=0, ub=1
                    )
                    all_flags.append(is_left)
                    all_flags.append(is_right)
                    all_flags.append(is_above)
                    all_flags.append(is_below)

                    # Bind out of region to the binary exclusion flags.
                    c_left = make_inequality_flag(min_x, x_var, is_left, inf_val=inf_val)
                    c_right = make_inequality_flag(
                        x_var, max_x, is_right, inf_val=inf_val
                    )
                    c_up = make_inequality_flag(y_var, max_y, is_above, inf_val=inf_val)
                    c_down = make_inequality_flag(min_y, y_var, is_below, inf_val=inf_val)

                    # print(min_x <= x_var-r)
                    # print(c_left)
                    # print(x_var + r <= max_x)
                    # print(c_right)

                    mdl.add_constraint(c_left)
                    mdl.add_constraint(c_right)
                    mdl.add_constraint(c_up)
                    mdl.add_constraint(c_down)

                    sum_bools = is_left + is_right + is_above + is_below
                    c_bind = make_inequality_flag(4, sum_bools, variable, inf_val=inf_val)
                    mdl.add_constraint(c_bind)

                    object_in_region_vars.append(variable)

            # print(object_in_region_vars)
            mdl.add_constraint(mdl.sum(object_in_region_vars) >= 1)
            all_flags.extend(object_in_region_vars)
            object_in_region_vars_list.append(object_in_region_vars)

        # Add the constraint that pairwise objects do not overlap.
        q_constraints = []
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j:
                    continue

                x_1, y_1 = pts[i]
                x_2, y_2 = pts[j]
                r_1 = objects[i][2]
                r_2 = objects[j][2]

                left_1 = mdl.integer_var(name="is_{}_left_{}".format(j, i), lb=0, ub=1)
                right_1 = mdl.integer_var(name="is_{}_right_{}".format(j, i), lb=0, ub=1)
                above_1 = mdl.integer_var(name="is_{}_above_{}".format(j, i), lb=0, ub=1)
                below_1 = mdl.integer_var(name="is_{}_below_{}".format(j, i), lb=0, ub=1)

                c_left = make_inequality_flag(
                    x_2 + r_2, x_1 - r_1, left_1, inf_val=inf_val
                )
                c_right = make_inequality_flag(
                    x_1 + r_1, x_2 - r_2, right_1, inf_val=inf_val
                )
                c_up = make_inequality_flag(
                    y_1 + r_1, y_2 - r_2, above_1, inf_val=inf_val
                )
                c_down = make_inequality_flag(
                    y_2 + r_2, y_1 - r_1, below_1, inf_val=inf_val
                )

                mdl.add_constraint(c_left)
                mdl.add_constraint(c_right)
                mdl.add_constraint(c_up)
                mdl.add_constraint(c_down)
                mdl.add_constraint(left_1 + right_1 + above_1 + below_1 >= 1)

        costs = []
        for i, (x, y, r) in enumerate(objects):
            x_1, y_1 = pts[i]
            dx = x_1 - x
            dy = y_1 - y
            costs.append(dx ** 2 + dy ** 2)

        mdl.minimize(mdl.sum(costs))
        solution = mdl.solve()
        # print("cost", solution.get_objective_value())

        new_pts = []
        # Only goals matter for object_in_region_list
        object_in_region_list = np.zeros(3, dtype=int)
        for i in range(len(objects)):
            x = solution.get_value(pts[i][0])
            y = solution.get_value(pts[i][1])
            # print(f"p_{i} = ({x},{y}) vs ({objects[i][0]},{objects[i][1]})")
            new_pts.append([x, y, objects[i][2]])

            if i < 3:
                for j, oir_var in enumerate(object_in_region_vars_list[i]):
                    if solution.get_value(oir_var) == 1:
                        object_in_region_list[i] = j
                        break

        # for cur_var in all_flags:
        #     print(cur_var.name, solution.get_value(cur_var))

        return np.array(new_pts), solution.get_objective_value(), object_in_region_list


def main():
    eps = 1e-6
    regions = np.array(
        [[0.6, 0.05, 0.08 - eps, 0.13 - eps], [0.218, 0.626, 0.08 - eps, 0.13 - eps]]
    )
    object_to_region = np.ones((3, 2), dtype=bool)
    bounding_rect = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    r = np.ones(3) * 0.025 + eps

    obj_x = np.zeros(3)
    obj_y = np.zeros(3)
    objects = np.array(list(zip(obj_x, obj_y, r)))

    new_objs, cost, obj_regions = repair_objects(
        objects, regions, object_to_region, bounding_rect
    )
    print(objects)
    print(new_objs)
    print(cost)
    print(obj_regions)

    viz_objects(objects, regions, bounding_rect)
    viz_objects(new_objs, regions, bounding_rect)


if __name__ == "__main__":
    main()
