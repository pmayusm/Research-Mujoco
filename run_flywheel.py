import time
import mujoco
import mujoco.viewer
import random
import numpy as np

from flywheel_rewards import (
    compute_closeness,
    compute_hit_score,
    compute_miss_score,
    impact_distance_on_face,
)

model = mujoco.MjModel.from_xml_path("flywheel_test.xml")
data = mujoco.MjData(model)


<<<<<<< HEAD

actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel_motor")
yaw_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
spawn_ctrl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_spawn")

hood_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launcher_middle_hood")
spawn_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shooter_spawn")
spawn_local = model.site_pos[spawn_site_id].copy()


def hood_local_to_world(local_pos):
    rotation = data.xmat[hood_id].reshape(3, 3)
    return data.xpos[hood_id] + rotation @ local_pos


def orient_target_toward_shooter(body_id, target_pos, shooter_pos):
    """Align the target's flat face normal (local +Z) toward the shooter."""
    forward = shooter_pos - target_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return

    z_axis = forward / forward_norm
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.flatten())
    model.body_quat[body_id] = quat


HIDDEN_BALL_POS = np.array([0.0, 0.0, -10.0])

def ball_contacts_target():
    for i in range(data.ncon):
        contact = data.contact[i]
        geoms = {contact.geom1, contact.geom2}
        if geoms == {ball_geom_id, target_geom_id}:
            return True
    return False

try:
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "platform_body")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "random_platform")
    target_half_thickness = model.geom_size[target_geom_id][2]
    ball_geom_id = model.body_geomadr[ball_id]
    ball_radius = model.geom_size[ball_geom_id][0]
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z = model.geom_pos[floor_geom_id][2]
    ball_joint_id = model.body_jntadr[ball_id]
    ball_qpos_adr = model.jnt_qposadr[ball_joint_id]
    ball_dof_adr = model.jnt_dofadr[ball_joint_id]

    random_x = random.uniform(-10, 10)
    random_y = random.uniform(-10, 10)
    random_z = random.uniform(2, 10)
    target_pos = np.array([random_x, random_y, random_z])

    model.body_pos[body_id] = target_pos

    mujoco.mj_forward(model, data)
    shooter_pos = hood_local_to_world(spawn_local)
    orient_target_toward_shooter(body_id, target_pos, shooter_pos)
    mujoco.mj_forward(model, data)

    face_normal = data.xmat[body_id].reshape(3, 3)[:, 2]
    alignment = np.dot(face_normal, shooter_pos - target_pos) / np.linalg.norm(shooter_pos - target_pos)
    print(
        f"Spawned target at: X={random_x:.2f}, Y={random_y:.2f}, Z={random_z:.2f} "
        f"(face alignment: {alignment:.3f})"
    )

except ValueError:
    print("Warning: Could not find a body named 'platform_body' in your XML file.")

prev_spawn_ctrl = 0.0
ball_spawned = False
episode_finished = False
final_reward = 0.0
best_miss_distance = None
best_lateral_distance = None
episode_message = ""


def set_ball_collision(enabled):
    mask = 1 if enabled else 0
    model.geom_contype[ball_geom_id] = mask
    model.geom_conaffinity[ball_geom_id] = mask


def hide_ball():
    data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = HIDDEN_BALL_POS
    data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[ball_dof_adr:ball_dof_adr + 6] = 0
    set_ball_collision(False)


def finish_episode(message, reward):
    global episode_finished, final_reward, episode_message
    episode_finished = True
    final_reward = reward
    episode_message = message
    print(f"\n{message}")
    print(f"Total Reward: {final_reward:.3f}")


def spawn_ball_in_shooter():
    global ball_spawned, episode_finished, final_reward, best_miss_distance, best_lateral_distance, episode_message

    spawn_pos = hood_local_to_world(spawn_local)
    data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = spawn_pos
    data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[ball_dof_adr:ball_dof_adr + 6] = 0
    set_ball_collision(True)
    ball_spawned = True
    episode_finished = False
    final_reward = 0.0
    best_miss_distance = None
    best_lateral_distance = None
    episode_message = ""
    mujoco.mj_forward(model, data)
    print("\n[BALL SPAWN] Ball placed at shooter spawn point.")


hide_ball()
mujoco.mj_forward(model, data)

data.ctrl[actuator_id] = -1
data.ctrl[yaw_motor_id] = 0
data.ctrl[spawn_ctrl_id] = 0
=======
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel_motor")
yaw_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
>>>>>>> 29c09aa10ae4094d9548d2ae41260f19acaa359c

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation started. Press Ctrl+C in the terminal to exit.")

    while viewer.is_running():
        step_start = time.time()

<<<<<<< HEAD
        viewer.sync()

        spawn_ctrl = data.ctrl[spawn_ctrl_id]
        if spawn_ctrl > 0.5 and prev_spawn_ctrl <= 0.5:
            spawn_ball_in_shooter()
        prev_spawn_ctrl = spawn_ctrl

        if not ball_spawned:
            hide_ball()
            print("Push ball_spawn past 0.5 to start the episode.", end="\r")
        else:
            ball_pos = np.array(data.xpos[ball_id])
            slab_pos = np.array(data.xpos[body_id])
            slab_matrix = data.xmat[body_id].reshape(3, 3)
            plane_normal = slab_matrix[:, 2]
            miss_distance, lateral_distance, _ = compute_closeness(
                ball_pos, slab_pos, plane_normal, target_half_thickness
            )

            if not episode_finished:
                if best_miss_distance is None:
                    best_miss_distance = miss_distance
                    best_lateral_distance = lateral_distance
                else:
                    best_miss_distance = min(best_miss_distance, miss_distance)
                    best_lateral_distance = min(best_lateral_distance, lateral_distance)

                miss_score = compute_miss_score(best_lateral_distance)
                hit_potential = compute_hit_score(lateral_distance)

            if not episode_finished:
                print(
                    f"Aim: {lateral_distance:.2f}m | Best aim: {best_lateral_distance:.2f}m | "
                    f"Miss score: {miss_score:.3f} | Hit if now: {hit_potential:.3f}",
                    end="\r",
                )
            else:
                print(f"Episode Finished. {episode_message} Total Reward: {final_reward:.3f}", end="\r")

            pre_step_ball_pos = ball_pos.copy()
=======
        
        data.ctrl[actuator_id] = -1
        data.ctrl[yaw_motor_id] = 0
>>>>>>> 29c09aa10ae4094d9548d2ae41260f19acaa359c

        mujoco.mj_step(model, data)

        if ball_spawned and not episode_finished:
            ball_pos = np.array(data.xpos[ball_id])
            slab_pos = np.array(data.xpos[body_id])
            plane_normal = data.xmat[body_id].reshape(3, 3)[:, 2]
            signed_distance = np.dot(ball_pos - slab_pos, plane_normal)
            hit_front_face = signed_distance <= target_half_thickness + ball_radius

            if ball_contacts_target() or hit_front_face:
                impact_distance = impact_distance_on_face(
                    pre_step_ball_pos, slab_pos, plane_normal, target_half_thickness
                )
                finish_episode(
                    f"[IMPACT] Ball hit the target! Distance to center: {impact_distance:.2f} m",
                    compute_hit_score(impact_distance),
                )
            elif ball_pos[2] <= floor_z + ball_radius:
                finish_episode(
                    "[MISS] Ball hit the floor.",
                    compute_miss_score(best_lateral_distance),
                )

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            #mj run_flywheel.py
