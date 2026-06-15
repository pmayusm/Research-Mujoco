import time
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("flywheel_test.xml")
data = mujoco.MjData(model)


actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel_joint")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation started. Press Ctrl+C in the terminal to exit.")
    
    while viewer.is_running():
        step_start = time.time()

        
        data.ctrl[actuator_id] = -1

     
        mujoco.mj_step(model, data)

       
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        #mj run_flywheel.py