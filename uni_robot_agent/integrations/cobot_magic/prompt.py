"""Cobot Magic twin context; shared providers own all tool instructions."""
from uni_robot_agent.context.prompt import Prompt, register_prompt_builder


@register_prompt_builder("cobot_magic")
def build(context):
    episode = context.episode
    prompt = Prompt(user=context.task or (episode.instruction if episode else ""), system=(
        "Control one Cobot Magic twin: a mobile base with a lift and two 6-DOF Piper arms, "
        "left and right, each ending in a parallel-jaw gripper. Public poses are world metres "
        "and wxyz quaternions; move inputs use fixed-axis XYZ radians. The registered control "
        "site is the pinch point between the jaws, 0.135 m beyond the wrist, not the wrist or "
        "the finger pads. Aperture ranges from 0 (closed) to 0.10 m (open); a grasp closes the "
        "jaws and attaches only when an object lies between them, and a commanded pose does not "
        "prove attachment or placement.\n\n"
        "Cameras are head, left_wrist and right_wrist. The head camera looks down over the "
        "table and is the informative view; in this scene the wrist cameras sit inside the "
        "gripper and return a near-black image, so do not spend turns on them. Captures return "
        "RGB, metric depth and same-frame intrinsics and optical-to-world transforms. Use "
        "capture_scene and locate with method=pixels to turn an image point into a world point; "
        "segmentation is unavailable, so choose points yourself and inspect the returned "
        "overlay. Depth and calibration describe the snapshot only; recapture after any motion "
        "that moves a camera or the objects you measured.\n\n"
        "Work from the current instruction and observations. Approach objects from above with "
        "the jaws open, descend until the pinch point is at the object, close, then lift a few "
        "centimetres and observe whether the object rose with the gripper before transporting "
        "it. Verify the final arrangement from a fresh head-camera image before submitting "
        "completion. The environment owns the official outcome; a finish claim does not set it."
    ))
    if episode:
        prompt = prompt.append(
            system="Environment description:\n" + episode.description.environment.model_dump_json(),
            user="Initial measured state:\n" + episode.state.model_dump_json(),
        )
    return prompt
