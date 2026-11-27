import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


SRC_XML = "assets/my_buggy.xml"
DST_XML = "assets/my_buggy_v2_lidar.xml"
NUM_RAYS = 16
RING_RADIUS = 0.045
CUTOFF = 2.0


def build_lidar_xml(src_path: str = SRC_XML, dst_path: str = DST_XML):
    tree = ET.parse(src_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    assert worldbody is not None, "No <worldbody> found in buggy XML."
    chassis = worldbody.find("body")
    assert chassis is not None, "No chassis <body> found in buggy XML."

    # Place ring slightly above existing rf_front site to avoid wheel/caster self-hits
    ring_z = 0.068
    for s in chassis.findall("site"):
        if s.attrib.get("name") == "rf_front" and "pos" in s.attrib:
            parts = [float(v) for v in s.attrib["pos"].split()]
            if len(parts) == 3:
                ring_z = max(parts[2] + 0.015, 0.062)
            break

    for s in list(chassis.findall("site")):
        if s.attrib.get("name", "").startswith("lidar_"):
            chassis.remove(s)

    for g in list(chassis.findall("geom")):
        if g.attrib.get("name") == "lidar_turret":
            chassis.remove(g)

    ET.SubElement(
        chassis,
        "geom",
        {
            "name": "lidar_turret",
            "type": "cylinder",
            "pos": f"0 0 {ring_z:.4f}",
            "size": "0.028 0.010",
            "rgba": "0.10 0.72 0.95 1",
            "contype": "0",
            "conaffinity": "0",
            "mass": "0.02",
        },
    )

    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")

    for rf in list(sensor_elem.findall("rangefinder")):
        if rf.attrib.get("name", "").startswith("lidar_"):
            sensor_elem.remove(rf)

    # 16 evenly spaced rays starting at 0 deg (+X Front), going CCW every 22.5 deg
    for i in range(NUM_RAYS):
        ang = 2.0 * math.pi * i / NUM_RAYS
        cx, cy = math.cos(ang), math.sin(ang)
        sx, sy = RING_RADIUS * cx, RING_RADIUS * cy
        name = f"lidar_{i:02d}"

        ET.SubElement(
            chassis,
            "site",
            {
                "name": name,
                "pos": f"{sx:.4f} {sy:.4f} {ring_z:.4f}",
                "zaxis": f"{cx:.6f} {cy:.6f} 0.0",
                "size": "0.004",
                "rgba": "0.1 0.9 1.0 0.6",
            },
        )
        ET.SubElement(
            sensor_elem,
            "rangefinder",
            {
                "name": name,
                "site": name,
                "cutoff": f"{CUTOFF:.2f}",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(dst_path, encoding="utf-8", xml_declaration=False)
    print(f"Saved {NUM_RAYS}-ray LIDAR buggy to: {dst_path} (ring_z={ring_z:.4f} m)")


def verify_raycast(dst_path: str = DST_XML):
    # Inner box wall faces at +/- 1.00 m around origin
    scene_xml = """
    <mujoco model="lidar_verify">
      <option timestep="0.005"/>
      <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
        <geom name="w_east"  type="box" pos=" 1.025  0.000 0.15" size="0.025 1.05 0.15"/>
        <geom name="w_west"  type="box" pos="-1.025  0.000 0.15" size="0.025 1.05 0.15"/>
        <geom name="w_north" type="box" pos=" 0.000  1.025 0.15" size="1.05 0.025 0.15"/>
        <geom name="w_south" type="box" pos=" 0.000 -1.025 0.15" size="1.05 0.025 0.15"/>
      </worldbody>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(scene_xml)
    bot_spec = mujoco.MjSpec.from_file(dst_path)
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.035], quat=[1.0, 0.0, 0.0, 0.0])
    if hasattr(spec, "attach"):
        spec.attach(bot_spec, prefix="bot/", frame=frame)
    else:
        frame.attach_body(bot_spec.worldbody.first_body(), "bot/", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(15):
        mujoco.mj_step(model, data)

    self_hits = []
    print("\n--- 16-Ray Planar LIDAR Verification (2.0m x 2.0m Box) ---")
    for i in range(NUM_RAYS):
        ang_deg = i * (360.0 / NUM_RAYS)
        ang_rad = math.radians(ang_deg)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"bot/lidar_{i:02d}")
        assert sid >= 0, f"Sensor bot/lidar_{i:02d} not found!"
        adr = int(model.sensor_adr[sid])
        val = float(data.sensordata[adr])

        # Exact expected distance from ring site (radius 0.045m) to +/-1.0m box wall
        denom = max(abs(math.cos(ang_rad)), abs(math.sin(ang_rad)))
        expected = (1.0 / denom) - RING_RADIUS
        err = abs(val - expected)
        status = "OK" if (val > 0.20 and err < 0.05) else "CHECK"
        if val < 0.20:
            self_hits.append((i, val))
        print(f"  lidar_{i:02d} ({ang_deg:5.1f} deg): measured={val:.3f}m | expected={expected:.3f}m | {status}")

    if self_hits:
        print(f"\nWARNING: Potential chassis self-hit on rays {self_hits}. Raise ring_z or RING_RADIUS.")
    else:
        print("\nPASS: All 16 planar LIDAR rays clear the chassis with zero self-hits.")


if __name__ == "__main__":
    build_lidar_xml()
    verify_raycast()
