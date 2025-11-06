# Reference target trajectory message
# PC → Teensy

float32 theta_ref_now         # Current target angle (rad)
float32 dtheta_ref_now        # Current target angular velocity (rad/s), 0.0 for static
float32[] theta_ref_preview   # Future target angles (K elements, typically 32 or 64)
